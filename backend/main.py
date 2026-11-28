import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from alpaca_connector import AlpacaConnector
from config import (
    PRIMARY_TICKER,
    WS_MARKET_HOURS_INTERVAL,
    WS_OFF_HOURS_INTERVAL,
)
from scheduler import TradingScheduler
from ws_manager import WSManager

ws_manager = WSManager()
scheduler = TradingScheduler()
alpaca = AlpacaConnector()

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, "data", "cache")
_STATUS_FILE = os.path.join(_CACHE_DIR, "model_status.json")
_SIGNAL_FILE = os.path.join(_CACHE_DIR, "last_signal.json")

_state = {
    "model_status": "stopped",
    "start_time": None,
    "last_signal": None,
}


# ── State persistence ─────────────────────────────────────────────────


def _save_status():
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_STATUS_FILE, "w") as f:
            json.dump({"model_status": _state["model_status"]}, f)
    except Exception as e:
        logger.error(f"Failed to save status: {e}")


def _save_signal(signal: dict):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_SIGNAL_FILE, "w") as f:
            json.dump(signal, f, default=str)
    except Exception as e:
        logger.error(f"Failed to save signal: {e}")


def _load_persisted_state():
    if os.path.exists(_STATUS_FILE):
        try:
            with open(_STATUS_FILE) as f:
                saved = json.load(f)
                _state["model_status"] = saved.get("model_status", "stopped")
        except Exception:
            pass
    if os.path.exists(_SIGNAL_FILE):
        try:
            with open(_SIGNAL_FILE) as f:
                _state["last_signal"] = json.load(f)
        except Exception:
            pass


# ── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading backend starting up")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(_HERE, "models", "checkpoints"), exist_ok=True)

    _load_persisted_state()

    if _state["model_status"] == "active":
        scheduler.start()
        _state["start_time"] = datetime.now(timezone.utc)
        logger.info(
            "Scheduler auto-restarted (was active before server restart)"
        )

    yield
    scheduler.stop()
    logger.info("Trading backend shut down")


app = FastAPI(title="Trading AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REST endpoints ────────────────────────────────────────────────────


@app.get("/api/status")
def get_status():
    uptime = ""
    if _state["start_time"]:
        delta = datetime.now(timezone.utc) - _state["start_time"]
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        uptime = f"{hours}h {rem // 60}m"
    last = _state.get("last_signal") or {}
    return {
        "model_status": _state["model_status"],
        "uptime": uptime,
        "last_signal_at": last.get("timestamp", None),
        "last_strategy": last.get("strategy", "—"),
    }


@app.get("/api/portfolio")
def get_portfolio():
    try:
        return alpaca.get_portfolio()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/trades")
def get_trades():
    try:
        return alpaca.get_trade_history()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/metrics")
def get_metrics():
    try:
        return alpaca.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/prices/{ticker}")
def get_prices(ticker: str, days: int = 90):
    try:
        import pandas as pd

        path = os.path.join(_CACHE_DIR, f"{ticker}_prices.parquet")
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404, detail=f"No price data for {ticker}"
            )
        df = pd.read_parquet(path).tail(days)
        df.columns = [c.lower() for c in df.columns]
        return [
            {
                "time": (
                    str(row.Index.date())
                    if hasattr(row.Index, "date")
                    else str(row.Index)
                ),
                "open": round(float(row.open), 2),
                "high": round(float(row.high), 2),
                "low": round(float(row.low), 2),
                "close": round(float(row.close), 2),
            }
            for row in df.itertuples()
            if all(hasattr(row, c) for c in ("open", "high", "low", "close"))
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/history")
def get_portfolio_history():
    try:
        if not alpaca._client:
            return [
                {
                    "time": str(__import__("datetime").date.today()),
                    "value": 100_000.0,
                }
            ]
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        hist = alpaca._client.get_portfolio_history(
            GetPortfolioHistoryRequest(period="1M", timeframe="1D")
        )
        timestamps = hist.timestamp or []
        equities = hist.equity or []
        result = [
            {
                "time": str(
                    __import__("datetime").datetime.utcfromtimestamp(t).date()
                ),
                "value": round(float(e), 2),
            }
            for t, e in zip(timestamps, equities)
            if e and e > 0
        ]
        if not result:
            result = [
                {
                    "time": str(__import__("datetime").date.today()),
                    "value": 100_000.0,
                }
            ]
        return result
    except Exception as e:
        logger.error(f"portfolio history failed: {e}")
        return [
            {
                "time": str(__import__("datetime").date.today()),
                "value": 100_000.0,
            }
        ]


@app.post("/api/model/start")
def model_start():
    if _state["model_status"] == "active":
        return {"status": "already_active"}
    scheduler.start()
    _state["model_status"] = "active"
    _state["start_time"] = datetime.now(timezone.utc)
    _save_status()
    logger.info("Model started")
    return {"status": "started"}


@app.post("/api/model/stop")
def model_stop():
    scheduler.stop()
    _state["model_status"] = "stopped"
    _save_status()
    logger.info("Model stopped")
    return {"status": "stopped"}


@app.post("/api/model/run-now")
async def model_run_now():
    """Trigger inference in the background and return immediately.

    The dashboard is updated via WebSocket.
    """
    if _state["model_status"] != "active":
        raise HTTPException(
            status_code=400, detail="Model must be started first"
        )

    async def _run():
        try:
            from data.feature_engineering import build_features
            from data.fetch_prices import fetch_latest_prices

            logger.info("Manual run-now triggered")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, fetch_latest_prices)
            features = await loop.run_in_executor(None, build_features)
            result = await loop.run_in_executor(
                None, alpaca.run_inference_and_trade, features
            )
            if result:
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                _state["last_signal"] = result
                _save_signal(result)
                logger.info(
                    f"run-now complete: score={result.get('score')} "
                    f"strategy={result.get('strategy')}"
                )
        except Exception as e:
            logger.error(f"run-now failed: {e}")

    asyncio.create_task(_run())
    return {
        "status": "running",
        "detail": (
            "Inference started — dashboard will update in ~30s via WebSocket"
        ),
    }


@app.post("/api/model/kill")
async def model_kill(body: dict):
    if not body.get("confirm"):
        raise HTTPException(status_code=400, detail="confirm=true required")
    scheduler.stop()
    _state["model_status"] = "stopped"
    _save_status()
    try:
        alpaca.liquidate_all()
        logger.warning("KILL SWITCH activated — all positions liquidated")
    except Exception as e:
        logger.error(f"Kill switch liquidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "killed", "message": "All positions liquidated"}


# ── WebSocket ─────────────────────────────────────────────────────────


def _is_market_hours() -> bool:
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and 9 <= now.hour < 16


def _build_live_payload() -> dict:
    last = _state.get("last_signal") or {}
    horizons = last.get("horizons", {})
    portfolio = {}
    try:
        portfolio = alpaca.get_portfolio()
    except Exception:
        pass
    return {
        "type": "live_update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": last.get("ticker", PRIMARY_TICKER),
        "score": last.get("score", 5.0),
        "confidence": last.get("confidence", 0.0),
        "position_pct": last.get("position_pct", 0),
        "predicted_return": last.get("predicted_return", 0.0),
        "predicted_return_5d": last.get("predicted_return_5d", 0.0),
        "predicted_return_21d": last.get("predicted_return_21d", 0.0),
        "stress": last.get("stress", 0.0),
        "strategy": last.get("strategy", "No inference run yet"),
        "weekly": horizons.get("weekly", {"predicted_return_pct": 0.0}),
        "monthly": horizons.get("monthly", {"predicted_return_pct": 0.0}),
        "portfolio_value": portfolio.get("total_value", 100_000.0),
        "pnl_today_pct": portfolio.get("pnl_today_pct", 0.0),
        "pnl_total_pct": portfolio.get("pnl_total_pct", 0.0),
        "sharpe_rolling": last.get("sharpe_rolling", 0.0),
        "drawdown_current": last.get("drawdown_current", 0.0),
        "model_status": _state["model_status"],
        "last_signal_at": last.get("timestamp", None),
    }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            interval = (
                WS_MARKET_HOURS_INTERVAL
                if _is_market_hours()
                else WS_OFF_HOURS_INTERVAL
            )
            payload = _build_live_payload()
            await ws_manager.send_personal(json.dumps(payload), websocket)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ── Static files ──────────────────────────────────────────────────────
# MUST be mounted last so WebSocket + API routes take precedence.
_DIST = os.path.join(_HERE, "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="static")
