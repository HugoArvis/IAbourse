import os
from datetime import datetime, timezone

from loguru import logger

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, EQUITY_TICKERS, MAX_OPEN_POSITIONS,
    MAX_PER_SECTOR, SECTOR_MAP, STOP_LOSS_PCT,
)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — running in stub mode")


class AlpacaConnector:
    def __init__(self):
        self._client = None
        self._start_value = 100_000.0
        self._trades: list[dict] = []
        if _ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_API_KEY != "your_key_here":
            self._client = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=True,
            )
            logger.info("Alpaca paper trading client initialized")
        else:
            logger.warning("Alpaca client not configured — paper trade mode (stub)")

    def get_portfolio(self) -> dict:
        if self._client:
            try:
                account = self._client.get_account()
                equity = float(account.equity)
                last_equity = float(account.last_equity)
                pnl_today = (equity - last_equity) / last_equity * 100 if last_equity else 0
                pnl_total = (equity - self._start_value) / self._start_value * 100
                positions = []
                for pos in self._client.get_all_positions():
                    positions.append({
                        "ticker": pos.symbol,
                        "qty": float(pos.qty),
                        "market_value": float(pos.market_value),
                        "unrealized_pnl": float(pos.unrealized_pl),
                    })
                return {
                    "total_value": equity,
                    "pnl_today_pct": round(pnl_today, 2),
                    "pnl_total_pct": round(pnl_total, 2),
                    "positions": positions,
                }
            except Exception as e:
                logger.error(f"get_portfolio failed: {e}")
        return {
            "total_value": 100_000.0,
            "pnl_today_pct": 0.0,
            "pnl_total_pct": 0.0,
            "positions": [],
        }

    def get_trade_history(self) -> list[dict]:
        if self._client:
            try:
                from alpaca.trading.enums import QueryOrderStatus
                from alpaca.trading.requests import GetOrdersRequest
                orders = self._client.get_orders(
                    GetOrdersRequest(status=QueryOrderStatus.ALL)
                )
                return [
                    {
                        "id": str(o.id),
                        "ticker": o.symbol,
                        "side": o.side.value,
                        "qty": float(o.qty or 0),
                        "filled_avg_price": float(o.filled_avg_price or 0),
                        "filled_at": str(o.filled_at),
                        "status": o.status.value,
                    }
                    for o in orders
                ]
            except Exception as e:
                logger.error(f"get_trade_history failed: {e}")
        return self._trades

    def get_metrics(self) -> dict:
        portfolio = self.get_portfolio()
        base = {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "n_trades": 0, "total_value": portfolio["total_value"],
        }
        if not self._client:
            return base
        try:
            import numpy as np
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            # Pull up to 1 month of daily equity
            hist = self._client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1M", timeframe="1D")
            )
            equity = [e for e in (hist.equity or []) if e and e > 0]
            if len(equity) < 3:
                return {**base, "n_trades": self._count_trades()}

            equity = np.array(equity, dtype=float)
            daily_returns = np.diff(equity) / equity[:-1]

            # Sharpe (annualised, daily returns)
            mean_r = daily_returns.mean()
            std_r = daily_returns.std() + 1e-9
            sharpe = float(mean_r / std_r * np.sqrt(252))

            # Sortino (downside deviation only)
            neg = daily_returns[daily_returns < 0]
            sortino = float(mean_r / (neg.std() + 1e-9) * np.sqrt(252)) if len(neg) > 1 else 0.0

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / (peak + 1e-9)
            max_dd = float(dd.max())

            # Win rate from closed orders
            n_trades, n_wins = self._count_trades(return_wins=True)
            win_rate = (n_wins / n_trades) if n_trades > 0 else 0.0

            return {
                "sharpe": round(sharpe, 3),
                "sortino": round(sortino, 3),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 4),
                "n_trades": n_trades,
                "total_value": portfolio["total_value"],
            }
        except Exception as e:
            logger.error(f"get_metrics failed: {e}")
            return base

    def _count_trades(self, return_wins: bool = False):
        try:
            orders = self._client.get_orders() if self._client else []
            filled = [o for o in orders if o.status.value == "filled"]
            if not return_wins:
                return len(filled)
            wins = sum(
                1 for o in filled
                if o.side.value == "buy" and float(o.filled_avg_price or 0) > 0
            )
            return len(filled), wins
        except Exception:
            return (0, 0) if return_wins else 0

    def place_order(self, ticker: str, side: str, notional: float):
        if not self._client:
            logger.info(f"[STUB] Would place {side} order for {ticker} notional=${notional:,.0f}")
            self._trades.append({
                "ticker": ticker,
                "side": side,
                "notional": notional,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            return
        try:
            # Cancel any open orders for this ticker to avoid wash-trade rejection
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            open_orders = self._client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
            )
            for o in open_orders:
                self._client.cancel_order_by_id(o.id)
                logger.info(f"Cancelled open order {o.id} for {ticker} before new {side}")

            if side == "sell":
                # Alpaca does not allow fractional notional short orders.
                # Interpret "sell" as "close any existing long position".
                try:
                    self._client.close_position(ticker)
                    logger.info(f"Closed position for {ticker}")
                except Exception:
                    logger.info(f"No open position to close for {ticker}")
                return

            req = MarketOrderRequest(
                symbol=ticker,
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            logger.info(f"Order submitted: {order.id} buy {ticker} ${notional:,.0f}")
        except Exception as e:
            logger.error(f"Order placement failed for {ticker}: {e}")

    def liquidate_all(self):
        if self._client:
            self._client.close_all_positions(cancel_orders=True)
            logger.warning("All positions closed via Alpaca")
        else:
            logger.warning("[STUB] Would liquidate all positions")

    def _check_stop_losses(self) -> set[str]:
        """Close any position down more than STOP_LOSS_PCT. Returns set of closed tickers."""
        closed = set()
        if not self._client:
            return closed
        try:
            for pos in self._client.get_all_positions():
                plpc = float(pos.unrealized_plpc or 0)
                if plpc < STOP_LOSS_PCT:
                    logger.warning(
                        f"Stop-loss triggered: {pos.symbol} unrealized={plpc:.1%} "
                        f"< threshold {STOP_LOSS_PCT:.0%}"
                    )
                    self.place_order(pos.symbol, "sell", 0)
                    closed.add(pos.symbol)
        except Exception as e:
            logger.error(f"Stop-loss check failed: {e}")
        return closed

    def _select_with_sector_constraint(self, candidates: list) -> list:
        """Pick top MAX_OPEN_POSITIONS candidates with at most MAX_PER_SECTOR per sector."""
        sector_count: dict[str, int] = {}
        selected = []
        for c in candidates:
            sector = SECTOR_MAP.get(c["ticker"], "other")
            if sector_count.get(sector, 0) < MAX_PER_SECTOR:
                selected.append(c)
                sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(selected) == MAX_OPEN_POSITIONS:
                break
        return selected

    def _get_episode_drawdown(self, portfolio: dict) -> float:
        current_value = portfolio.get("total_value", self._start_value)
        return max(0.0, 1.0 - current_value / self._start_value)

    def _get_ticker_position(self, ticker: str, portfolio: dict) -> float:
        """Returns current position as % of total portfolio value for one ticker."""
        total_value = max(portfolio.get("total_value", self._start_value), 1)
        for p in portfolio.get("positions", []):
            if p.get("ticker") == ticker:
                return (p.get("market_value", 0) / total_value) * 100.0
        return 0.0

    def run_inference_and_trade(self, features) -> dict:
        """Score all equity tickers, buy top N, close stale positions."""
        logger.info("Running multi-ticker inference pipeline...")
        try:
            from models.rl_scorer import RLScorer
            from risk_manager import RiskManager

            portfolio = self.get_portfolio()
            total_value = portfolio["total_value"]
            ep_drawdown = self._get_episode_drawdown(portfolio)

            scorer = RLScorer().load()
            if not scorer.is_ready:
                logger.warning("No trained models found — skipping inference")
                return {}

            logger.info(
                f"Using {'RL+supervised' if scorer.rl_active else 'supervised-only'} scorer "
                f"on {len(EQUITY_TICKERS)} tickers"
            )

            risk = RiskManager()
            if risk.check_global_stop_loss(total_value, self._start_value):
                self.liquidate_all()
                return {}

            # Close positions that hit the per-position stop-loss before re-scoring
            closed_by_stop = self._check_stop_losses()

            # ── Score every equity ticker ─────────────────────────────────────
            candidates = []
            for ticker in EQUITY_TICKERS:
                ticker_features = features[features["ticker"] == ticker]
                if ticker_features.empty:
                    continue
                current_pos_pct = self._get_ticker_position(ticker, portfolio)
                try:
                    result = scorer.score(
                        ticker_features,
                        current_position_pct=current_pos_pct,
                        episode_drawdown=ep_drawdown,
                    )
                    score, position_pct, should_trade = risk.validate_order(
                        ticker=ticker,
                        confidence=result["confidence"],
                        score=result["score"],
                        position_pct=result["position_pct"],
                        portfolio_value=total_value,
                        start_value=self._start_value,
                        avg_volume=5_000_000,
                    )
                    result.update({
                        "ticker": ticker,
                        "score": score,
                        "position_pct": position_pct,
                        "should_trade": should_trade,
                        "current_position": current_pos_pct,
                        "portfolio_value": total_value,
                    })
                    candidates.append(result)
                except Exception as e:
                    logger.error(f"Scoring failed for {ticker}: {e}")

            if not candidates:
                logger.warning("No valid signals produced")
                return {}

            # ── Pick top N long signals with sector constraint ────────────────
            sorted_buys = sorted(
                [c for c in candidates if c["should_trade"] and c["position_pct"] > 0],
                key=lambda x: x.get("predicted_return_21d", 0.0),
                reverse=True,
            )
            buy_candidates = self._select_with_sector_constraint(sorted_buys)

            selected_tickers = {c["ticker"] for c in buy_candidates}
            held_tickers = {p["ticker"] for p in portfolio.get("positions", [])}

            # Close stale positions (skip those already closed by stop-loss)
            for ticker in held_tickers - selected_tickers - closed_by_stop:
                logger.info(f"Closing stale position: {ticker}")
                self.place_order(ticker, "sell", 0)

            # Buy / rebalance: allocate portfolio evenly across MAX_OPEN_POSITIONS slots
            per_slot = total_value / MAX_OPEN_POSITIONS
            for c in buy_candidates:
                notional = round(abs(c["position_pct"] / 100.0) * per_slot, 2)
                if notional >= 1.0:
                    logger.info(
                        f"BUY {c['ticker']}  score={c['score']:.2f}  "
                        f"ret_21d={c.get('predicted_return_21d', 0):+.2f}%  "
                        f"conf={c['confidence']:.3f}  ${notional:,.0f}"
                    )
                    self.place_order(c["ticker"], "buy", notional)

            # Return top signal for the dashboard
            top = buy_candidates[0] if buy_candidates else max(
                candidates, key=lambda x: x.get("predicted_return_21d", 0.0)
            )
            logger.info(
                f"Top signal: {top['ticker']}  score={top['score']:.2f}  "
                f"ret_5d={top.get('predicted_return_5d', 0):+.2f}%  "
                f"ret_21d={top.get('predicted_return_21d', 0):+.2f}%  "
                f"confidence={top['confidence']:.3f}  → {top.get('strategy', '')}"
            )
            return top

        except Exception as e:
            logger.error(f"Inference pipeline error: {e}")
            return {}
