from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import (
    RETRAIN_DAY_OF_WEEK,
    RETRAIN_HOUR,
    RETRAIN_MINUTE,
    SCHEDULE_HOUR,
    SCHEDULE_MINUTE,
    SCHEDULE_TIMEZONE,
)


def _daily_job():
    """Main daily trading decision pipeline at 15:30 NY."""
    logger.info("Daily trading job started")
    try:
        # Import here to avoid circular imports at startup
        from data.feature_engineering import build_features
        from data.fetch_news import fetch_and_embed_news
        from data.fetch_prices import fetch_latest_prices

        logger.info("Fetching latest prices and news...")
        fetch_latest_prices()
        fetch_and_embed_news()
        features = build_features()
        logger.info(
            "Features built:"
            f" {features.shape if hasattr(features, 'shape') else 'ok'}"
        )

        # Inference + order placement handled by alpaca_connector
        # (model must already be trained and loaded)
        from datetime import datetime, timezone

        import main as _main  # update shared state so WebSocket picks it up

        from alpaca_connector import AlpacaConnector

        connector = AlpacaConnector()
        result = connector.run_inference_and_trade(features)
        if result:
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            _main._state["last_signal"] = result
            _main._save_signal(result)
    except Exception as e:
        logger.error(f"Daily job failed: {e}")


def _weekly_retrain_job():
    """Sunday evening retraining.

    Keeps the models calibrated to the current regime.
    """
    logger.info("Weekly retraining job started")
    try:
        from data.feature_engineering import build_features
        from data.fetch_news import fetch_and_embed_news
        from data.fetch_prices import fetch_all_prices
        from models.catboost_model import CatBoostModel
        from models.lgbm_model import LGBMModel

        logger.info("Fetching latest data for retraining...")
        fetch_all_prices(force=False)
        fetch_and_embed_news()
        features = build_features()

        for horizon in ("5d", "21d"):
            lgbm = LGBMModel.train(features, horizon=horizon)
            lgbm.save()
            cb = CatBoostModel.train(features, horizon=horizon)
            cb.save()

        logger.info(
            "Weekly retraining complete — LightGBM + CatBoost updated for all"
            " horizons"
        )
    except Exception as e:
        logger.error(f"Weekly retraining failed: {e}")


class TradingScheduler:
    def __init__(self):
        self._scheduler = BackgroundScheduler()
        self._running = False

    def start(self):
        if self._running:
            return
        trigger = CronTrigger(
            hour=SCHEDULE_HOUR,
            minute=SCHEDULE_MINUTE,
            timezone=SCHEDULE_TIMEZONE,
        )
        self._scheduler.add_job(
            _daily_job,
            trigger,
            id="daily_trade",
            replace_existing=True,
            misfire_grace_time=300,
            coalesce=True,
        )

        retrain_trigger = CronTrigger(
            day_of_week=RETRAIN_DAY_OF_WEEK,
            hour=RETRAIN_HOUR,
            minute=RETRAIN_MINUTE,
            timezone=SCHEDULE_TIMEZONE,
        )
        self._scheduler.add_job(
            _weekly_retrain_job,
            retrain_trigger,
            id="weekly_retrain",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,
        )

        self._scheduler.start()
        self._running = True
        logger.info(
            "Scheduler started — daily trade at"
            f" {SCHEDULE_HOUR}:{SCHEDULE_MINUTE:02d}, weekly retrain on"
            f" {RETRAIN_DAY_OF_WEEK.upper()}"
            f" {RETRAIN_HOUR}:{RETRAIN_MINUTE:02d} {SCHEDULE_TIMEZONE}"
        )

    def stop(self):
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")
