import json
import os
import pickle

import numpy as np
import pandas as pd
from loguru import logger

from config import (
    CRISIS_PERIODS,
    CRISIS_SAMPLE_WEIGHT,
    MODEL_DIR,
    TEMPORAL_WEIGHT_HALF_LIFE,
)
from data.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS

# Maps horizon → (regression target column, checkpoint filename)
_HORIZON_MAP = {
    "5d": ("relative_return", "lgbm_5d.pkl"),
    "21d": ("relative_return_21d", "lgbm_21d.pkl"),
}


class LGBMModel:
    """
    LightGBM regressor predicting forward relative return vs SP500.
    Positive output = stock expected to outperform SP500 over the horizon.
    Negative output = expected underperformance.

    Position sizing is handled by strategy_scorer.py using the raw return value.
    horizon: "5d" (weekly), "21d" (monthly, primary)
    """

    def __init__(self, model=None, horizon: str = "5d"):
        self._model = model
        self._horizon = horizon
        if horizon not in _HORIZON_MAP:
            raise ValueError(f"horizon must be one of {list(_HORIZON_MAP)}")

    @property
    def _target_col(self) -> str:
        return _HORIZON_MAP[self._horizon][0]

    @property
    def _ckpt(self) -> str:
        return os.path.join(MODEL_DIR, _HORIZON_MAP[self._horizon][1])

    @classmethod
    def _load_best_params(cls, horizon: str) -> dict | None:
        path = os.path.join(MODEL_DIR, f"lgbm_best_params_{horizon}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    @classmethod
    def _make_regressor(cls, custom_params: dict | None = None):
        import lightgbm as lgb

        base = dict(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="huber",  # robust to outlier returns
            alpha=1.0,  # Huber loss threshold
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        if custom_params:
            base.update(custom_params)
        return lgb.LGBMRegressor(**base)

    @classmethod
    def train(cls, features: pd.DataFrame, horizon: str = "5d") -> "LGBMModel":
        """Train on forward relative return (regression target)."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError(
                "lightgbm not installed — run: pip install lightgbm"
            )

        target_col = _HORIZON_MAP[horizon][0]
        df = features.copy().dropna(subset=[target_col])
        df = df.sort_index()

        X = df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
        y = df[target_col].astype(np.float32).values

        # Temporal sample weights
        sample_weight = np.ones(len(df), dtype=np.float32)
        if TEMPORAL_WEIGHT_HALF_LIFE is not None:
            dates = pd.to_datetime(df.index).to_series()
            days_from_end = (dates.max() - dates).dt.days.values
            decay = np.log(2) / TEMPORAL_WEIGHT_HALF_LIFE
            sample_weight = np.exp(-decay * days_from_end).astype(np.float32)

        # Crisis oversampling
        idx = pd.to_datetime(df.index)
        for start, end in CRISIS_PERIODS:
            mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
            sample_weight[mask] *= CRISIS_SAMPLE_WEIGHT

        feat_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        best_params = cls._load_best_params(horizon)
        if best_params:
            logger.info(f"LightGBM [{horizon}] using tuned hyperparameters")
        model = cls._make_regressor(custom_params=best_params)
        model.fit(
            feat_df,
            y,
            sample_weight=sample_weight,
            categorical_feature=CATEGORICAL_FEATURES,
        )
        logger.info(
            f"LightGBM [{horizon}] trained — {model.n_estimators_} trees, "
            f"target={target_col}, samples={len(feat_df)}"
        )
        return cls(model, horizon=horizon)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = path or self._ckpt
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"LightGBM [{self._horizon}] saved to {path}")

    @classmethod
    def load_latest(cls, horizon: str = "5d") -> "LGBMModel | None":
        inst = cls(horizon=horizon)
        if not os.path.exists(inst._ckpt):
            return None
        try:
            with open(inst._ckpt, "rb") as f:
                model = pickle.load(f)
            logger.info(f"LightGBM [{horizon}] loaded from {inst._ckpt}")
            return cls(model, horizon=horizon)
        except Exception as e:
            logger.error(f"Failed to load LightGBM [{horizon}]: {e}")
            return None

    def predict_latest(self, features: pd.DataFrame) -> float:
        """Return predicted forward relative return for the most recent row.
        Positive = expected outperformance vs SP500, negative = underperformance.
        """
        if self._model is None:
            return 0.0
        try:
            row = (
                features[FEATURE_COLUMNS]
                .iloc[[-1]]
                .fillna(0)
                .astype(np.float32)
            )
            return float(self._model.predict(row)[0])
        except Exception as e:
            logger.error(f"LightGBM [{self._horizon}] prediction failed: {e}")
            return 0.0

    def evaluate(self, features: pd.DataFrame) -> dict:
        """Direction accuracy, Spearman IC, MAE and RMSE on a provided DataFrame."""
        if self._model is None:
            return {}
        try:
            from scipy import stats
            from sklearn.metrics import mean_absolute_error

            df = features.dropna(subset=[self._target_col])
            X = df[FEATURE_COLUMNS].fillna(0).astype(np.float32)
            y_true = df[self._target_col].astype(np.float32).values
            y_pred = self._model.predict(X).astype(np.float32)

            dir_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())
            ic, _ = stats.spearmanr(y_pred, y_true)
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            return {
                "direction_accuracy": dir_acc,
                "spearman_ic": float(ic) if not np.isnan(ic) else 0.0,
                "mae": mae,
                "rmse": rmse,
            }
        except Exception as e:
            logger.error(f"LightGBM [{self._horizon}] evaluation failed: {e}")
            return {}
