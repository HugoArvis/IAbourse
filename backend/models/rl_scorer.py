import os
import numpy as np
import pandas as pd
from loguru import logger
from config import MODEL_DIR, MAX_PREDICTED_RETURN


class RLScorer:
    """
    Inference wrapper for the trained PPO agent.

    Uses MultiHorizonScorer (LightGBM + CatBoost) to get base predictions,
    then feeds a compact 8-dim observation to the PPO agent which decides
    the final position, having learned WHEN and HOW MUCH to act.

    Falls back gracefully to MultiHorizonScorer if no PPO checkpoint exists.
    """

    _CKPT = os.path.join(MODEL_DIR, "ppo_agent.zip")
    _CKPT_BEST = os.path.join(MODEL_DIR, "best_model.zip")

    def __init__(self):
        self._ppo = None
        self._base = None   # MultiHorizonScorer

    def load(self) -> "RLScorer":
        from models.strategy_scorer import MultiHorizonScorer
        self._base = MultiHorizonScorer().load()

        ckpt = self._CKPT if os.path.exists(self._CKPT) else (
            self._CKPT_BEST if os.path.exists(self._CKPT_BEST) else None)
        if ckpt is None:
            logger.info("No PPO checkpoint found — RLScorer will use base scorer only")
            return self

        try:
            from stable_baselines3 import PPO
            self._ppo = PPO.load(ckpt)
            logger.info(f"PPO agent loaded from {ckpt}")
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            self._ppo = None
        return self

    @property
    def is_ready(self) -> bool:
        return self._base is not None and self._base.is_ready

    @property
    def rl_active(self) -> bool:
        return self._ppo is not None

    def score(self, features: pd.DataFrame,
              current_position_pct: float = 0.0,
              episode_drawdown: float = 0.0) -> dict:
        """
        Returns the same dict structure as MultiHorizonScorer.score().
        If the PPO checkpoint is loaded, it overrides the position decision.
        """
        # Always run the base supervised scorer (for predictions + reporting)
        result = self._base.score(features)

        if self._ppo is None:
            return result

        # Build 8-dim observation for PPO
        last = features.iloc[-1]
        pred_5d = result["predicted_return_5d"] / 100.0   # % → fraction
        pred_21d = result["predicted_return_21d"] / 100.0
        confidence = result["confidence"]
        stress = result["stress"]
        vix_z = float(last.get("vix_z", 0.0))
        mkt_dd = float(last.get("market_drawdown", 0.0))

        obs = np.array([
            np.clip(pred_5d / MAX_PREDICTED_RETURN, -3.0, 3.0),
            np.clip(pred_21d / MAX_PREDICTED_RETURN, -3.0, 3.0),
            confidence,
            np.clip(vix_z / 3.0, -3.0, 3.0),
            np.clip(mkt_dd / 0.5, -2.0, 0.0),
            stress,
            np.clip(current_position_pct / 100.0, -1.0, 1.0),
            -np.clip(episode_drawdown, 0.0, 1.0),
        ], dtype=np.float32)

        try:
            action, _ = self._ppo.predict(obs, deterministic=True)
            position_pct = float(np.clip(action[0], -1.0, 1.0)) * 100.0
            score = round(5.0 + (position_pct / 100.0) * 5.0, 2)

            from models.strategy_scorer import _strategy_text
            strategy = _strategy_text(pred_21d, confidence, stress)

            result = {**result, "score": score, "position_pct": round(position_pct, 1),
                      "strategy": f"[RL] {strategy}"}
        except Exception as e:
            logger.error(f"PPO predict failed: {e}")

        return result
