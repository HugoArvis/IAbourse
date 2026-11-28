import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import MAX_PREDICTED_RETURN

EPISODE_LENGTH = 252  # 1 trading year


class TradingEnv(gym.Env):
    """
    Multi-ticker trading environment for PPO fine-tuning.

    Observation (8 dims) — compact vector derived from supervised model
    predictions:
      [pred_5d_norm, pred_21d_norm, confidence, vix_z_norm,
       mkt_drawdown_norm, stress, current_position, episode_drawdown]

    Using the compact representation (vs raw 60-day feature window)
    because LightGBM/CatBoost already compress the full window into a
    single alpha estimate. The RL agent learns WHEN and HOW MUCH to act
    on those predictions.

    Action (1 dim): target position in [-1, 1], i.e. -100% to +100% of
    the portfolio.
    """

    metadata = {"render_modes": []}

    def __init__(self, ticker_data: dict):
        super().__init__()
        # ticker → DataFrame with columns: pred_5d, pred_21d, vix_z,
        # market_drawdown, close
        self.ticker_data = ticker_data
        self.tickers = [
            t for t, df in ticker_data.items() if len(df) >= EPISODE_LENGTH + 5
        ]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._df = None
        self._start = 0
        self._step = 0
        self._position = 0.0
        self._portfolio = 1.0
        self._peak = 1.0
        self._returns: list[float] = []

    def _obs(self, idx: int) -> np.ndarray:
        row = self._df.iloc[idx]
        pred_5d = float(row.get("pred_5d", 0.0))
        pred_21d = float(row.get("pred_21d", 0.0))
        confidence = float(
            np.clip(abs(pred_21d) / MAX_PREDICTED_RETURN, 0.0, 1.0)
        )
        vix_z = float(row.get("vix_z", 0.0))
        mkt_dd = float(row.get("market_drawdown", 0.0))
        stress = float(np.clip(vix_z / 3.0 + abs(mkt_dd) * 5.0, 0.0, 1.0))
        ep_dd = float((self._peak - self._portfolio) / max(self._peak, 1e-8))
        return np.array(
            [
                pred_5d / MAX_PREDICTED_RETURN,  # ~[-1, 1]
                pred_21d / MAX_PREDICTED_RETURN,  # ~[-1, 1]
                confidence,  # [0, 1]
                vix_z / 3.0,  # ~[-1, 1]
                mkt_dd / 0.5,  # ~[-1, 0]
                stress,  # [0, 1]
                self._position,  # [-1, 1]
                -ep_dd,  # [-1, 0]
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ticker = self.tickers[self.np_random.integers(len(self.tickers))]
        self._df = self.ticker_data[ticker]
        max_start = len(self._df) - EPISODE_LENGTH - 1
        self._start = int(self.np_random.integers(0, max(max_start, 1)))
        self._step = 0
        self._position = 0.0
        self._portfolio = 1.0
        self._peak = 1.0
        self._returns = []
        return self._obs(self._start), {}

    def step(self, action):
        new_pos = float(np.clip(action[0], -1.0, 1.0))
        delta = abs(new_pos - self._position)

        cur = self._start + self._step
        nxt = cur + 1
        done = self._step + 1 >= EPISODE_LENGTH or nxt >= len(self._df)
        safe = min(nxt, len(self._df) - 1)

        c0 = float(self._df.iloc[cur].get("close", 1.0))
        c1 = float(self._df.iloc[safe].get("close", c0))
        daily_ret = (c1 / c0 - 1.0) if c0 > 0 else 0.0

        # Portfolio update: use old position for today's return, then
        # move to new position
        tx_cost = delta * 0.001
        pf_ret = self._position * daily_ret - tx_cost
        self._position = new_pos
        self._portfolio *= 1.0 + pf_ret
        self._peak = max(self._peak, self._portfolio)
        self._returns.append(pf_ret)
        self._step += 1

        from rl.reward_functions import compute_reward

        drawdown = (self._peak - self._portfolio) / max(self._peak, 1e-8)
        reward = compute_reward(
            self._returns, drawdown, delta, is_terminal=done
        )

        return self._obs(safe), float(reward), done, False, {}
