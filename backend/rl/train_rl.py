"""
RL training — PPO fine-tuning on top of LightGBM/CatBoost base
predictions.

Architecture:
  - LightGBM (21d) + CatBoost (5d) predictions are pre-computed for
    every historical row, compressing the 60-day feature window into two
    floats.
  - The PPO agent receives a compact 8-dim observation and learns WHEN
    and HOW MUCH to act on those predictions, taking into account regime
    stress and current portfolio state.
  - Action space: target position in [-1, 1] (i.e., -100% to +100%).

Exit criterion (CLAUDE.md Phase 2):
  Sharpe > 1.0 and max drawdown < 20% on out-of-sample validation.

Usage:
    py train.py   →  select profile 9 (full) or 10 (quick test)
"""

import os
import sys

import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    MODEL_DIR,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LR,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
)
from data.feature_engineering import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)
from rl.callbacks import build_callbacks  # noqa: E402
from rl.reward_functions import _episode_sharpe  # noqa: E402
from rl.trading_env import EPISODE_LENGTH, TradingEnv  # noqa: E402

# ── Data preparation ──────────────────────────────────────────────────


def prepare_rl_data(features: pd.DataFrame) -> tuple[dict, dict]:
    """
    1. 60/20/20 temporal split (train / val / test — test unused here).
    2. Pre-compute LightGBM-21d and CatBoost-5d predictions for every
       row. This avoids calling models inside the hot training loop.
    3. Return (train_dict, val_dict): ticker → DataFrame.
    """
    from models.catboost_model import CatBoostModel
    from models.lgbm_model import LGBMModel
    from training.tune_hyperparams import three_way_split

    train_df, val_df, _, train_cut, val_cut = three_way_split(features)

    lgbm21 = LGBMModel.load_latest("21d")
    cb5 = CatBoostModel.load_latest("5d")

    if lgbm21 is None:
        raise RuntimeError(
            "LGBM 21d model not found — run train.py profile 1 or 5 first"
        )

    logger.info(
        f"RL data: train up to {train_cut.date()} ({len(train_df):,} rows) | "
        f"val {train_cut.date()} → {val_cut.date()} ({len(val_df):,} rows)"
    )

    def _add_preds(df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS].fillna(0)
        for c in CATEGORICAL_FEATURES:
            if c in X.columns:
                X[c] = X[c].astype(int)
        df = df.copy()
        df["pred_21d"] = lgbm21._model.predict(X)
        df["pred_5d"] = cb5._model.predict(X) if cb5 else np.zeros(len(df))
        return df

    def _to_dict(df: pd.DataFrame) -> dict:
        out = {}
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_index()
            if len(grp) >= EPISODE_LENGTH + 5:
                out[ticker] = _add_preds(grp)
        return out

    train_dict = _to_dict(train_df)
    val_dict = _to_dict(val_df)
    logger.info(f"RL tickers — train: {len(train_dict)}, val: {len(val_dict)}")
    return train_dict, val_dict


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_rl(model: PPO, val_dict: dict, n_episodes: int = 30) -> dict:
    """
    Run n_episodes on the validation set.
    Returns mean Sharpe, worst max drawdown, and mean episode return.
    """
    env = TradingEnv(val_dict)
    sharpes, drawdowns, ep_returns = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        sharpe = _episode_sharpe(env._returns)
        dd = (env._peak - env._portfolio) / max(env._peak, 1e-8)
        sharpes.append(sharpe)
        drawdowns.append(dd)
        ep_returns.append(env._portfolio - 1.0)

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "worst_drawdown": float(np.max(drawdowns)),
        "mean_return": float(np.mean(ep_returns)),
        "n_episodes": n_episodes,
    }


# ── Training ──────────────────────────────────────────────────────────


def train_rl(
    features: pd.DataFrame, total_timesteps: int = PPO_TOTAL_TIMESTEPS
) -> PPO:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("  Preparing RL data (pre-computing model predictions)...")
    train_dict, val_dict = prepare_rl_data(features)
    print(
        f"  Train tickers: {len(train_dict)}  |  Val tickers: {len(val_dict)}"
    )

    # DummyVecEnv: works on all platforms (SubprocVecEnv deadlocks on
    # Windows because closure-captured DataFrames can't be pickled
    # across spawn processes)
    n_envs = 4
    env = DummyVecEnv([lambda: TradingEnv(train_dict)] * n_envs)
    print(f"  Vectorized environment ready ({n_envs} envs, DummyVecEnv)")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_LR,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
    )

    callbacks = build_callbacks(lambda: TradingEnv(val_dict))

    print(f"\n  Starting PPO training ({total_timesteps:,} timesteps)...")
    print("  Training stats will appear every ~2 048 steps\n")
    logger.info(
        f"PPO training: {total_timesteps:,} timesteps · {len(train_dict)}"
        " tickers"
    )
    model.learn(
        total_timesteps=total_timesteps, callback=callbacks, progress_bar=True
    )

    # Save
    save_path = os.path.join(MODEL_DIR, "ppo_agent")
    model.save(save_path)
    logger.info(f"PPO model saved → {save_path}.zip")

    # Evaluate on val set
    print("\n  Evaluating on validation set...")
    metrics = evaluate_rl(model, val_dict, n_episodes=30)

    print("\n" + "=" * 60)
    print("  PPO Validation Results")
    print("=" * 60)
    print(
        f"  Mean Sharpe      : {metrics['mean_sharpe']:.3f}  (exit criterion:"
        " > 1.0)"
    )
    print(
        f"  Worst Drawdown   : {metrics['worst_drawdown']:.1%}  (exit"
        " criterion: < 20%)"
    )
    print(f"  Mean Ep. Return  : {metrics['mean_return']:+.2%}")
    print(f"  Episodes run     : {metrics['n_episodes']}")
    print()

    sharpe_ok = metrics["mean_sharpe"] > 1.0
    dd_ok = metrics["worst_drawdown"] < 0.20
    if sharpe_ok and dd_ok:
        print("  ✓ EXIT CRITERION MET — model ready for paper trading")
    else:
        reasons = []
        if not sharpe_ok:
            reasons.append(f"Sharpe {metrics['mean_sharpe']:.3f} < 1.0")
        if not dd_ok:
            reasons.append(f"Drawdown {metrics['worst_drawdown']:.1%} >= 20%")
        print(f"  ✗ Exit criterion NOT met ({', '.join(reasons)})")
        print("    Consider more training or adjusting reward shaping.")
    print("=" * 60 + "\n")

    return model


if __name__ == "__main__":
    from data.feature_engineering import build_features

    features = build_features()
    train_rl(features)
