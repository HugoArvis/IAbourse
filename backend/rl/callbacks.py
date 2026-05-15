from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv


def build_callbacks(make_env_fn, log_path: str = "models/eval_logs/"):
    eval_env = DummyVecEnv([make_env_fn])

    # Episode reward ~20 corresponds to Sharpe ≈ 1.0 (10 from incremental + 10 terminal bonus)
    stop = StopTrainingOnRewardThreshold(reward_threshold=20.0, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/checkpoints/",
        log_path=log_path,
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        callback_on_new_best=stop,
        verbose=1,
    )
    return [eval_cb]
