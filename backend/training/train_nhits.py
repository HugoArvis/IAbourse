import os
import sys

import lightning as pl
import mlflow
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    MLFLOW_TRACKING_URI, MODEL_DIR, TFT_BATCH_SIZE,
    TFT_EARLY_STOPPING_PATIENCE, TFT_FEATURE_NOISE_STD, TFT_MAX_EPOCHS,
)
from models.nhits_model import NHiTSModel  # noqa: E402
from training.callbacks import FeatureNoiseCallback  # noqa: E402
from training.dataset import (  # noqa: E402
    build_timeseries_dataset, build_weighted_train_dataloader,
)


def _walk_forward_splits(df, n_folds: int = 5):
    all_dates = sorted(df.index.unique())
    fold_size = len(all_dates) // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = all_dates[fold_size * i - 1]
        val_end = all_dates[min(fold_size * (i + 1) - 1, len(all_dates) - 1)]
        yield str(train_end.date()), str(val_end.date())


def train_nhits(features_df, n_folds: int = 5):
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_loss = float("inf")
    best_model = None

    for fold, (train_cut, val_cut) in enumerate(_walk_forward_splits(features_df, n_folds)):
        logger.info(f"N-HiTS Fold {fold + 1}/{n_folds}: train→{train_cut} val→{val_cut}")

        train_dataset = build_timeseries_dataset(features_df, cutoff_date=train_cut)
        val_dataset = build_timeseries_dataset(features_df, cutoff_date=val_cut)

        train_dl = build_weighted_train_dataloader(train_dataset, batch_size=TFT_BATCH_SIZE)
        val_dl = val_dataset.to_dataloader(train=False, batch_size=TFT_BATCH_SIZE, num_workers=0)

        wrapper = NHiTSModel.from_dataset(train_dataset)
        model = wrapper._model

        early_stop = EarlyStopping(monitor="val_loss", patience=TFT_EARLY_STOPPING_PATIENCE, mode="min")
        checkpoint = ModelCheckpoint(
            dirpath=MODEL_DIR,
            filename=f"nhits_fold{fold + 1}",
            monitor="val_loss",
            save_top_k=1,
        )
        noise = FeatureNoiseCallback(noise_std=TFT_FEATURE_NOISE_STD)

        trainer = pl.Trainer(
            max_epochs=TFT_MAX_EPOCHS,
            callbacks=[early_stop, checkpoint, noise],
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            accelerator="gpu",
            devices=1,
        )

        with mlflow.start_run(run_name=f"nhits_fold{fold + 1}", nested=True):
            mlflow.log_params({"fold": fold + 1, "train_cut": train_cut, "val_cut": val_cut})
            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
            val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
            mlflow.log_metric("val_loss", float(val_loss))

        if float(val_loss) < best_loss:
            best_loss = float(val_loss)
            best_model = wrapper

    if best_model:
        best_model.save(os.path.join(MODEL_DIR, "nhits_latest.ckpt"))
        logger.info(f"Best N-HiTS saved — val_loss={best_loss:.4f}")
    return best_model


if __name__ == "__main__":
    from data.feature_engineering import build_features
    features = build_features()
    train_nhits(features)
