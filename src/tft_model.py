"""
Temporal Fusion Transformer (TFT) for multi-step clinical score forecasting.

Provides:
  - TFTScoreModel — wraps pytorch-forecasting TemporalFusionTransformer with
                    a lightning Trainer. Outputs H predicted score values per window.

Interface (raw DataFrame path):
    fit(train_df, val_df, score_col, feature_cols, ...)  -> None
    predict(val_df, score_col, ...)                      -> np.ndarray shape (N_windows, H)

Requirements:
    pip install pytorch-forecasting lightning
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# TFTScoreModel
# ---------------------------------------------------------------------------

class TFTScoreModel:
    """Temporal Fusion Transformer for multi-step score prediction.

    Uses pytorch-forecasting's ``TemporalFusionTransformer`` trained with
    ``lightning.Trainer``.  The model naturally handles multi-step outputs
    via ``max_prediction_length``.

    Parameters
    ----------
    prediction_horizon : int
        Number of future timesteps to predict (H = max_prediction_length).
    window_size : int
        Number of past timesteps used as encoder context (max_encoder_length).
    max_epochs : int
        Training epochs.
    hidden_size : int
        TFT hidden layer dimension.
    attention_head_size : int
        Number of multi-head attention heads.
    dropout : float
    learning_rate : float
    batch_size : int
    n_workers : int
        DataLoader worker processes.
    accelerator : str
        'auto' lets Lightning choose GPU/CPU.
    """

    def __init__(
        self,
        prediction_horizon: int = 6,
        window_size: int = 6,
        max_epochs: int = 30,
        hidden_size: int = 32,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        n_workers: int = 4,
        accelerator: str = 'auto',
    ):
        self.prediction_horizon = prediction_horizon
        self.window_size = window_size
        self.max_epochs = max_epochs
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.accelerator = accelerator

        self._model = None       # TemporalFusionTransformer instance
        self._trainer = None     # lightning.Trainer instance
        self._training_ds = None # TimeSeriesDataSet (needed to build val dataset)

        print(
            f"[TFTScoreModel] H={prediction_horizon} | window={window_size} | "
            f"epochs={max_epochs} | hidden={hidden_size} | heads={attention_head_size}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _check_imports():
        try:
            import pytorch_forecasting  # noqa: F401
            import lightning            # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pytorch-forecasting and lightning are required for TFTScoreModel.\n"
                "Install with: pip install pytorch-forecasting lightning"
            ) from exc

    # ------------------------------------------------------------------
    def _build_dataset(
        self,
        df: pd.DataFrame,
        score_col: str,
        feature_cols: List[str],
        training_dataset=None,
    ):
        """Build a pytorch-forecasting TimeSeriesDataSet.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'stay_id' (int), 'timestep' (int), score_col, feature_cols.
        score_col : str
        feature_cols : list of str
        training_dataset : TimeSeriesDataSet or None
            If provided, builds a validation/test dataset from this training reference
            (ensures identical normalizers are reused).

        Returns
        -------
        TimeSeriesDataSet
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        # Ensure timestep is integer and stay_id is string (TFT group id)
        df = df.copy()
        df['stay_id'] = df['stay_id'].astype(str)
        df['timestep'] = df['timestep'].astype(int)

        # Drop NaN in target
        df = df.dropna(subset=[score_col])

        # Filter to patients with enough timesteps
        min_len = self.window_size + self.prediction_horizon
        counts = df.groupby('stay_id')['timestep'].count()
        valid_ids = counts[counts >= min_len].index
        df = df[df['stay_id'].isin(valid_ids)]

        if training_dataset is not None:
            return TimeSeriesDataSet.from_dataset(
                training_dataset, df, predict=True, stop_randomization=True
            )

        return TimeSeriesDataSet(
            df,
            time_idx='timestep',
            target=score_col,
            group_ids=['stay_id'],
            max_encoder_length=self.window_size,
            max_prediction_length=self.prediction_horizon,
            time_varying_unknown_reals=feature_cols,
            time_varying_known_reals=[],
            static_categoricals=[],
            static_reals=[],
            target_normalizer=GroupNormalizer(groups=['stay_id'], transformation='softplus'),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        score_col: str,
        feature_cols: List[str],
        stay_id_col: str = 'stay_id',
        timestep_col: str = 'timestep',
    ) -> None:
        """Train the TFT model.

        Parameters
        ----------
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        score_col : str
        feature_cols : list of str
            Covariates passed to the TFT as time_varying_unknown_reals.
        stay_id_col : str
        timestep_col : str
        """
        self._check_imports()
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import MAE
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping

        print(f"[TFTScoreModel] Building TimeSeriesDataSet for '{score_col}'...")
        self._training_ds = self._build_dataset(train_df, score_col, feature_cols)
        val_ds = self._build_dataset(
            val_df, score_col, feature_cols,
            training_dataset=self._training_ds,
        )

        train_loader = self._training_ds.to_dataloader(
            train=True, batch_size=self.batch_size,
            num_workers=self.n_workers, pin_memory=_CUDA_AVAILABLE,
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=self.batch_size * 2,
            num_workers=self.n_workers,
        )

        print(f"[TFTScoreModel] Instantiating TemporalFusionTransformer...")
        self._model = TemporalFusionTransformer.from_dataset(
            self._training_ds,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size // 2,
            output_size=7,          # 7 quantiles (default TFT output)
            loss=MAE(),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, mode='min', verbose=False
        )

        self._trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            enable_model_summary=False,
            gradient_clip_val=0.1,
            callbacks=[early_stop],
            logger=False,
            enable_checkpointing=False,
        )

        print(f"[TFTScoreModel] Training for up to {self.max_epochs} epochs...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._trainer.fit(
                self._model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

        print(f"[TFTScoreModel] Training complete.")

    # ------------------------------------------------------------------
    def predict(
        self,
        val_df: pd.DataFrame,
        score_col: str,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Generate multi-step predictions.

        Parameters
        ----------
        val_df : pd.DataFrame
        score_col : str
        feature_cols : list of str

        Returns
        -------
        np.ndarray, shape (N_windows, H)
            Median (quantile index 3) predictions from the TFT quantile output.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        val_ds = self._build_dataset(
            val_df, score_col, feature_cols,
            training_dataset=self._training_ds,
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=self.batch_size * 2,
            num_workers=self.n_workers,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_predictions = self._model.predict(
                val_loader, mode='prediction', return_x=False
            )

        # raw_predictions: Tensor shape (N, H) — already median by default in 'prediction' mode
        preds = raw_predictions.cpu().numpy().astype(np.float32)
        return preds  # shape (N_windows, H)

    # ------------------------------------------------------------------
    def attention_weights(self, val_df: pd.DataFrame, score_col: str, feature_cols: List[str]) -> dict:
        """Return TFT variable importance (attention-based).

        Returns
        -------
        dict with keys: 'encoder_variables', 'decoder_variables', 'static_variables'
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        val_ds = self._build_dataset(
            val_df, score_col, feature_cols,
            training_dataset=self._training_ds,
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=self.batch_size * 2,
            num_workers=self.n_workers,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpretation = self._model.interpret_output(
                self._model.predict(val_loader, mode='raw', return_x=True)[0],
                reduction='sum',
            )

        return {
            'encoder_variables': dict(zip(
                val_ds.encoder_reals,
                interpretation['encoder_variables'].cpu().numpy().tolist(),
            )),
        }
