"""
Prophet-based multi-step clinical score forecasting.

Provides:
  - ProphetScoreModel — per-patient Prophet model fitted in parallel via joblib.
                        Forecasts the next H score values from the last observed timestep.

Interface (raw DataFrame path — different from the (N,T,F) array models):
    fit(train_df, score_col, ...)   -> None
    predict(val_df, score_col, ...) -> np.ndarray shape (N_patients, H)

Evaluation unit: one (H,) forecast per patient (from their last observed timestep).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level helper (must be top-level for joblib pickling)
# ---------------------------------------------------------------------------

def _fit_one_patient(
    stay_id: int,
    group: pd.DataFrame,
    score_col: str,
    base_date: pd.Timestamp,
    prophet_kwargs: dict,
) -> tuple[int, object]:
    """Fit a single Prophet model for one patient.

    Parameters
    ----------
    stay_id : int
    group : pd.DataFrame
        Rows for this patient, sorted by timestep.
    score_col : str
    base_date : pd.Timestamp
        Reference date used to build synthetic datetime index from integer timesteps.
    prophet_kwargs : dict
        Extra keyword arguments forwarded to Prophet().

    Returns
    -------
    (stay_id, fitted_model_or_None)
    """
    try:
        from prophet import Prophet  # lazy import — not everyone has prophet installed
    except ImportError as exc:
        raise ImportError(
            "prophet is required for ProphetScoreModel. "
            "Install with: pip install prophet"
        ) from exc

    # Drop rows with NaN target
    group = group.dropna(subset=[score_col])
    if len(group) < 3:
        return stay_id, None  # not enough data for Prophet

    # Build ds/y DataFrame using synthetic timestamps
    ds = base_date + pd.to_timedelta(group['timestep'].values, unit='h')
    prophet_df = pd.DataFrame({'ds': ds, 'y': group[score_col].values.astype(float)})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        **prophet_kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)

    return stay_id, model


# ---------------------------------------------------------------------------
# ProphetScoreModel
# ---------------------------------------------------------------------------

class ProphetScoreModel:
    """Per-patient Prophet model for multi-step clinical score forecasting.

    One Prophet model is fitted per ICU stay using joblib parallelism.
    Prediction produces H forecasts starting one timestep after the last
    observed timestep for each patient.

    Parameters
    ----------
    prediction_horizon : int
        Number of future timesteps to forecast (H).
    n_jobs : int
        Number of parallel jobs for fitting. -1 = all CPUs.
    base_date : str
        ISO date string used as the anchor for synthetic Prophet timestamps.
        Must be consistent between fit and predict calls.
    prophet_kwargs : dict, optional
        Extra keyword arguments forwarded to Prophet() (e.g. changepoint_prior_scale).
    """

    def __init__(
        self,
        prediction_horizon: int = 6,
        n_jobs: int = -1,
        base_date: str = "2020-01-01",
        prophet_kwargs: Optional[dict] = None,
    ):
        self.prediction_horizon = prediction_horizon
        self.n_jobs = n_jobs
        self.base_date = pd.Timestamp(base_date)
        self.prophet_kwargs = prophet_kwargs or {}
        self._models: Dict[int, object] = {}  # stay_id -> fitted Prophet

        print(
            f"[ProphetScoreModel] H={prediction_horizon} | n_jobs={n_jobs}"
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        score_col: str,
        stay_id_col: str = 'stay_id',
        timestep_col: str = 'timestep',
    ) -> None:
        """Fit one Prophet model per patient in parallel.

        Parameters
        ----------
        train_df : pd.DataFrame
            Full training timeseries (one row per patient-timestep).
        score_col : str
            Target score column name.
        stay_id_col : str
        timestep_col : str
        """
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "joblib is required. Install with: pip install joblib"
            ) from exc

        grouped = [
            (sid, grp.sort_values(timestep_col))
            for sid, grp in train_df.groupby(stay_id_col)
        ]

        print(f"[ProphetScoreModel] Fitting {len(grouped)} patient models "
              f"(n_jobs={self.n_jobs})...")

        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=0)(
            delayed(_fit_one_patient)(
                sid, grp, score_col, self.base_date, self.prophet_kwargs
            )
            for sid, grp in grouped
        )

        self._models = {}
        skipped = 0
        for sid, model in results:
            if model is not None:
                self._models[sid] = model
            else:
                skipped += 1

        print(f"[ProphetScoreModel] Fitted {len(self._models)} models "
              f"({skipped} skipped — too few observations).")

    # ------------------------------------------------------------------
    def predict(
        self,
        val_df: pd.DataFrame,
        score_col: str,
        stay_id_col: str = 'stay_id',
        timestep_col: str = 'timestep',
    ) -> tuple[np.ndarray, List[int]]:
        """Forecast H steps ahead for each validation patient.

        Parameters
        ----------
        val_df : pd.DataFrame
        score_col : str
        stay_id_col : str
        timestep_col : str

        Returns
        -------
        preds : np.ndarray, shape (N_patients, H)
            Predicted score values. Rows for patients without a fitted model
            are filled with NaN.
        stay_ids : list of int
            Stay IDs corresponding to each row of preds.
        """
        stay_ids = sorted(val_df[stay_id_col].unique())
        preds = np.full((len(stay_ids), self.prediction_horizon), np.nan, dtype=np.float32)

        missing = 0
        for row_idx, sid in enumerate(stay_ids):
            if sid not in self._models:
                missing += 1
                continue

            patient_rows = val_df[val_df[stay_id_col] == sid].sort_values(timestep_col)
            last_timestep = int(patient_rows[timestep_col].iloc[-1])

            # Build future dataframe: H periods of 1-hour frequency
            last_ds = self.base_date + pd.to_timedelta(last_timestep, unit='h')
            future_ds = [
                last_ds + pd.to_timedelta(h, unit='h')
                for h in range(1, self.prediction_horizon + 1)
            ]
            future_df = pd.DataFrame({'ds': future_ds})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self._models[sid].predict(future_df)

            preds[row_idx] = forecast['yhat'].values[:self.prediction_horizon].astype(np.float32)

        if missing:
            warnings.warn(
                f"[ProphetScoreModel] {missing}/{len(stay_ids)} val patients had no fitted model "
                "(they were not in the training set). Their predictions are NaN."
            )

        return preds, stay_ids

    # ------------------------------------------------------------------
    def get_actual_multistep(
        self,
        df: pd.DataFrame,
        stay_ids: List[int],
        score_col: str,
        stay_id_col: str = 'stay_id',
        timestep_col: str = 'timestep',
    ) -> np.ndarray:
        """Extract the actual H future score values for each patient.

        For evaluation: returns the H steps starting one timestep after
        the last observed timestep per patient.

        Returns
        -------
        actuals : np.ndarray, shape (N_patients, H), dtype float32
        """
        actuals = np.full((len(stay_ids), self.prediction_horizon), np.nan, dtype=np.float32)
        for row_idx, sid in enumerate(stay_ids):
            patient_rows = df[df[stay_id_col] == sid].sort_values(timestep_col)
            last_timestep = int(patient_rows[timestep_col].iloc[-1])
            future_rows = patient_rows[patient_rows[timestep_col] > last_timestep]
            vals = future_rows[score_col].values[:self.prediction_horizon].astype(np.float32)
            actuals[row_idx, :len(vals)] = vals
        return actuals
