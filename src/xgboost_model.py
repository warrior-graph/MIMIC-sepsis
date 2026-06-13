"""
Gradient-boosting models for clinical score prediction.

Provides:
  - XGBoostModel  — wraps xgboost.XGBRegressor / XGBClassifier
  - LightGBMModel — wraps lightgbm.LGBMRegressor / LGBMClassifier
  - MultiScoreModel — runs one model per score (sofa / sirs / news2)
                      with a swappable model_factory for easy backend switching

Interface contract (same as LinearTimeSeriesModel and LSTMModel):
    fit(X: np.ndarray, y: np.ndarray) -> None
        X shape: (N, T, F)  — N windows, T timesteps, F features
    predict(X: np.ndarray) -> np.ndarray
        returns shape (N,) — predicted score or probability
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

_XGB_DEVICE  = "cuda" if _CUDA_AVAILABLE else "cpu"
_LGBM_DEVICE = "gpu"  if _CUDA_AVAILABLE else "cpu"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten(X: np.ndarray) -> np.ndarray:
    """Reshape (N, T, F) → (N, T*F) for tree-based models."""
    if X.ndim == 3:
        N, T, F = X.shape
        return X.reshape(N, T * F)
    return X  # already 2-D


def _to_dmatrix(X: np.ndarray, y: np.ndarray = None, device: str = "cpu"):
    """Wrap a numpy array in xgboost.DMatrix placed on *device*.

    Passing an explicit DMatrix avoids the ``inplace_predict`` device-mismatch
    warning that occurs when the booster runs on CUDA but receives a CPU array.
    """
    from xgboost import DMatrix
    X_flat = _flatten(X)
    if y is not None:
        dm = DMatrix(X_flat, label=y)
    else:
        dm = DMatrix(X_flat)
    dm.set_info(feature_names=[f"f{i}" for i in range(X_flat.shape[1])])
    # XGBoost ≥ 2.0 supports set_device; use setParam for older versions
    try:
        dm.set_info(device=device)
    except TypeError:
        pass  # older xgboost — device param not supported; data stays on CPU
    return dm


# ---------------------------------------------------------------------------
# XGBoostModel
# ---------------------------------------------------------------------------

class XGBoostModel:
    """XGBoost wrapper with the same fit/predict interface as LinearTimeSeriesModel.

    Parameters
    ----------
    task_type : {'regression', 'classification'}
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    colsample_bytree : float
    random_state : int
    n_jobs : int
        -1 uses all available CPU cores.
    early_stopping_rounds : int or None
        If set, pass an ``eval_set`` to ``fit()`` via the keyword argument.
    **kwargs
        Any additional keyword arguments forwarded to XGBRegressor/XGBClassifier.
    """

    def __init__(
        self,
        task_type: str = "regression",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ):
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostModel. Install with: pip install xgboost"
            ) from exc

        self.task_type = task_type
        self.early_stopping_rounds = early_stopping_rounds

        common = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            device=_XGB_DEVICE,
            **kwargs,
        )

        if task_type == "regression":
            self.model = XGBRegressor(**common)
        else:
            self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **common)

        print(
            f"[XGBoostModel] task={task_type} | n_est={n_estimators} | "
            f"depth={max_depth} | lr={learning_rate}"
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, F) or (N, T*F)
        y : np.ndarray, shape (N,)
        eval_set : list of (X_val, y_val) tuples, optional
            Only used when *early_stopping_rounds* is set.
        """
        fit_kwargs: dict = {}

        if self.early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            if eval_set is not None:
                # Flatten eval arrays; keep as numpy — sklearn wrapper handles device transfer
                fit_kwargs["eval_set"] = [
                    (_flatten(Xv), yv) for Xv, yv in eval_set
                ]
            else:
                warnings.warn(
                    "early_stopping_rounds set but no eval_set provided — "
                    "early stopping will not be applied.",
                    UserWarning,
                )

        X_flat = _flatten(X)
        # Pass numpy to the sklearn wrapper — it places data on the booster device internally
        self.model.fit(X_flat, y, **fit_kwargs)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Returns probabilities (positive class) for classification, or
        predicted values for regression.

        Uses ``xgboost.DMatrix(device=...)`` so data and booster sit on the
        same device, eliminating the "Falling back to prediction using DMatrix
        due to mismatched devices" warning.
        """
        from xgboost import DMatrix
        X_flat = _flatten(X)
        # Explicitly wrap in DMatrix and call booster.predict() directly.
        # This bypasses the inplace_predict code path that triggers the
        # "Falling back to prediction using DMatrix due to mismatched devices"
        # warning when the booster lives on CUDA but receives a CPU numpy array.
        dtest = DMatrix(X_flat)
        raw = self.model.get_booster().predict(dtest)
        # For binary classification the booster returns positive-class probability directly.
        # For multi-class it returns shape (N, n_classes) — caller handles as needed.
        return raw

    # ------------------------------------------------------------------
    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the trained XGBoost model (gain-based)."""
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# LightGBMModel
# ---------------------------------------------------------------------------

class LightGBMModel:
    """LightGBM wrapper with the same fit/predict interface as XGBoostModel.

    Parameters
    ----------
    task_type : {'regression', 'classification'}
    n_estimators : int
    max_depth : int
        -1 means no limit (LightGBM default).
    learning_rate : float
    subsample : float
        LightGBM calls this ``bagging_fraction``.
    colsample_bytree : float
        LightGBM calls this ``feature_fraction``.
    random_state : int
    n_jobs : int
    early_stopping_rounds : int or None
    **kwargs
        Forwarded to LGBMRegressor/LGBMClassifier.
    """

    def __init__(
        self,
        task_type: str = "regression",
        n_estimators: int = 300,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_samples: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = None,
        **kwargs,
    ):
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LightGBMModel. Install with: pip install lightgbm"
            ) from exc

        self.task_type = task_type
        self.early_stopping_rounds = early_stopping_rounds

        common = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,            # bagging_fraction alias
            colsample_bytree=colsample_bytree,  # feature_fraction alias
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            device=_LGBM_DEVICE,
            verbose=-1,
            **kwargs,
        )

        if task_type == "regression":
            self.model = LGBMRegressor(**common)
        else:
            self.model = LGBMClassifier(**common)

        print(
            f"[LightGBMModel] task={task_type} | n_est={n_estimators} | "
            f"depth={max_depth} | lr={learning_rate}"
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> None:
        X_flat = _flatten(X)
        fit_kwargs: dict = {}

        if self.early_stopping_rounds is not None:
            fit_kwargs["callbacks"] = []
            try:
                from lightgbm import early_stopping, log_evaluation
                fit_kwargs["callbacks"].append(early_stopping(self.early_stopping_rounds, verbose=False))
                fit_kwargs["callbacks"].append(log_evaluation(period=-1))
            except ImportError:
                pass  # older lightgbm — silently skip callbacks
            if eval_set is not None:
                flat_eval = [(_flatten(Xv), yv) for Xv, yv in eval_set]
                fit_kwargs["eval_set"] = flat_eval
            else:
                warnings.warn(
                    "early_stopping_rounds set but no eval_set provided.",
                    UserWarning,
                )

        self.model.fit(X_flat, y, **fit_kwargs)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = _flatten(X)
        if self.task_type == "classification":
            return self.model.predict_proba(X_flat)[:, 1]
        return self.model.predict(X_flat)

    # ------------------------------------------------------------------
    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# MultiScoreModel
# ---------------------------------------------------------------------------

SCORE_TARGETS: List[str] = ["sofa_score", "sirs_score", "news2_score"]


class MultiScoreModel:
    """Train and predict all three clinical scores in one call.

    Each score gets its own independent model instance, created by
    ``model_factory``.  Swap the factory to change the backend for all
    three scores at once.

    Parameters
    ----------
    model_factory : callable
        A zero-argument (or keyword-argument) callable that returns a
        fresh model implementing ``fit(X, y)`` / ``predict(X)``.
        Defaults to ``XGBoostModel`` with regression task type.
    scores : list of str
        Score column names to predict.  Defaults to
        ``['sofa_score', 'sirs_score', 'news2_score']``.

    Examples
    --------
    Using the default XGBoost backend::

        multi = MultiScoreModel()
        multi.fit(X_train, {"sofa_score": y_sofa, "sirs_score": y_sirs, "news2_score": y_news2})
        preds = multi.predict(X_val)
        # preds == {'sofa_score': arr, 'sirs_score': arr, 'news2_score': arr}

    Using LightGBM::

        from xgboost_model import LightGBMModel, MultiScoreModel
        multi = MultiScoreModel(
            model_factory=lambda: LightGBMModel(task_type="regression", n_estimators=200)
        )

    Using any other sklearn-compatible regressor::

        from sklearn.ensemble import RandomForestRegressor

        class _RFWrapper:
            def __init__(self): self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            def fit(self, X, y): self.model.fit(X.reshape(len(X), -1), y)
            def predict(self, X): return self.model.predict(X.reshape(len(X), -1))

        multi = MultiScoreModel(model_factory=_RFWrapper)
    """

    def __init__(
        self,
        model_factory: Optional[Callable[[], object]] = None,
        scores: Optional[List[str]] = None,
    ):
        if model_factory is None:
            model_factory = lambda: XGBoostModel(task_type="regression")  # noqa: E731

        self.model_factory = model_factory
        self.scores = scores if scores is not None else SCORE_TARGETS

        self.models: Dict[str, object] = {score: model_factory() for score in self.scores}

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> None:
        """Fit one model per score.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, F)
        y_dict : dict mapping score name → target array of shape (N,)
        """
        for score in self.scores:
            if score not in y_dict:
                warnings.warn(f"[MultiScoreModel] Target '{score}' missing from y_dict — skipping.")
                continue
            print(f"\n[MultiScoreModel] Fitting model for '{score}' ...")
            self.models[score].fit(X, y_dict[score])

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions for all scores.

        Returns
        -------
        dict
            Keys are score names; values are prediction arrays of shape (N,).
        """
        return {score: self.models[score].predict(X) for score in self.scores}

    # ------------------------------------------------------------------
    def feature_importances(self) -> Dict[str, np.ndarray]:
        """Return feature importances for each score model (if supported)."""
        out: Dict[str, np.ndarray] = {}
        for score, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                out[score] = model.feature_importances_
            else:
                warnings.warn(f"[MultiScoreModel] Model for '{score}' has no feature_importances_.")
        return out
