import sys
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import warnings


# ---------------------------------------------------------------------------
# Lightweight progress bar — no external dependency, works in any environment
# ---------------------------------------------------------------------------

class _ProgressBar:
    """Minimal tqdm-free progress bar that works in terminals and Jupyter."""

    def __init__(self, total: int, width: int = 40):
        self._total   = max(total, 1)
        self._current = 0
        self._width   = width
        self._start   = time.time()
        self._print()

    def _print(self):
        pct   = self._current / self._total
        filled = int(self._width * pct)
        bar   = '█' * filled + '░' * (self._width - filled)
        elapsed = time.time() - self._start
        print(f'\r  [{bar}] {self._current}/{self._total}  {elapsed:.1f}s',
              end='', flush=True)

    def update(self):
        self._current = min(self._current + 1, self._total)
        self._print()
        if self._current >= self._total:
            print()   # newline when done


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Clinical thresholds that define "at-risk" for each score.
# A patient is considered a positive event if ANY timestep in the future
# window has a score >= the corresponding threshold.
SCORE_THRESHOLDS: Dict[str, int] = {
    'sofa_score':  2,   # SOFA ≥ 2 → organ dysfunction (Sepsis-3)
    'sirs_score':  2,   # SIRS ≥ 2 → systemic inflammatory response
    'news2_score': 5,   # NEWS2 ≥ 5 → medium clinical risk (escalation trigger)
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def balance_dataframe(
    df: pd.DataFrame,
    score_cols: List[str],
    strategy: str = 'undersample',
    random_state: int = 42,
) -> pd.DataFrame:
    """Balance a patient-level timeseries DataFrame by score strata.

    For each score column present, bins patients into low / high risk groups
    and under-samples the majority group so that the final dataset has a more
    even representation across risk levels.

    Parameters
    ----------
    df : pd.DataFrame
        Full timeseries DataFrame (one row per timestep).
    score_cols : list of str
        Score columns to use for stratification (e.g. 'sofa_score').
        Uses the first column that is present in df.
    strategy : str
        'undersample'  — randomly drop majority-class patients (default, safest).
        'oversample'   — duplicate minority-class patients.
        'combined'     — oversample minority to 50 %, then undersample majority.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame (still one row per timestep).
    """
    rng = np.random.default_rng(random_state)

    # Pick first available score column
    target_col = next((c for c in score_cols if c in df.columns), None)
    if target_col is None:
        warnings.warn(
            f"None of the score columns {score_cols} found in DataFrame. "
            "Returning original DataFrame unchanged."
        )
        return df

    # Aggregate to patient level: use max score as strata key
    patient_scores = df.groupby('stay_id')[target_col].max()

    # Bin into low / high risk
    median_score = patient_scores.median()
    low_ids  = patient_scores[patient_scores <= median_score].index.values
    high_ids = patient_scores[patient_scores >  median_score].index.values

    print(f"\n[balance_dataframe] Stratifying on '{target_col}' (median={median_score:.1f})")
    print(f"  Low-risk patients  (score <= {median_score:.1f}): {len(low_ids)}")
    print(f"  High-risk patients (score >  {median_score:.1f}): {len(high_ids)}")

    minority_ids = low_ids  if len(low_ids)  < len(high_ids) else high_ids
    majority_ids = high_ids if len(low_ids)  < len(high_ids) else low_ids

    n_min = len(minority_ids)
    n_maj = len(majority_ids)

    if strategy == 'undersample':
        sampled_maj = rng.choice(majority_ids, size=n_min, replace=False)
        keep_ids = np.concatenate([minority_ids, sampled_maj])

    elif strategy == 'oversample':
        extra = rng.choice(minority_ids, size=n_maj - n_min, replace=True)
        keep_ids = np.concatenate([majority_ids, minority_ids, extra])

    elif strategy == 'combined':
        target_n = int((n_min + n_maj) / 2)
        if n_min < target_n:
            extra = rng.choice(minority_ids, size=target_n - n_min, replace=True)
            minority_ids = np.concatenate([minority_ids, extra])
        sampled_maj = rng.choice(majority_ids, size=target_n, replace=False)
        keep_ids = np.concatenate([minority_ids, sampled_maj])

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'undersample', 'oversample', or 'combined'.")

    balanced = df[df['stay_id'].isin(keep_ids)].copy()
    print(f"  Final balanced patients: {balanced['stay_id'].nunique()} "
          f"(was {df['stay_id'].nunique()})")
    return balanced


# ---------------------------------------------------------------------------
# TimeSeriesDataProcessor
# ---------------------------------------------------------------------------

class TimeSeriesDataProcessor:
    def __init__(self,
                 features: List[str],
                 task: str,
                 window_size: int = None,
                 prediction_horizon: int = None,
                 stride: int = 2,
                 balance: bool = False,
                 balance_strategy: str = 'undersample',
                 random_state: int = 42,
                 score_thresholds: Optional[Dict[str, int]] = None):
        """
        Parameters
        ----------
        features : list of str
        task : str
            One of: 'morta_hosp', 'los', 'mechvent', 'septic_shock', 'sepsis',
            'vasopressor', 'sofa_score', 'sirs_score', 'news2_score'
        window_size : int, optional
        prediction_horizon : int, optional
        stride : int
            Step size between consecutive sliding windows (default 2).
            stride=1 gives maximum overlap (original behaviour).
            stride=2 halves the number of windows and reduces memorization.
        balance : bool
            If True, apply class / strata balancing after window extraction.
        balance_strategy : str
            'undersample' | 'oversample' | 'combined'
        random_state : int
        score_thresholds : dict, optional
            Per-score clinical exceedance thresholds used by score tasks.
            Keys: 'sofa_score', 'sirs_score', 'news2_score'.
            Defaults to module-level SCORE_THRESHOLDS if None.
            Example: {'sofa_score': 3, 'sirs_score': 2, 'news2_score': 7}
        """
        self.features = features
        self.task = task
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.balance = balance
        self.balance_strategy = balance_strategy
        self.random_state = random_state
        self.score_thresholds = {**SCORE_THRESHOLDS, **(score_thresholds or {})}
        self.scalers = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets based on the task type.
        Returns:
            features: Array of feature values
            targets: Array of target values
        """
        if self.task == 'morta_hosp':
            X, y = self._prepare_mortality_data(df)
        elif self.task == 'los':
            X, y = self._prepare_los_data(df)
        elif self.task == 'mechvent':
            X, y = self._prepare_mechvent_data(df)
        elif self.task == 'septic_shock':
            X, y = self._prepare_septic_shock_data(df)
        elif self.task == 'sepsis':
            X, y = self._prepare_sepsis_data(df)
        elif self.task == 'vasopressor':
            X, y = self._prepare_vasopressor_data(df)
        elif self.task in ('sofa_score', 'sirs_score', 'news2_score'):
            threshold = self.score_thresholds[self.task]
            X, y = self._prepare_score_threshold_data(df, self.task, threshold)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        if self.balance and len(X) > 0:
            X, y = self.balance_windows(X, y)

        return X, y
        
    def _prepare_sepsis_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For sepsis prediction:
        - Use sliding windows of fixed size
        - Predict sepsis onset within prediction horizon (in timesteps)
        - Handles irregular timesteps
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for sepsis prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing sepsis data...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)

                    # Check if sepsis occurs within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    sepsis_occurs = future_window[self.task].max() > 0
                    targets.append(1 if sepsis_occurs else 0)
            bar.update()
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int8)


    def _prepare_mortality_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For mortality prediction:
        - Use fixed time window from admission
        - Single binary target per stay
        """
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing mortality data...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            window_data = group.head(self.window_size)[self.features].values
            if len(window_data) == self.window_size:  # Only use complete windows
                features.append(window_data)
                targets.append(group[self.task].iloc[-1])
            bar.update()
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int8)

    def _prepare_los_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For length of stay prediction:
        - Use fixed observation window
        - Predict total LOS using data from observation window
        - Exclude cases where LOS < observation window
        """
        # Temporary fix: Remove stays with NaN 'los' values
        df = df.dropna(subset=[self.task])
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing length of stay data...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Only include if we have enough data for the observation window
            if len(group) >= self.window_size:
                window_data = group.head(self.window_size)[self.features].values
                if len(window_data) == self.window_size:
                    features.append(window_data)
                    # Use the LOS value from the data (assuming it's in the self.task column)
                    total_los = group[self.task].iloc[-1]  # Get LOS from the last row
                    targets.append(total_los)
            bar.update()
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _prepare_mechvent_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For mechanical ventilation prediction:
        - Use sliding windows of fixed size
        - Predict mechanical ventilation onset within prediction horizon
        - Handles irregular timesteps
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for mechanical ventilation prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing mechanical ventilation data...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)

                    # Check if mechanical ventilation occurs within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    vent_occurs = future_window[self.task].max() > 0
                    targets.append(1 if vent_occurs else 0)
            bar.update()
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int8)

    def _prepare_septic_shock_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For septic shock prediction:
        - Use sliding windows of fixed size
        - Predict septic shock onset within prediction horizon
        - Handles irregular timesteps
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for septic shock prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing septic shock data...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)

                    # Check if septic shock occurs within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]
                    shock_occurs = future_window[self.task].max() > 0
                    targets.append(1 if shock_occurs else 0)
            bar.update()
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int8)

    def _prepare_vasopressor_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For vasopressor requirement prediction:
        - Use sliding windows of fixed size
        - Predict vasopressor requirement within prediction horizon
        - Handles irregular timesteps
        - Creates binary target based on vaso_median or vaso_max > 0
        """
        if self.prediction_horizon is None:
            raise ValueError("prediction_horizon must be set for vasopressor prediction")
        
        grouped = df.groupby('stay_id')
        features, targets = [], []
        
        print("Processing vasopressor requirement data...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            # Sort by timestep to ensure temporal order
            group = group.sort_values('timestep')
            
            # Create windows
            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)

                    # Check if vasopressor is required within prediction horizon
                    future_window = group.iloc[i + self.window_size:
                                             i + self.window_size + self.prediction_horizon]

                    # Check if either vaso_median or vaso_max is > 0
                    vaso_required = (future_window['vaso_median'].max() > 0) or (future_window['vaso_max'].max() > 0)
                    targets.append(1 if vaso_required else 0)
            bar.update()
        
        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int8)

    def _prepare_score_threshold_data(
        self,
        df: pd.DataFrame,
        score_col: str,
        threshold: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding-window binary classification: predict threshold exceedance.

        For each sliding window of ``window_size`` timesteps the target is:
            y = 1  if ANY timestep in the next ``prediction_horizon`` steps
                   has ``score_col`` >= ``threshold``
            y = 0  otherwise

        This produces a continuous probability P(threshold exceeded) when any
        probabilistic classifier is applied, and a hard binary prediction when
        the probability is thresholded at 0.5 (or a custom decision_threshold).

        Parameters
        ----------
        df : pd.DataFrame
        score_col : str
            One of 'sofa_score', 'sirs_score', 'news2_score'.
        threshold : int
            Clinical exceedance threshold (e.g. 2 for SOFA, 5 for NEWS2).
        """
        if self.prediction_horizon is None:
            raise ValueError(
                f"prediction_horizon must be set for score threshold task '{score_col}'"
            )
        if score_col not in df.columns:
            raise ValueError(f"Score column '{score_col}' not found in DataFrame.")

        grouped = df.groupby('stay_id')
        features, targets = [], []

        print(f"Processing {score_col} threshold-exceedance data "
              f"(threshold={threshold}, H={self.prediction_horizon})...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            group = group.sort_values('timestep')

            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)

                    future_window = group.iloc[
                        i + self.window_size:
                        i + self.window_size + self.prediction_horizon
                    ]
                    # Binary label: 1 if score exceeds threshold in any future step
                    exceeds = int(future_window[score_col].max() >= threshold)
                    targets.append(exceeds)
            bar.update()

        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int8)

    def _prepare_score_multistep_data(
        self,
        df: pd.DataFrame,
        score_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding-window regression returning the full future sequence as target.

        Identical window extraction to ``_prepare_score_regression_data`` except
        the target is the complete future sequence of length ``prediction_horizon``
        rather than its mean — yielding ``y`` shape ``(N, H)``.

        Parameters
        ----------
        df : pd.DataFrame
        score_col : str
            One of 'sofa_score', 'sirs_score', 'news2_score'.

        Returns
        -------
        X : np.ndarray, shape (N, window_size, n_features)
        y : np.ndarray, shape (N, prediction_horizon), dtype float32
        """
        if self.prediction_horizon is None:
            raise ValueError(
                f"prediction_horizon must be set for multistep score task '{score_col}'"
            )
        if score_col not in df.columns:
            raise ValueError(f"Score column '{score_col}' not found in DataFrame.")

        grouped = df.groupby('stay_id')
        features, targets = [], []

        print(f"Processing {score_col} multistep data (H={self.prediction_horizon})...")
        bar = _ProgressBar(len(grouped))
        for _, group in grouped:
            group = group.sort_values('timestep')

            for i in range(0, len(group) - self.window_size - self.prediction_horizon + 1, self.stride):
                window = group.iloc[i:i + self.window_size]
                if len(window) == self.window_size:
                    features.append(window[self.features].values)

                    future_window = group.iloc[
                        i + self.window_size:
                        i + self.window_size + self.prediction_horizon
                    ]
                    # Keep full future sequence — shape (H,)
                    future_seq = future_window[score_col].values.astype(np.float32)
                    if len(future_seq) == self.prediction_horizon:
                        targets.append(future_seq)
                    else:
                        # Pad with NaN if sequence is shorter than horizon (edge case)
                        padded = np.full(self.prediction_horizon, np.nan, dtype=np.float32)
                        padded[:len(future_seq)] = future_seq
                        targets.append(padded)
            bar.update()

        return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)

    def prepare_multistep_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Public entry point for multi-step score prediction.

        Only score regression tasks are supported (sofa_score, sirs_score, news2_score).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        X : np.ndarray, shape (N, window_size, n_features)
        y : np.ndarray, shape (N, prediction_horizon), dtype float32
        """
        SCORE_TASKS = ('sofa_score', 'sirs_score', 'news2_score')
        if self.task not in SCORE_TASKS:
            raise ValueError(
                f"prepare_multistep_data() only supports score tasks {SCORE_TASKS}, "
                f"got '{self.task}'"
            )
        X, y = self._prepare_score_multistep_data(df, self.task)
        if self.balance and len(X) > 0:
            rng = np.random.default_rng(self.random_state)
            y_mean = y.mean(axis=1)
            n_bins = 10
            bin_edges = np.nanpercentile(y_mean, np.linspace(0, 100, n_bins + 1))
            bin_edges[-1] += 1e-6
            bin_ids = np.digitize(y_mean, bin_edges[1:])
            unique_bins, counts = np.unique(bin_ids, return_counts=True)
            min_count = counts.min()
            keep_idx = []
            for b in unique_bins:
                bin_mask = np.where(bin_ids == b)[0]
                keep_idx.extend(rng.choice(bin_mask, size=min_count, replace=False).tolist())
            keep_idx = np.array(keep_idx)
            rng.shuffle(keep_idx)
            X, y = X[keep_idx], y[keep_idx]
        return X, y

    def balance_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance extracted windows by class or score stratum.

        For binary classification tasks: under-samples the majority class
        (or uses SMOTE oversample if strategy='oversample').

        For regression score tasks: bins targets into deciles and
        under-samples the dominant decile bin until all bins are equal size.

        Parameters
        ----------
        features : np.ndarray, shape (n_windows, timesteps, n_features)
        targets  : np.ndarray, shape (n_windows,)

        Returns
        -------
        features_bal, targets_bal : np.ndarray
        """
        rng = np.random.default_rng(self.random_state)

        # Score tasks are now binary classification (threshold exceedance).
        # Only 'los' remains a true regression task in the single-step path.
        regression_tasks = {'los'}
        is_regression = self.task in regression_tasks

        n_windows = len(targets)
        orig_shape = features.shape  # (n, timesteps, feats)

        if is_regression:
            # Bin targets into deciles, then under-sample each bin to min-bin size
            n_bins = 10
            bin_edges = np.nanpercentile(targets, np.linspace(0, 100, n_bins + 1))
            bin_edges[-1] += 1e-6  # ensure last bin includes max value
            bin_ids = np.digitize(targets, bin_edges[1:])  # 0..n_bins-1

            unique_bins, counts = np.unique(bin_ids, return_counts=True)
            if len(unique_bins) < 2:
                warnings.warn("Not enough score variation to balance windows. Returning unchanged.")
                return features, targets

            min_count = counts.min()
            keep_idx = []
            for b in unique_bins:
                bin_mask = np.where(bin_ids == b)[0]
                sampled = rng.choice(bin_mask, size=min_count, replace=False)
                keep_idx.extend(sampled.tolist())

            keep_idx = np.array(keep_idx)
            rng.shuffle(keep_idx)
            print(f"[balance_windows] Regression strata balancing: "
                  f"{n_windows} → {len(keep_idx)} windows "
                  f"({n_bins} decile bins, min_count={min_count})")

        else:
            # Binary classification: under-sample majority class
            pos_idx = np.where(targets == 1)[0]
            neg_idx = np.where(targets == 0)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                warnings.warn("Only one class present in targets. Returning unchanged.")
                return features, targets

            minority_idx = pos_idx if len(pos_idx) < len(neg_idx) else neg_idx
            majority_idx = neg_idx if len(pos_idx) < len(neg_idx) else pos_idx
            n_min = len(minority_idx)

            if self.balance_strategy == 'undersample':
                sampled_maj = rng.choice(majority_idx, size=n_min, replace=False)
                keep_idx = np.concatenate([minority_idx, sampled_maj])

            elif self.balance_strategy == 'oversample':
                try:
                    from imblearn.over_sampling import SMOTE
                    flat = features.reshape(n_windows, -1)
                    sm = SMOTE(random_state=self.random_state)
                    flat_bal, targets_bal = sm.fit_resample(flat, targets)
                    features_bal = flat_bal.reshape(-1, orig_shape[1], orig_shape[2])
                    print(f"[balance_windows] SMOTE: {n_windows} → {len(targets_bal)} windows")
                    return features_bal, targets_bal
                except ImportError:
                    warnings.warn(
                        "imbalanced-learn not installed; falling back to random oversampling."
                    )
                    extra = rng.choice(minority_idx, size=len(majority_idx) - n_min, replace=True)
                    keep_idx = np.concatenate([majority_idx, minority_idx, extra])

            elif self.balance_strategy == 'combined':
                target_n = (n_min + len(majority_idx)) // 2
                extra = rng.choice(minority_idx, size=max(0, target_n - n_min), replace=True)
                minority_aug = np.concatenate([minority_idx, extra])
                sampled_maj = rng.choice(majority_idx, size=target_n, replace=False)
                keep_idx = np.concatenate([minority_aug, sampled_maj])

            else:
                raise ValueError(
                    f"Unknown balance_strategy '{self.balance_strategy}'. "
                    "Use 'undersample', 'oversample', or 'combined'."
                )

            rng.shuffle(keep_idx)
            n_pos_bal = (targets[keep_idx] == 1).sum()
            n_neg_bal = (targets[keep_idx] == 0).sum()
            print(f"[balance_windows] {self.balance_strategy}: "
                  f"{n_windows} → {len(keep_idx)} windows "
                  f"(pos={n_pos_bal}, neg={n_neg_bal})")

        return features[keep_idx], targets[keep_idx]

    def normalize_features(self, train_data: np.ndarray, val_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features using training data statistics.
        All tasks now use the same normalization approach.
        """
        scaler = StandardScaler()
        
        # Reshape to 2D for scaling
        train_shape = train_data.shape
        val_shape = val_data.shape
        
        # Reshape to (n_samples * n_timesteps, n_features) if 3D
        if len(train_shape) == 3:
            train_reshaped = train_data.reshape(-1, train_shape[-1])
            val_reshaped = val_data.reshape(-1, val_shape[-1])
        else:
            train_reshaped = train_data
            val_reshaped = val_data
            
        # Fit on training data and transform both
        train_normalized = scaler.fit_transform(train_reshaped)
        val_normalized = scaler.transform(val_reshaped)
        
        # Reshape back to original shape if necessary
        if len(train_shape) == 3:
            train_normalized = train_normalized.reshape(train_shape)
            val_normalized = val_normalized.reshape(val_shape)
        
        self.scalers['features'] = scaler
        return train_normalized, val_normalized
    

