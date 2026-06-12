import argparse
import gc
import pandas as pd
import numpy as np
from datetime import datetime
from data_processor import TimeSeriesDataProcessor
from linear_model import LinearTimeSeriesModel
from lstm_model import LSTMModel
from transformer_model import TimeSeriesTransformer
from xgboost_model import XGBoostModel, LightGBMModel
from prophet_model import ProphetScoreModel
from tft_model import TFTScoreModel
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import torch

# ── GPU auto-detection ────────────────────────────────────────────────────────
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEFAULT_BATCH_SIZE = 128 if _CUDA_AVAILABLE else 32


def _log_gpu_info():
    """Print GPU device information at startup."""
    if _CUDA_AVAILABLE:
        dev_name = torch.cuda.get_device_name(0)
        dev_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[GPU] {dev_name} — {dev_mem:.1f} GB VRAM")
        print(f"[GPU] Default batch size auto-set to {_DEFAULT_BATCH_SIZE}")
    else:
        print("[GPU] No CUDA device detected — running on CPU")
        print(f"[GPU] Default batch size: {_DEFAULT_BATCH_SIZE}")


# Set random seeds at the top of your file
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries used"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make TensorFlow deterministic
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set NumPy print options for consistent output
    np.set_printoptions(precision=3, suppress=True)

    # cuDNN: allow auto-tuning for speed (deterministic=True ensures reproducibility)
    if _CUDA_AVAILABLE:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # auto-tune kernels for input size
    
    print(f"Random seeds set to {seed} for reproducibility")



def load_data(data_path: str) -> pd.DataFrame:
    """Load the patient timeseries data with memory-efficient dtypes.

    Reads the CSV and immediately downcasts float64→float32 and int64→int32
    to roughly halve memory consumption before any processing begins.
    """
    print("Loading patient timeseries data...")
    df = pd.read_csv(data_path)
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes('int64').columns:
        df[col] = df[col].astype('int32')
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  Loaded {len(df):,} rows, {df['stay_id'].nunique():,} patients "
          f"— {mem_mb:.0f} MB in RAM (float32/int32)")
    return df

def get_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """Get feature columns by excluding specific columns"""
    # Always-excluded columns (identifiers, future leakage)
    exclude_columns = [
        'morta_hosp',  # future information — exclude to avoid data leakage
        'morta_90',    # future information — exclude to avoid data leakage
        'timestep',    # temporal index
        'stay_id',     # identifier
        target_col,    # target variable itself
        'los',         # future information — exclude to avoid data leakage
        # Exclude other target-like columns unless we are predicting them
        'mechvent'     if target_col != 'mechvent'     else None,
        'septic_shock' if target_col != 'septic_shock' else None,
        'vasopressor'  if target_col != 'vasopressor'  else None,
        'vaso_median'  if target_col != 'vasopressor'  else None,
        'vaso_max'     if target_col != 'vasopressor'  else None,
        # Exclude score columns from features when predicting a different score
        # (avoid trivial cross-prediction leakage)
        'sofa_score'   if target_col != 'sofa_score'   else None,
        'sirs_score'   if target_col != 'sirs_score'   else None,
        'news2_score'  if target_col != 'news2_score'  else None,
        # Also exclude SOFA sub-scores — they are derived from features already present
        'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_cns', 'sofa_renal',
        'sepsis',  # flag computed post-hoc
    ]
    return [col for col in df.columns if col not in exclude_columns]

def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets"""
    patient_ids = df['stay_id'].unique()
    train_size = int(len(patient_ids) * train_ratio)
    
    # Use np.random.RandomState with fixed seed for shuffling
    rs = np.random.RandomState(42)
    shuffled_ids = patient_ids.copy()
    rs.shuffle(shuffled_ids)
    
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:]
    
    train_df = df[df['stay_id'].isin(train_ids)]
    val_df = df[df['stay_id'].isin(val_ids)]
    
    print(f"\nTrain set: {len(train_ids)} patients")
    print(f"Val set: {len(val_ids)} patients")
    
    return train_df, val_df

def evaluate_model_multistep(
    targets: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, float]:
    """Compute per-step and aggregate metrics for multi-step regression.

    Parameters
    ----------
    targets : np.ndarray, shape (N, H)
    preds   : np.ndarray, shape (N, H)

    Returns
    -------
    dict with keys: mean_rmse, mean_mae, rmse_step_1..H, mae_step_1..H
    """
    # Remove rows where either target or pred contains NaN
    valid_mask = ~(np.isnan(targets).any(axis=1) | np.isnan(preds).any(axis=1))
    targets = targets[valid_mask]
    preds = preds[valid_mask]

    H = targets.shape[1]
    metrics: Dict[str, float] = {}
    step_rmses, step_maes = [], []

    for h in range(H):
        rmse_h = float(np.sqrt(np.mean((targets[:, h] - preds[:, h]) ** 2)))
        mae_h  = float(np.mean(np.abs(targets[:, h] - preds[:, h])))
        metrics[f'rmse_step_{h + 1}'] = rmse_h
        metrics[f'mae_step_{h + 1}']  = mae_h
        step_rmses.append(rmse_h)
        step_maes.append(mae_h)

    metrics['mean_rmse'] = float(np.mean(step_rmses))
    metrics['mean_mae']  = float(np.mean(step_maes))
    return metrics


def print_results_multistep(
    model_metrics: Dict[str, Dict[str, float]],
    task: str,
    model_type: str,
) -> None:
    """Print multi-step regression metrics."""
    print("\nResults (multi-step):")
    for split in ('train', 'val'):
        m = model_metrics[split]
        print(f"  {split.capitalize()} Mean RMSE: {m['mean_rmse']:.3f}  |  Mean MAE: {m['mean_mae']:.3f}")
        H = sum(1 for k in m if k.startswith('rmse_step_'))
        step_str = '  '.join(f"h{h+1}={m[f'rmse_step_{h+1}']:.3f}" for h in range(H))
        print(f"  {split.capitalize()} RMSE per step:  {step_str}")


def evaluate_model(targets: np.ndarray, predictions: np.ndarray, task_type: str,
                   decision_threshold: float = 0.5) -> Dict[str, float]:
    """Calculate performance metrics based on task type.

    Parameters
    ----------
    targets : np.ndarray
        Ground-truth labels (binary for classification, continuous for regression).
    predictions : np.ndarray
        Model outputs: probability in [0, 1] for classification, continuous for regression.
    task_type : str
        'classification' or 'regression'.
    decision_threshold : float
        Probability cut-off for hard binary prediction (default 0.5).
        Only used for classification tasks.
    """
    if task_type == 'classification':
        # Continuous probability → hard binary decision at decision_threshold
        binary_predictions = (predictions >= decision_threshold).astype(int)
        accuracy = np.mean(binary_predictions == targets)
        return {
            'auroc': roc_auc_score(targets, predictions),
            'auprc': average_precision_score(targets, predictions),
            'accuracy': accuracy,
            'decision_threshold': decision_threshold,
        }
    else:  # regression
        mse = np.mean((targets - predictions) ** 2)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': np.mean(np.abs(targets - predictions))
        }

def get_baseline_metrics(train_targets: np.ndarray, val_targets: np.ndarray,
                         task_type: str,
                         decision_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """Calculate baseline performance based on task type"""
    if task_type == 'classification':
        majority_pred = train_targets.mean() > 0.5
        train_baseline = np.ones_like(train_targets, dtype=float) * majority_pred
        val_baseline   = np.ones_like(val_targets,   dtype=float) * majority_pred
    else:  # regression
        mean_pred = np.mean(train_targets)
        train_baseline = np.ones_like(train_targets) * mean_pred
        val_baseline   = np.ones_like(val_targets)   * mean_pred

    return {
        'train': evaluate_model(train_targets, train_baseline, task_type, decision_threshold),
        'val':   evaluate_model(val_targets,   val_baseline,   task_type, decision_threshold),
    }

def print_results(model_metrics: Dict[str, Dict[str, float]], baseline_metrics: Dict[str, Dict[str, float]], task_type: str):
    """Print model and baseline performance metrics based on task type"""
    print("\nResults:")
    print("Model Performance:")
    if task_type == 'classification':
        print(f"Train Accuracy: {model_metrics['train']['accuracy']:.3f}")
        print(f"Train AUROC: {model_metrics['train']['auroc']:.3f}")
        print(f"Train AUPRC: {model_metrics['train']['auprc']:.3f}")
        print(f"Val Accuracy: {model_metrics['val']['accuracy']:.3f}")
        print(f"Val AUROC: {model_metrics['val']['auroc']:.3f}")
        print(f"Val AUPRC: {model_metrics['val']['auprc']:.3f}")
        
        print("\nBaseline (Majority Class) Performance:")
        print(f"Train Accuracy: {baseline_metrics['train']['accuracy']:.3f}")
        print(f"Train AUROC: {baseline_metrics['train']['auroc']:.3f}")
        print(f"Train AUPRC: {baseline_metrics['train']['auprc']:.3f}")
        print(f"Val Accuracy: {baseline_metrics['val']['accuracy']:.3f}")
        print(f"Val AUROC: {baseline_metrics['val']['auroc']:.3f}")
        print(f"Val AUPRC: {baseline_metrics['val']['auprc']:.3f}")
    else:  # regression
        print(f"Train RMSE: {model_metrics['train']['rmse']:.3f}")
        print(f"Train MAE: {model_metrics['train']['mae']:.3f}")
        print(f"Val RMSE: {model_metrics['val']['rmse']:.3f}")
        print(f"Val MAE: {model_metrics['val']['mae']:.3f}")
        
        print("\nBaseline (Mean Prediction) Performance:")
        print(f"Train RMSE: {baseline_metrics['train']['rmse']:.3f}")
        print(f"Train MAE: {baseline_metrics['train']['mae']:.3f}")
        print(f"Val RMSE: {baseline_metrics['val']['rmse']:.3f}")
        print(f"Val MAE: {baseline_metrics['val']['mae']:.3f}")

def run_benchmark(task: str, model_type: str, include_treatments: bool = True,
                 prediction_horizon: int = None, random_state: int = 42,
                 regularization: str = 'ridge', alpha: float = 1.0,
                 balance: bool = False, balance_strategy: str = 'undersample',
                 data_path: str = "processed_files/patient_timeseries_v4.csv",
                 # Score threshold-exceedance parameters
                 score_thresholds: dict = None,
                 decision_threshold: float = 0.5,
                 # XGBoost / LightGBM hyperparameters
                 gbm_n_estimators: int = 300,
                 gbm_max_depth: int = 6,
                 gbm_learning_rate: float = 0.05,
                 gbm_subsample: float = 0.8,
                 gbm_colsample: float = 0.8,
                 gbm_early_stopping: int = None,
                 # LSTM multi-step
                 lstm_output_dim: int = 1,
                 # TFT hyperparameters
                 tft_max_epochs: int = 30,
                 tft_hidden_size: int = 32,
                 tft_attention_heads: int = 4,
                 tft_dropout: float = 0.1,
                 tft_batch_size: int = 64):
    # Determine task type based on target column.
    # Score tasks are binary threshold-exceedance classification, not regression.
    SCORE_TASKS = ['sofa_score', 'sirs_score', 'news2_score']
    TEMPORAL_TASKS = ['mechvent', 'septic_shock', 'sepsis', 'vasopressor'] + SCORE_TASKS
    task_type = 'regression' if task in ['los'] else 'classification'

    # Load and prepare data
    df = load_data(data_path)
    features = get_feature_columns(df, task)

    # Filter out treatment variables if specified
    if not include_treatments:
        treatment_vars = ['mechvent', 'vaso_median', 'vaso_max', 'abx_given',
                          'hours_since_first_abx', 'num_abx', 'fluid_total', 'fluid_step',
                          'peep', 'tidal_volume', 'minute_volume',
                          'peak_inspiratory_pressure', 'mean_airway_pressure']
        features = [f for f in features if f not in treatment_vars]

    # Initialize processor
    print("\nInitializing data processor...")
    processor = TimeSeriesDataProcessor(
        features=features,
        task=task,
        window_size=6,
        prediction_horizon=prediction_horizon if task in TEMPORAL_TASKS else None,
        balance=balance,
        balance_strategy=balance_strategy,
        random_state=random_state,
        score_thresholds=score_thresholds,  # None → uses module-level SCORE_THRESHOLDS
    )

    
    # Split data — then immediately free the full DataFrame
    train_df, val_df = split_data(df)
    del df
    gc.collect()

    # Process and normalize data
    print("\nProcessing data...")
    train_features, train_targets = processor.prepare_data(train_df)
    val_features, val_targets = processor.prepare_data(val_df)
    train_features_norm, val_features_norm = processor.normalize_features(train_features, val_features)
    
    # ── Multi-step path: Prophet and TFT operate on raw DataFrames ───────────
    MULTISTEP_MODELS = ('prophet', 'tft', 'lstm_multistep')
    if model_type in MULTISTEP_MODELS:
        if task not in SCORE_TASKS:
            raise ValueError(
                f"model_type='{model_type}' only supports score tasks {SCORE_TASKS}, "
                f"got '{task}'"
            )
        print(f"\nTraining {model_type} model (multi-step, H={prediction_horizon})...")

        if model_type == 'prophet':
            model = ProphetScoreModel(prediction_horizon=prediction_horizon)
            model.fit(train_df, score_col=task)
            train_preds, train_ids = model.predict(train_df, score_col=task)
            val_preds,   val_ids   = model.predict(val_df,   score_col=task)
            # Build matching actual arrays from the raw DataFrames
            train_targets_ms = model.get_actual_multistep(train_df, train_ids, score_col=task)
            val_targets_ms   = model.get_actual_multistep(val_df,   val_ids,   score_col=task)

        elif model_type == 'tft':
            model = TFTScoreModel(
                prediction_horizon=prediction_horizon,
                window_size=processor.window_size,
                max_epochs=tft_max_epochs,
                hidden_size=tft_hidden_size,
                attention_head_size=tft_attention_heads,
                dropout=tft_dropout,
                batch_size=tft_batch_size,
            )
            model.fit(train_df, val_df, score_col=task, feature_cols=features)
            val_preds   = model.predict(val_df,   score_col=task, feature_cols=features)
            train_preds = model.predict(train_df, score_col=task, feature_cols=features)
            # Actuals via processor multistep method
            train_targets_ms, _ = processor.prepare_multistep_data(train_df)
            # targets shape (N_windows, H) — align with preds
            val_targets_ms, _   = processor.prepare_multistep_data(val_df)
            # TFT may return fewer rows than windows if some patients are filtered
            min_train = min(len(train_preds), len(train_targets_ms))
            min_val   = min(len(val_preds),   len(val_targets_ms))
            train_preds      = train_preds[:min_train]
            train_targets_ms = train_targets_ms[:min_train]
            val_preds        = val_preds[:min_val]
            val_targets_ms   = val_targets_ms[:min_val]

        elif model_type == 'lstm_multistep':
            H = prediction_horizon
            train_targets_ms, val_targets_ms_raw = None, None
            # Use processor to get (N, H) targets
            train_features_ms, train_targets_ms = processor.prepare_multistep_data(train_df)
            val_features_ms,   val_targets_ms   = processor.prepare_multistep_data(val_df)
            train_features_ms_norm, val_features_ms_norm = processor.normalize_features(
                train_features_ms, val_features_ms
            )
            input_dim = train_features_ms_norm.shape[2]
            model = LSTMModel(
                task_type='regression',
                input_dim=input_dim,
                output_dim=H,
            )
            model.fit(train_features_ms_norm, train_targets_ms, batch_size=32)
            train_preds = model.predict(train_features_ms_norm, batch_size=32)
            val_preds   = model.predict(val_features_ms_norm,   batch_size=32)

        model_metrics_ms = {
            'train': evaluate_model_multistep(train_targets_ms, train_preds),
            'val':   evaluate_model_multistep(val_targets_ms,   val_preds),
        }
        print_results_multistep(model_metrics_ms, task, model_type)

        result = {
            'task': task,
            'model_type': model_type,
            'include_treatments': include_treatments,
            'prediction_horizon': prediction_horizon,
            'regularization': regularization,
            'alpha': alpha,
        }
        for split in ('train', 'val'):
            for metric, value in model_metrics_ms[split].items():
                result[f'{split}_{metric}'] = value
        return result

    # ── Single-step path (all existing models) ────────────────────────────────

    # Configure batch size based on model type — auto-scale for GPU
    batch_size = _DEFAULT_BATCH_SIZE if model_type in ['lstm', 'transformer'] else None
    
    # Train model
    print(f"\nTraining {model_type} model...")
    if model_type == 'linear':
        # Print regularization information for regression tasks
        if task_type == 'regression':
            if regularization == 'ridge':
                print(f"Using Ridge regression with alpha={alpha} (L2 regularization)")
            elif regularization == 'lasso':
                print(f"Using Lasso regression with alpha={alpha} (L1 regularization)")
            elif regularization == 'elasticnet':
                print(f"Using ElasticNet regression with alpha={alpha}, l1_ratio=0.5 (combined L1/L2 regularization)")
            else:
                print("Using standard Linear Regression (no regularization)")
        else:
            print("Using Logistic Regression for classification task")
            
        model = LinearTimeSeriesModel(
            task_type=task_type,
            random_state=random_state,
            regularization=regularization if task_type == 'regression' else None,
            alpha=alpha
        )
    elif model_type == 'lstm':
        input_dim = train_features_norm.shape[2]
        model = LSTMModel(task_type=task_type, input_dim=input_dim, output_dim=1)
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(task_type=task_type)
    elif model_type == 'xgboost':
        model = XGBoostModel(
            task_type=task_type,
            n_estimators=gbm_n_estimators,
            max_depth=gbm_max_depth,
            learning_rate=gbm_learning_rate,
            subsample=gbm_subsample,
            colsample_bytree=gbm_colsample,
            random_state=random_state,
            early_stopping_rounds=gbm_early_stopping,
        )
    elif model_type == 'lightgbm':
        model = LightGBMModel(
            task_type=task_type,
            n_estimators=gbm_n_estimators,
            max_depth=gbm_max_depth,
            learning_rate=gbm_learning_rate,
            subsample=gbm_subsample,
            colsample_bytree=gbm_colsample,
            random_state=random_state,
            early_stopping_rounds=gbm_early_stopping,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}. "
                         f"Choose from: linear, lstm, lstm_multistep, transformer, "
                         f"xgboost, lightgbm, prophet, tft")
    
    if batch_size:
        # For LSTM and Transformer models, use batched training
        model.fit(train_features_norm, train_targets, batch_size=batch_size)
        train_preds = model.predict(train_features_norm, batch_size=batch_size)
        val_preds = model.predict(val_features_norm, batch_size=batch_size)
    elif model_type in ('xgboost', 'lightgbm') and gbm_early_stopping is not None:
        # Pass validation set for early stopping
        model.fit(
            train_features_norm, train_targets,
            eval_set=[(val_features_norm, val_targets)],
        )
        train_preds = model.predict(train_features_norm)
        val_preds = model.predict(val_features_norm)
    else:
        # Linear, XGBoost (no early stopping), LightGBM (no early stopping)
        model.fit(train_features_norm, train_targets)
        train_preds = model.predict(train_features_norm)
        val_preds = model.predict(val_features_norm)

    # Print feature importances for tree-based models
    if hasattr(model, 'feature_importances_') and model_type in ('xgboost', 'lightgbm'):
        fi = model.feature_importances_
        # Build flat feature names: f0_t0, f0_t1, ...
        n_timesteps = train_features_norm.shape[1]
        n_feats = train_features_norm.shape[2]
        flat_names = [f"{features[fi_idx % n_feats]}_t{fi_idx // n_feats}"
                      for fi_idx in range(n_timesteps * n_feats)]
        top_n = min(20, len(flat_names))
        top_idx = np.argsort(fi)[::-1][:top_n]
        print(f"\nTop-{top_n} feature importances ({model_type}):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank:2d}. {flat_names[idx]:<35s} {fi[idx]:.4f}")
    
    # Calculate metrics
    model_metrics = {
        'train': evaluate_model(train_targets, train_preds, task_type, decision_threshold),
        'val':   evaluate_model(val_targets,   val_preds,   task_type, decision_threshold),
    }

    baseline_metrics = get_baseline_metrics(train_targets, val_targets, task_type,
                                            decision_threshold)
    
    # Print results
    print_results(model_metrics, baseline_metrics, task_type)
    
    # Return metrics for saving to CSV
    result = {
        'task': task,
        'model_type': model_type,
        'include_treatments': include_treatments,
        'prediction_horizon': prediction_horizon,
        'regularization': regularization,
        'alpha': alpha
    }
    
    # Add model metrics
    for split in ['train', 'val']:
        for metric, value in model_metrics[split].items():
            result[f'{split}_{metric}'] = value
    
    # Add baseline metrics
    for split in ['train', 'val']:
        for metric, value in baseline_metrics[split].items():
            result[f'{split}_baseline_{metric}'] = value
            
    return result

def run_all_experiments():
    """Run experiments with different configurations and save results to CSV"""
    # Define tasks and their types
    tasks = {
        'morta_hosp': 'static',      # Static outcome
        'los': 'static',             # Static outcome
        'septic_shock': 'temporal',  # Time-varying outcome
        'vasopressor': 'temporal',   # Time-varying outcome
        # Score regression tasks
        'sofa_score': 'temporal',
        'sirs_score': 'temporal',
        'news2_score': 'temporal',
    }
    
    model_types = ['linear', 'lstm', 'transformer']
    treatment_options = [True, False]
    
    # Set fixed prediction horizon for temporal tasks
    fixed_prediction_horizon = 6  # Hours ahead to predict
    
    results = []
    
    # Calculate total experiments
    total_experiments = 0
    for task, task_type in tasks.items():
        if task_type == 'static':
            total_experiments += len(model_types) * len(treatment_options)
        else:  # temporal
            total_experiments += len(model_types) * len(treatment_options)
    
    experiment_count = 0
    
    # Run experiments for all tasks
    for task, task_type in tasks.items():
        for model_type in model_types:
            for include_treatments in treatment_options:
                if task_type == 'static':
                    # For static tasks, run once with no prediction horizon
                    experiment_count += 1
                    print(f"\n\n{'='*80}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
                    print(f"{'='*80}\n")
                    
                    result = run_benchmark(
                        task=task,
                        model_type=model_type,
                        include_treatments=include_treatments,
                        prediction_horizon=None
                    )
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_df = pd.DataFrame(results)
                    results_df.to_csv("benchmark_results.csv", index=False)
                    print(f"Results saved to benchmark_results.csv")
                else:
                    # For temporal tasks, use fixed prediction horizon
                    experiment_count += 1
                    print(f"\n\n{'='*80}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
                    print(f"Prediction Horizon: {fixed_prediction_horizon} hours")
                    print(f"{'='*80}\n")
                    
                    result = run_benchmark(
                        task=task,
                        model_type=model_type,
                        include_treatments=include_treatments,
                        prediction_horizon=fixed_prediction_horizon
                    )
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_df = pd.DataFrame(results)
                    results_df.to_csv("benchmark_results.csv", index=False)
                    print(f"Results saved to benchmark_results.csv")
    
    return results

def run_selected_experiments(task: str, include_treatments: bool = False,
                              balance: bool = False,
                              balance_strategy: str = 'undersample',
                              score_thresholds: dict = None,
                              decision_threshold: float = 0.5):
    """Run experiments with all models for a specific task and treatment setting"""
    model_types = ['linear', 'lstm', 'transformer']

    # Determine if this is a temporal task
    temporal_tasks = ['septic_shock', 'mechvent', 'sepsis', 'vasopressor',
                      'sofa_score', 'sirs_score', 'news2_score']
    is_temporal = task in temporal_tasks

    # Define prediction horizons for temporal tasks
    prediction_horizons = [1, 2, 3, 4, 5, 6] if is_temporal else [None]
    
    results = []
    
    total_experiments = len(model_types) * len(prediction_horizons)
    experiment_count = 0
    
    for model_type in model_types:
        for horizon in prediction_horizons:
            experiment_count += 1
            print(f"\n\n{'='*80}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
            if is_temporal:
                print(f"Prediction Horizon: {horizon} hours")
            print(f"{'='*80}\n")
            
            result = run_benchmark(
                task=task,
                model_type=model_type,
                include_treatments=include_treatments,
                prediction_horizon=horizon,
                score_thresholds=score_thresholds,
                decision_threshold=decision_threshold,
            )
            results.append(result)
            
            # Save intermediate results after each experiment
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{task}_benchmark_results.csv", index=False)
            print(f"Results saved to {task}_benchmark_results.csv")
    
    return results

if __name__ == "__main__":
    # Log GPU info and set seeds
    _log_gpu_info()
    set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device: auto (default), cpu, or cuda")
    parser.add_argument("--run_all", action="store_true", help="Run all experiments")
    parser.add_argument("--run_selected", action="store_true", help="Run all models for a specific task")
    parser.add_argument("--task", type=str, default="mechvent", help="Target column name")
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["linear", "lstm", "lstm_multistep", "transformer",
                                 "xgboost", "lightgbm", "prophet", "tft"],
                        help="Model type (lstm_multistep/prophet/tft output H steps ahead)")
    parser.add_argument("--include_treatments", type=bool, default=False,
                        help="Whether to include treatment variables")
    parser.add_argument("--prediction_horizon", type=int, default=6,
                        help="Prediction horizon for temporal tasks (hours)")
    parser.add_argument("--regularization", type=str, default="ridge",
                        choices=["ridge", "lasso", "elasticnet", "none"],
                        help="Regularization type for linear models")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--balance", action="store_true", default=False,
                        help="Apply class / strata balancing to extracted windows before training")
    parser.add_argument("--balance_strategy", type=str, default="undersample",
                        choices=["undersample", "oversample", "combined"],
                        help="Balancing strategy (default: undersample)")
    parser.add_argument("--data_path", type=str,
                        default="processed_files/patient_timeseries_v4.csv",
                        help="Path to patient timeseries CSV (use _balanced.csv for pre-balanced data)")
    # ── Score threshold-exceedance parameters ───────────────────────────────
    parser.add_argument("--sofa_threshold", type=int, default=2,
                        help="SOFA score exceedance threshold (default: 2 = Sepsis-3 criterion)")
    parser.add_argument("--sirs_threshold", type=int, default=2,
                        help="SIRS score exceedance threshold (default: 2 = SIRS criterion)")
    parser.add_argument("--news2_threshold", type=int, default=5,
                        help="NEWS2 score exceedance threshold (default: 5 = medium clinical risk)")
    parser.add_argument("--decision_threshold", type=float, default=0.5,
                        help="Probability cut-off for hard binary prediction (default: 0.5). "
                             "Only used for classification tasks. Affects Accuracy; "
                             "AUROC and AUPRC are threshold-free.")
    # ── XGBoost / LightGBM hyperparameters ──────────────────────────────────
    parser.add_argument("--gbm_n_estimators", type=int, default=300,
                        help="[xgboost/lightgbm] Number of boosting rounds (default: 300)")
    parser.add_argument("--gbm_max_depth", type=int, default=6,
                        help="[xgboost/lightgbm] Max tree depth; -1 = unlimited for LightGBM (default: 6)")
    parser.add_argument("--gbm_learning_rate", type=float, default=0.05,
                        help="[xgboost/lightgbm] Learning rate / shrinkage (default: 0.05)")
    parser.add_argument("--gbm_subsample", type=float, default=0.8,
                        help="[xgboost/lightgbm] Row sub-sampling ratio (default: 0.8)")
    parser.add_argument("--gbm_colsample", type=float, default=0.8,
                        help="[xgboost/lightgbm] Column sub-sampling ratio per tree (default: 0.8)")
    parser.add_argument("--gbm_early_stopping", type=int, default=None,
                        help="[xgboost/lightgbm] Early stopping rounds (default: disabled)")
    # ── TFT / LSTM-multistep hyperparameters ────────────────────────────────
    parser.add_argument("--tft_max_epochs", type=int, default=30,
                        help="[tft] Max training epochs (default: 30)")
    parser.add_argument("--tft_hidden_size", type=int, default=32,
                        help="[tft] Hidden layer size (default: 32)")
    parser.add_argument("--tft_attention_heads", type=int, default=4,
                        help="[tft] Number of attention heads (default: 4)")
    parser.add_argument("--tft_dropout", type=float, default=0.1,
                        help="[tft] Dropout rate (default: 0.1)")
    parser.add_argument("--tft_batch_size", type=int, default=64,
                        help="[tft] Batch size (default: 64)")
    parser.add_argument("--output_csv", type=str,
                        default="results/score_benchmark.csv",
                        help="CSV file to append results to (created if missing, default: results/score_benchmark.csv)")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Free-form label added as a 'run_tag' column for tracking experiment variants")

    args = parser.parse_args()

    # Call this function at the beginning of your main function or script
    set_random_seeds(args.random_state)

    TEMPORAL_TASKS = ['septic_shock', 'mechvent', 'sepsis', 'vasopressor',
                      'sofa_score', 'sirs_score', 'news2_score']

    # Build per-score thresholds dict from CLI args
    _score_thresholds = {
        'sofa_score':  args.sofa_threshold,
        'sirs_score':  args.sirs_threshold,
        'news2_score': args.news2_threshold,
    }

    if args.run_all:
        run_all_experiments()
    elif args.run_selected:
        run_selected_experiments(
            args.task,
            args.include_treatments,
            score_thresholds=_score_thresholds,
            decision_threshold=args.decision_threshold,
        )
    else:
        result = run_benchmark(
            args.task,
            args.model_type,
            args.include_treatments,
            prediction_horizon=args.prediction_horizon if args.task in TEMPORAL_TASKS else None,
            random_state=args.random_state,
            regularization=args.regularization,
            alpha=args.alpha,
            balance=args.balance,
            balance_strategy=args.balance_strategy,
            data_path=args.data_path,
            score_thresholds=_score_thresholds,
            decision_threshold=args.decision_threshold,
            gbm_n_estimators=args.gbm_n_estimators,
            gbm_max_depth=args.gbm_max_depth,
            gbm_learning_rate=args.gbm_learning_rate,
            gbm_subsample=args.gbm_subsample,
            gbm_colsample=args.gbm_colsample,
            gbm_early_stopping=args.gbm_early_stopping,
            lstm_output_dim=args.prediction_horizon if args.model_type == 'lstm_multistep' else 1,
            tft_max_epochs=args.tft_max_epochs,
            tft_hidden_size=args.tft_hidden_size,
            tft_attention_heads=args.tft_attention_heads,
            tft_dropout=args.tft_dropout,
            tft_batch_size=args.tft_batch_size,
        )
        # Append result to shared CSV with run_tag and timestamp
        result['run_tag'] = args.run_tag
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        out_path = args.output_csv
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        row_df = pd.DataFrame([result])
        write_header = not os.path.exists(out_path)
        row_df.to_csv(out_path, mode='a', index=False, header=write_header)
        print(f"Result appended to {out_path}")
