import argparse
import json
import numpy as np
import pandas as pd
import polars as pl
import os
from sklearn.impute import KNNImputer
import math
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Optional: Numba for sample_and_hold acceleration
NUMBA_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    print("Numba JIT available")
except ImportError:
    print("Numba not available, using pure Python loops")

# Optional: GPU
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available (CuPy)")
except ImportError:
    print("GPU not available, using CPU")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true')
    parser.add_argument("--output_dir", type=str, default="processed_files")
    parser.add_argument("--missing_threshold", type=float, default=0.8)
    parser.add_argument("--low_missing_threshold", type=float, default=0.05)
    parser.add_argument("--knn_neighbors", type=int, default=1)
    parser.add_argument("--knn_chunk_size", type=int, default=9999)
    parser.add_argument("--fluid_window", type=int, default=12)
    parser.add_argument("--min_fluid_threshold", type=float, default=2000)
    parser.add_argument("--map_threshold", type=float, default=65)
    parser.add_argument("--lactate_threshold", type=float, default=2)
    parser.add_argument("--timestep", type=int, default=4)
    parser.add_argument("--window_before", type=int, default=24)
    parser.add_argument("--window_after", type=int, default=72)
    parser.add_argument("--notes_dir", type=str, default="processed_files")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--balance", action='store_true', default=False,
                        help="If specified, also write a class-balanced version of the output CSV")
    parser.add_argument("--noise_ratio", type=float, default=0.10,
                        help="Fraction of non-scoring patients to inject as noise relative to primary cohort (default: 0.10)")
    parser.add_argument("--non_onset_cap", type=int, default=None,
                        help="Max number of non-onset (control) patients to include (default: None = all)")
    return parser.parse_args()


class ComputeBackend:
    def __init__(self, use_gpu=False, device_id=0):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device_id = device_id
        if self.use_gpu:
            cp.cuda.Device(device_id).use()
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            print(f"Using GPU device {device_id}: {props['name'].decode()}")
        else:
            print("Using CPU backend")

    @property
    def xp(self):
        return cp if self.use_gpu else np

    def to_gpu(self, arr):
        return cp.asarray(arr) if self.use_gpu else arr

    def to_cpu(self, arr):
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)


backend = None


# ============================================================
# NUMBA JIT FUNCTIONS (optional, falls back to pure Python)
# ============================================================

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _sample_hold_numba(col_values, charttimes, group_starts, group_ends, hold_period):
        """JIT-compiled sample and hold - deterministic, ~10x faster"""
        for g in range(len(group_starts)):
            g_start = group_starts[g]
            g_end = group_ends[g]
            last_value = np.nan
            last_time = 0.0

            for i in range(g_start, g_end):
                if not np.isnan(col_values[i]):
                    last_value = col_values[i]
                    last_time = charttimes[i]
                elif not np.isnan(last_value) and (charttimes[i] - last_time) <= hold_period:
                    col_values[i] = last_value

        return col_values
else:
    def _sample_hold_numba(col_values, charttimes, group_starts, group_ends, hold_period):
        """Pure Python fallback - same logic, deterministic"""
        for g in range(len(group_starts)):
            g_start = group_starts[g]
            g_end = group_ends[g]
            last_value = np.nan
            last_time = 0.0

            for i in range(g_start, g_end):
                if not np.isnan(col_values[i]):
                    last_value = col_values[i]
                    last_time = charttimes[i]
                elif not np.isnan(last_value) and (charttimes[i] - last_time) <= hold_period:
                    col_values[i] = last_value

        return col_values


# ============================================================
# DATA LOADING
# ============================================================

def load_processed_files():
    """Load with pandas (handles messy CSVs), keep as pandas for groupby.
    Use Polars only where it actually helps."""
    print('Loading processed files created from database using "preprocess.py"')
    files = {
        'stay': 'icustays.csv',
        'abx': 'abx_processed.csv',
        'bacterio': 'bacterio_processed.csv',
        'demog': 'demog_processed.csv',
        'ce': 'chartevents.csv',
        'MV': 'mechvent.csv',
        'fluid': 'fluid.csv',
        'vaso': 'vaso.csv',
        'UO': 'uo.csv',
        'labU': 'labu.csv',
        'onset': 'onset.csv',
        'non_onset': 'non_onset.csv',
    }

    data = {}
    for key, filename in files.items():
        filepath = f'processed_files/{filename}'
        print(f'  Loading {filename}...')
        data[key] = pd.read_csv(filepath, sep='|', low_memory=False)

    return data


def load_measurement_mappings():
    print('Loading measurement mappings')
    with open("src/ReferenceFiles/measurement_mappings.json", "r") as f:
        measurements = json.load(f)

    code_to_concept = {}
    for concept, info in measurements.items():
        for code in info['codes']:
            code_to_concept[code] = concept

    hold_times = {}
    for concept, info in measurements.items():
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']

    return measurements, code_to_concept, hold_times


# ============================================================
# PATIENT MEASUREMENT PROCESSING
# ============================================================

def process_patient_measurements_fast(ce_data, lab_data, mv_data, code_to_concept,
                                       icustayid, onset_time, winb4=24, winaft=72):
    """Process patient measurements using pandas groups (already filtered by stay_id)"""
    t_start = onset_time - winb4 * 3600
    t_end = onset_time + winaft * 3600

    frames = []

    if ce_data is not None and len(ce_data) > 0:
        temp = ce_data[(ce_data['charttime'] >= t_start) & (ce_data['charttime'] < t_end)]
        if len(temp) > 0:
            temp_mapped = temp[['charttime', 'itemid', 'valuenum']].copy()
            temp_mapped = temp_mapped.sort_values(['charttime', 'itemid'])
            temp_mapped['itemid'] = temp_mapped['itemid'].astype(int).astype(str)
            temp_mapped['concept'] = temp_mapped['itemid'].map(code_to_concept)
            temp_mapped = temp_mapped.dropna(subset=['concept'])
            if len(temp_mapped) > 0:
                pivot_ce = temp_mapped.pivot_table(
                    index='charttime', columns='concept', values='valuenum', aggfunc='last'
                )
                frames.append(pivot_ce)

    if lab_data is not None and len(lab_data) > 0:
        temp2 = lab_data[(lab_data['charttime'] >= t_start) & (lab_data['charttime'] < t_end)]
        if len(temp2) > 0:
            temp2_mapped = temp2[['charttime', 'itemid', 'valuenum']].copy()
            temp2_mapped = temp2_mapped.sort_values(['charttime', 'itemid'])
            temp2_mapped['itemid'] = temp2_mapped['itemid'].astype(int).astype(str)
            temp2_mapped['concept'] = temp2_mapped['itemid'].map(code_to_concept)
            temp2_mapped = temp2_mapped.dropna(subset=['concept'])
            if len(temp2_mapped) > 0:
                pivot_lab = temp2_mapped.pivot_table(
                    index='charttime', columns='concept', values='valuenum', aggfunc='last'
                )
                frames.append(pivot_lab)

    if mv_data is not None and len(mv_data) > 0:
        temp3 = mv_data[(mv_data['charttime'] >= t_start) & (mv_data['charttime'] < t_end)]
        if len(temp3) > 0:
            mv = temp3[['charttime', 'mechvent']].sort_values('charttime').drop_duplicates(
                subset=['charttime'], keep='last'
            )
            mv = mv.set_index('charttime')
            frames.append(mv)

    if not frames:
        return None

    patient_df = pd.concat(frames, axis=1)
    if patient_df.columns.duplicated().any():
        patient_df = patient_df.loc[:, ~patient_df.columns.duplicated(keep='last')]

    patient_df.index.name = 'charttime'
    patient_df = patient_df.reset_index()
    patient_df['stay_id'] = icustayid

    return patient_df

# ============================================================
# OUTLIER HANDLING
# ============================================================

def handle_outliers(df):
    """Handle outliers using Polars. Deterministic: pure conditional replacement."""
    global backend
    print('Handling outliers in patient timeseries data')

    pl_df = pl.from_pandas(df)

    outlier_rules = [
        ('weight_kg', 300, None),
        ('weight_lb', 660, None),
        ('heart_rate', 250, None),
        ('sbp_arterial', 300, None),
        ('map', 200, 0),
        ('dbp_arterial', 200, 0),
        ('respiratory_rate', 80, None),
        ('spo2', 150, None),
        ('oxygen_flow', 70, None),
        ('peep', 40, 0),
        ('tidal_volume', 1800, None),
        ('minute_volume', 50, None),
        ('potassium', 15, 1),
        ('sodium', 178, 95),
        ('chloride', 150, 70),
        ('glucose', 1000, 1),
        ('creatinine', 150, None),
        ('magnesium', 10, None),
        ('calcium_total', 20, None),
        ('calcium_ionized', 5, None),
        ('total_co2', 120, None),
        ('ast', 10000, None),
        ('alt', 10000, None),
        ('hemoglobin', 20, None),
        ('hematocrit', 65, None),
        ('wbc', 500, None),
        ('platelets', 2000, None),
        ('inr', 20, None),
        ('ph_arterial', 8, 6.7),
        ('arterial_o2_pressure', 700, None),
        ('arterial_co2_pressure', 200, None),
        ('arterial_base_excess', None, -50),
        ('lactic_acid', 30, None),
        ('bilirubin_total', 30, None),
    ]

    exprs = []
    for col, upper, lower in outlier_rules:
        if col not in pl_df.columns:
            continue
        if upper is not None and lower is not None:
            exprs.append(
                pl.when((pl.col(col) > upper) | (pl.col(col) < lower))
                  .then(None)
                  .otherwise(pl.col(col))
                  .alias(col)
            )
        elif upper is not None:
            exprs.append(
                pl.when(pl.col(col) > upper).then(None).otherwise(pl.col(col)).alias(col)
            )
        elif lower is not None:
            exprs.append(
                pl.when(pl.col(col) < lower).then(None).otherwise(pl.col(col)).alias(col)
            )

    if exprs:
        pl_df = pl_df.with_columns(exprs)

    # SpO2 cap at 100
    if 'spo2' in pl_df.columns:
        pl_df = pl_df.with_columns(
            pl.when(pl.col('spo2') > 100).then(100.0).otherwise(pl.col('spo2')).alias('spo2')
        )

    # Temperature: move misplaced values
    if 'temp_C' in pl_df.columns and 'temp_F' in pl_df.columns:
        pl_df = pl_df.with_columns([
            pl.when((pl.col('temp_C') > 90) & pl.col('temp_F').is_null())
              .then(pl.col('temp_C'))
              .otherwise(pl.col('temp_F'))
              .alias('temp_F'),
            pl.when(pl.col('temp_C') > 90)
              .then(None)
              .otherwise(pl.col('temp_C'))
              .alias('temp_C'),
        ])

    # FiO2: exact order (1) >100→null, (2) <1→*100, (3) <20→null
    if 'fio2' in pl_df.columns:
        pl_df = pl_df.with_columns(
            pl.when(pl.col('fio2') > 100).then(None).otherwise(pl.col('fio2')).alias('fio2')
        )
        pl_df = pl_df.with_columns(
            pl.when(pl.col('fio2').is_not_null() & (pl.col('fio2') < 1))
              .then(pl.col('fio2') * 100)
              .otherwise(pl.col('fio2'))
              .alias('fio2')
        )
        pl_df = pl_df.with_columns(
            pl.when(pl.col('fio2').is_not_null() & (pl.col('fio2') < 20))
              .then(None)
              .otherwise(pl.col('fio2'))
              .alias('fio2')
        )

    return pl_df.to_pandas()


# ============================================================
# GCS ESTIMATION
# ============================================================

def estimate_gcs_from_rass(df):
    """Memory-efficient GCS estimation - no Polars conversion."""
    if 'gcs' not in df.columns:
        df['gcs'] = np.nan
    if 'richmond_ras' not in df.columns:
        return df

    rass_to_gcs = {4: 15, 3: 15, 2: 15, 1: 15, 0: 15,
                   -1: 14, -2: 12, -3: 11, -4: 6, -5: 3}

    gcs = df['gcs'].values
    rass = df['richmond_ras'].values

    gcs_missing = np.isnan(gcs)

    for rass_val, gcs_val in rass_to_gcs.items():
        mask = gcs_missing & (rass == rass_val)
        gcs[mask] = gcs_val

    df['gcs'] = gcs
    return df


# ============================================================
# FiO2 ESTIMATION
# ============================================================

def estimate_fio2(df):
    """Memory-efficient FiO2 estimation - no Polars conversion."""
    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    existing_flow_cols = [c for c in flow_columns if c in df.columns]

    if existing_flow_cols:
        combined_flow = df[existing_flow_cols].bfill(axis=1).iloc[:, 0].values
    else:
        combined_flow = np.full(len(df), np.nan)

    if 'fio2' not in df.columns:
        df['fio2'] = np.nan

    if 'oxygen_flow_device' not in df.columns:
        return df

    fio2 = df['fio2'].values.astype(np.float64)
    device = df['oxygen_flow_device'].astype(str).values
    flow = combined_flow

    fio2_missing = np.isnan(fio2)
    has_flow = ~np.isnan(flow)

    # Case 1: nasal cannula with flow
    mask1 = fio2_missing & has_flow & np.isin(device, ['0', '2'])
    if mask1.any():
        f = flow[mask1]
        vals = np.full(mask1.sum(), 70.0)
        thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        values = [24, 28, 32, 36, 40, 44, 50, 55, 62, 70]
        for t, v in zip(thresholds, values):
            vals[f <= t] = v
        fio2[mask1] = vals

    # Case 2: no flow, nasal → room air
    mask2 = fio2_missing & (~has_flow) & np.isin(device, ['0', '2'])
    fio2[mask2] = 21

    fio2_missing = np.isnan(fio2)

    # Case 3: face mask
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12']
    mask3 = fio2_missing & has_flow & np.isin(device, face_mask_types)
    if mask3.any():
        f = flow[mask3]
        vals = np.full(mask3.sum(), 75.0)
        thresholds = [4, 6, 8, 10, 12, 15]
        values = [36, 40, 58, 66, 69, 75]
        for t, v in zip(thresholds, values):
            vals[f <= t] = v
        fio2[mask3] = vals

    fio2_missing = np.isnan(fio2)

    # Case 4: non-rebreather
    mask4 = fio2_missing & has_flow & (device == '7')
    if mask4.any():
        f = flow[mask4]
        fio2[mask4] = np.where(f >= 15, 100,
                     np.where(f >= 10, 90,
                     np.where(f > 8, 80,
                     np.where(f > 6, 70, 60))))

    fio2_missing = np.isnan(fio2)

    # Case 5: CPAP/BiPAP
    mask5 = fio2_missing & has_flow & (device == '13')
    if mask5.any():
        f = flow[mask5]
        fio2[mask5] = np.where(f >= 15, 100, np.where(f >= 10, 80, 60))

    fio2_missing = np.isnan(fio2)

    # Case 6: Oxymizer
    mask6 = fio2_missing & has_flow & (device == '14')
    if mask6.any():
        f = flow[mask6]
        fio2[mask6] = np.where(f >= 10, 80, np.where(f >= 5, 60, 40))

    df['fio2'] = fio2
    return df


# ============================================================
# UNIT CONVERSIONS
# ============================================================

def handle_unit_conversions(df):
    """Memory-efficient unit conversions - no Polars conversion."""
    if 'temp_F' in df.columns and 'temp_C' in df.columns:
        tc = df['temp_C'].values.astype(np.float64)
        tf = df['temp_F'].values.astype(np.float64)

        # tempF 25-45 looks like Celsius
        mask = (tf > 25) & (tf < 45)
        tc[mask] = tf[mask]
        tf[mask] = np.nan

        # tempC > 70 looks like Fahrenheit
        mask = tc > 70
        tf[mask] = tc[mask]
        tc[mask] = np.nan

        # C → F
        mask = (~np.isnan(tc)) & np.isnan(tf)
        tf[mask] = tc[mask] * 1.8 + 32

        # F → C
        mask = (~np.isnan(tf)) & np.isnan(tc)
        tc[mask] = (tf[mask] - 32) / 1.8

        df['temp_C'] = tc
        df['temp_F'] = tf

    if 'hemoglobin' in df.columns and 'hematocrit' in df.columns:
        hgb = df['hemoglobin'].values.astype(np.float64)
        hct = df['hematocrit'].values.astype(np.float64)

        mask = (~np.isnan(hgb)) & np.isnan(hct)
        hct[mask] = hgb[mask] * 2.862 + 1.216

        mask = (~np.isnan(hct)) & np.isnan(hgb)
        hgb[mask] = (hct[mask] - 1.216) / 2.862

        df['hemoglobin'] = hgb
        df['hematocrit'] = hct

    if 'bilirubin_total' in df.columns and 'bilirubin_direct' in df.columns:
        bt = df['bilirubin_total'].values.astype(np.float64)
        bd = df['bilirubin_direct'].values.astype(np.float64)

        mask = (~np.isnan(bt)) & np.isnan(bd)
        bd[mask] = bt[mask] * 0.6934 - 0.1752

        mask = (~np.isnan(bd)) & np.isnan(bt)
        bt[mask] = (bd[mask] + 0.1752) / 0.6934

        df['bilirubin_total'] = bt
        df['bilirubin_direct'] = bd

    return df


# ============================================================
# SAMPLE AND HOLD
# ============================================================

def sample_and_hold(df, vitalslab_hold):
    """Memory-efficient sample and hold - processes columns sequentially."""
    print(f'Performing sample and hold interpolation ({"Numba" if NUMBA_AVAILABLE else "Python"})')

    df = df.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    cols_to_process = [col for col in vitalslab_hold if col in df.columns
                       and np.issubdtype(df[col].dtype, np.number)]

    if not cols_to_process:
        return df

    # Extract these once (shared across all columns)
    stay_ids = df['stay_id'].values
    charttimes = df['charttime'].values.astype(np.float64)

    # Find group boundaries once
    stay_change = np.concatenate([[True], stay_ids[1:] != stay_ids[:-1]])
    group_starts = np.where(stay_change)[0].astype(np.int64)
    group_ends = np.concatenate([group_starts[1:], [len(df)]]).astype(np.int64)

    print(f'  {len(cols_to_process)} columns, {len(group_starts)} groups, {len(df)} rows')

    for col_idx, col in enumerate(cols_to_process):
        if (col_idx + 1) % 10 == 0:
            print(f'  Column {col_idx + 1}/{len(cols_to_process)}: {col}')

        hold_period = float(vitalslab_hold[col] * 3600)
        col_values = df[col].to_numpy(dtype=np.float64, copy=True)
        col_values = _sample_hold_numba(col_values, charttimes, group_starts, group_ends, hold_period)
        df[col] = col_values

    return df

# ============================================================
# COMBINE PATIENT DATA
# ============================================================

def combine_patient_data(patient_data, timestep=4, window_before=24, window_after=72):
    """Deterministic: fixed timestep grid, deterministic aggregation (mean)."""

    start_time = patient_data['start_time']
    stay_id = patient_data['stay_id']
    measurements = patient_data['measurements']
    fluid_data = patient_data['fluid']
    vaso_data = patient_data['vasopressors']
    uo_data = patient_data['urine_output']
    abx_data = patient_data['antibiotics']
    demographics = patient_data['demographics']

    if measurements is None or len(measurements) == 0:
        return None

    patient_times = measurements['charttime'].values
    if len(patient_times) == 0:
        return None

    patient_times_sorted = np.sort(patient_times)
    first_time = max(patient_times_sorted[0], start_time - window_before * 3600)
    last_time = min(patient_times_sorted[-1], start_time + window_after * 3600)

    total_hours = (last_time - first_time) / 3600
    num_timesteps = math.ceil(total_hours / timestep)

    if num_timesteps <= 0:
        return None

    window_starts = first_time + np.arange(num_timesteps) * timestep * 3600
    window_ends = window_starts + timestep * 3600

    meas_times = measurements['charttime'].values
    meas_cols = [c for c in measurements.columns if c not in ['stay_id']]

    # Precompute numpy arrays from supplementary data
    def _to_numpy(data, col):
        if data is None or len(data) == 0:
            return np.array([])
        if isinstance(data, pl.DataFrame):
            return data[col].to_numpy()
        return data[col].values

    abx_starts = _to_numpy(abx_data, 'starttime')
    abx_stops = _to_numpy(abx_data, 'stoptime')
    abx_drugs = _to_numpy(abx_data, 'drug')
    first_abx_time = float(abx_starts.min()) if len(abx_starts) > 0 else None

    fluid_starts = _to_numpy(fluid_data, 'starttime')
    fluid_ends = _to_numpy(fluid_data, 'endtime')
    fluid_amounts = _to_numpy(fluid_data, 'amount')

    vaso_starts = _to_numpy(vaso_data, 'starttime')
    vaso_ends = _to_numpy(vaso_data, 'endtime')
    vaso_rates = _to_numpy(vaso_data, 'rate_std')

    uo_times = _to_numpy(uo_data, 'charttime')
    uo_values_arr = _to_numpy(uo_data, 'value')

    processed_rows = []

    for idx in range(num_timesteps):
        ws = window_starts[idx]
        we = window_ends[idx]

        # Measurements - mean aggregation is deterministic for same input
        mask = (meas_times >= ws) & (meas_times < we)
        if mask.any():
            window_meas = measurements.loc[mask, meas_cols]
            meas_dict = window_meas.mean(axis=0, skipna=True).to_dict()
        else:
            meas_dict = {col: np.nan for col in meas_cols}
            meas_dict['charttime'] = (ws + we) / 2

        # Fluids
        if len(fluid_starts) > 0:
            step_mask = (fluid_starts < we) & (fluid_ends >= ws)
            fluid_step = float(fluid_amounts[step_mask].sum())
            total_mask = fluid_ends < we
            fluid_total = float(fluid_amounts[total_mask].sum())
        else:
            fluid_step = 0.0
            fluid_total = 0.0

        # Vasopressors
        if len(vaso_starts) > 0:
            vmask = (vaso_starts <= we) & (vaso_ends >= ws)
            if vmask.any():
                window_rates = vaso_rates[vmask]
                vaso_median = float(np.nanmedian(window_rates))
                vaso_max = float(np.nanmax(window_rates))
            else:
                vaso_median = 0.0
                vaso_max = 0.0
        else:
            vaso_median = 0.0
            vaso_max = 0.0

        # Urine output
        if len(uo_times) > 0:
            uo_mask = (uo_times >= ws) & (uo_times < we)
            uo_step = float(uo_values_arr[uo_mask].sum())
            uo_total_mask = uo_times < we
            uo_total = float(uo_values_arr[uo_total_mask].sum())
        else:
            uo_step = 0.0
            uo_total = 0.0

        # Antibiotics
        if first_abx_time is not None and len(abx_starts) > 0:
            abx_mask = (abx_starts <= we) & (abx_stops >= ws)
            abx_given = 1 if abx_mask.any() else 0
            hours_since_first_abx = (we - first_abx_time) / 3600
            num_abx = len(np.unique(abx_drugs[abx_mask])) if abx_mask.any() else 0
        else:
            abx_given = 0
            hours_since_first_abx = None
            num_abx = 0

        row = {
            'timestep': idx + 1,
            'stay_id': stay_id,
            'timestamp': ws,
            **demographics,
            **meas_dict,
            'fluid_total': fluid_total,
            'fluid_step': fluid_step,
            'uo_total': uo_total,
            'uo_step': uo_step,
            'balance': fluid_total - uo_total,
            'vaso_median': vaso_median,
            'vaso_max': vaso_max,
            'abx_given': abx_given,
            'hours_since_first_abx': hours_since_first_abx,
            'num_abx': num_abx,
        }
        processed_rows.append(row)

    return pd.DataFrame(processed_rows)


# ============================================================
# STANDARDIZE PATIENT TRAJECTORIES
# ============================================================

def standardize_patient_trajectories(init_traj, data_dict, timestep=4, window_before=24, window_after=72):
    """Deterministic: processes patients in sorted stay_id order."""
    print('Processing all patients with fixed time windows')
    all_patient_data = []

    # Build lookups from pandas
    onset_lookup = dict(zip(data_dict['onset']['stay_id'], data_dict['onset']['onset_time']))
    demog_pd = data_dict['demog'].set_index('stay_id')

    # Pre-group with pandas (fast, won't hang)
    print('  Grouping supplementary data...')
    fluid_groups = dict(list(data_dict['fluid'].groupby('stay_id')))
    vaso_groups = dict(list(data_dict['vaso'].groupby('stay_id')))
    uo_groups = dict(list(data_dict['UO'].groupby('stay_id')))
    abx_groups = dict(list(data_dict['abx'].groupby('stay_id')))
    traj_groups = dict(list(init_traj.groupby('stay_id')))

    # DETERMINISTIC: sorted order
    stay_ids = sorted(traj_groups.keys())

    print(f'  Processing {len(stay_ids)} patients')
    count = 0
    for stay_id in stay_ids:
        count += 1
        if count % 500 == 0:
            print(f'    Processed {count}/{len(stay_ids)} patients')

        if stay_id not in onset_lookup:
            continue

        start_time = onset_lookup[stay_id]

        try:
            demog_row = demog_pd.loc[stay_id]
            if isinstance(demog_row, pd.DataFrame):
                demog_row = demog_row.iloc[0]
            demographics = demog_row.to_dict()
        except KeyError:
            continue

        patient_data = {
            'stay_id': stay_id,
            'start_time': start_time,
            'measurements': traj_groups.get(stay_id, pd.DataFrame()),
            'demographics': demographics,
            'fluid': fluid_groups.get(stay_id, pd.DataFrame()),
            'vasopressors': vaso_groups.get(stay_id, pd.DataFrame()),
            'urine_output': uo_groups.get(stay_id, pd.DataFrame()),
            'antibiotics': abx_groups.get(stay_id, pd.DataFrame()),
        }

        processed_patient = combine_patient_data(
            patient_data, timestep=timestep,
            window_before=window_before, window_after=window_after
        )
        if processed_patient is not None:
            all_patient_data.append(processed_patient)

    return pd.concat(all_patient_data, ignore_index=True)


# ============================================================
# MISSING VALUES
# ============================================================

def fixgaps(x: np.ndarray) -> np.ndarray:
    """Deterministic linear interpolation."""
    y = np.copy(x)
    nan_mask = np.isnan(x)
    valid_indices = np.where(~nan_mask)[0]

    if len(valid_indices) < 2:
        return y

    interp_mask = nan_mask.copy()
    interp_mask[:valid_indices[0]] = False
    interp_mask[valid_indices[-1] + 1:] = False

    if interp_mask.any():
        y[interp_mask] = np.interp(
            np.where(interp_mask)[0],
            valid_indices,
            x[valid_indices]
        )

    return y


def handle_missing_values(df, missing_threshold=0.8):
    """Deterministic: KNNImputer with fixed chunk boundaries and sorted input."""
    print('Handling missing values...')

    measurement_cols = [col for col in df.columns if col not in [
        'timestep', 'stay_id', 'timestamp', 'gender', 'age',
        'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 'fluid_total', 'fluid_step',
        'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]]

    # Print statistics
    print("\nMissingness statistics before imputation:")
    print("-" * 50)
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_numeric_cols = [col for col in all_numeric_cols if col not in measurement_cols]

    miss_stats_meas = df[measurement_cols].isna().sum() / len(df)
    miss_stats_excl = df[excluded_numeric_cols].isna().sum() / len(df)

    print("Measurement columns to be imputed:")
    print(f"{'Variable':<30} {'Missing %':>10}")
    print("-" * 50)
    for var, miss_pct in miss_stats_meas.sort_values(ascending=False).items():
        print(f"{var:<30} {miss_pct:>10.1%}")

    print("\nExcluded numeric columns:")
    print(f"{'Variable':<30} {'Missing %':>10}")
    print("-" * 50)
    for var, miss_pct in miss_stats_excl.sort_values(ascending=False).items():
        print(f"{var:<30} {miss_pct:>10.1%}")
    print("-" * 50)

    # Score-critical columns that must never be dropped regardless of missingness
    PROTECTED_SCORE_COLS = {
        # SOFA
        'arterial_o2_pressure', 'fio2', 'mechvent', 'platelets',
        'bilirubin_total', 'map', 'gcs', 'creatinine',
        # SIRS
        'temp_C', 'temp_F', 'heart_rate', 'respiratory_rate',
        'arterial_co2_pressure', 'wbc',
        # NEWS2 extras
        'spo2', 'oxygen_flow', 'oxygen_flow_device', 'sbp_arterial', 'richmond_ras',
    }

    miss = df[measurement_cols].isna().sum() / len(df)

    # Drop high-missing columns — but NEVER drop score-critical columns
    cols_to_keep = sorted([
        col for col in miss.index
        if miss[col] < missing_threshold or col in PROTECTED_SCORE_COLS
    ])
    non_meas_cols = [c for c in df.columns if c not in measurement_cols]
    df = df[non_meas_cols + cols_to_keep]

    protected_kept = [c for c in PROTECTED_SCORE_COLS if c in df.columns]
    print(f"Protected score columns retained: {len(protected_kept)} -> {protected_kept}")

    # Linear interpolation (deterministic: np.interp is deterministic)
    low_missing_cols = sorted(miss[(miss > 0) & (miss < 0.05)].index.tolist())
    low_missing_cols = [c for c in low_missing_cols if c in df.columns]
    for col in low_missing_cols:
        df[col] = fixgaps(df[col].values)

    # KNN imputation — includes protected cols that exceed normal threshold
    cols_for_knn = sorted([c for c in cols_to_keep if c not in low_missing_cols and c in df.columns])
    if cols_for_knn:
        ref = df[cols_for_knn].values.astype(np.float64)

        chunk_size = 9999
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        print(f'KNN imputation: {total_chunks} chunks')

        imputer = KNNImputer(n_neighbors=1, weights='uniform')

        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            chunk = ref[i:chunk_end, :]

            if np.isnan(chunk).any():
                ref[i:chunk_end, :] = imputer.fit_transform(chunk)

            chunk_idx = i // chunk_size + 1
            if chunk_idx % 10 == 0 or chunk_idx == total_chunks:
                print(f'  Chunk {chunk_idx}/{total_chunks}')

        df[cols_for_knn] = ref

    return df


# ============================================================
# DERIVED VARIABLES
# ============================================================

def calculate_derived_variables(df):
    """Deterministic: vectorized comparisons produce same result for same input."""
    global backend
    print(f'Computing derived variables ({"GPU" if backend.use_gpu else "CPU"})')

    df = df.copy()

    if 'gender' in df.columns:
        df['gender'] = df['gender'] - 1
    if 'age' in df.columns:
        df.loc[df['age'] > 150, 'age'] = 91.4

    if 'mechvent' in df.columns:
        df['mechvent'] = df['mechvent'].fillna(0)
        df.loc[df['mechvent'] > 0, 'mechvent'] = 1

    if 'charlson_comorbidity_index' in df.columns:
        df['charlson_comorbidity_index'] = df['charlson_comorbidity_index'].fillna(
            df['charlson_comorbidity_index'].median()
        )

    df['vaso_median'] = df['vaso_median'].fillna(0)
    df['vaso_max'] = df['vaso_max'].fillna(0)

    n = len(df)
    xp = backend.xp

    def get_col(col_name):
        if col_name in df.columns:
            arr = df[col_name].values.astype(np.float64)
        else:
            arr = np.full(n, np.nan)
        return backend.to_gpu(arr) if backend.use_gpu else arr

    # P/F ratio
    if 'arterial_o2_pressure' in df.columns and 'fio2' in df.columns:
        ao2 = get_col('arterial_o2_pressure')
        fio2 = get_col('fio2')
        pf_ratio = ao2 / (fio2 / 100)
        df['pf_ratio'] = backend.to_cpu(pf_ratio)
    else:
        df['pf_ratio'] = np.nan

    # Shock Index
    if 'heart_rate' in df.columns and 'sbp_arterial' in df.columns:
        hr = get_col('heart_rate')
        sbp = get_col('sbp_arterial')
        si = hr / sbp
        si_cpu = backend.to_cpu(si)
        si_cpu[np.isinf(si_cpu)] = np.nan
        si_mean = np.nanmean(si_cpu)
        si_cpu[np.isnan(si_cpu)] = si_mean
        df['shock_index'] = si_cpu
    else:
        df['shock_index'] = np.nan

    # SOFA - vectorized, deterministic
    pf = get_col('pf_ratio')
    plt_vals = get_col('platelets')
    bili = get_col('bilirubin_total')
    map_vals = get_col('map')
    vaso_max_arr = get_col('vaso_max')
    gcs_vals = get_col('gcs')
    cr_vals = get_col('creatinine')
    uo_vals = get_col('uo_step')

    sofa_resp = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(pf)
    sofa_resp[valid & (pf < 100)] = 4
    sofa_resp[valid & (pf >= 100) & (pf < 200)] = 3
    sofa_resp[valid & (pf >= 200) & (pf < 300)] = 2
    sofa_resp[valid & (pf >= 300) & (pf < 400)] = 1

    sofa_coag = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(plt_vals)
    sofa_coag[valid & (plt_vals < 20)] = 4
    sofa_coag[valid & (plt_vals >= 20) & (plt_vals < 50)] = 3
    sofa_coag[valid & (plt_vals >= 50) & (plt_vals < 100)] = 2
    sofa_coag[valid & (plt_vals >= 100) & (plt_vals < 150)] = 1

    sofa_liver = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(bili)
    sofa_liver[valid & (bili >= 12)] = 4
    sofa_liver[valid & (bili >= 6) & (bili < 12)] = 3
    sofa_liver[valid & (bili >= 2) & (bili < 6)] = 2
    sofa_liver[valid & (bili >= 1.2) & (bili < 2)] = 1

    sofa_cv = xp.zeros(n, dtype=xp.int32)
    valid_map = ~xp.isnan(map_vals)
    map_na = xp.isnan(map_vals)
    vaso_na = xp.isnan(vaso_max_arr)
    sofa_cv[valid_map & (map_vals < 70) & (map_vals >= 65)] = 1
    sofa_cv[valid_map & (map_vals < 65)] = 2
    sofa_cv[map_na & ~vaso_na & (vaso_max_arr <= 0.1)] = 3
    sofa_cv[map_na & ~vaso_na & (vaso_max_arr > 0.1)] = 4

    sofa_cns = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(gcs_vals)
    sofa_cns[valid & (gcs_vals <= 5)] = 4
    sofa_cns[valid & (gcs_vals > 5) & (gcs_vals <= 9)] = 3
    sofa_cns[valid & (gcs_vals > 9) & (gcs_vals <= 12)] = 2
    sofa_cns[valid & (gcs_vals > 12) & (gcs_vals <= 14)] = 1

    sofa_renal = xp.zeros(n, dtype=xp.int32)
    valid_cr = ~xp.isnan(cr_vals)
    sofa_renal[valid_cr & (cr_vals >= 5)] = 4
    sofa_renal[valid_cr & (cr_vals >= 3.5) & (cr_vals < 5)] = 3
    sofa_renal[valid_cr & (cr_vals >= 2.0) & (cr_vals < 3.5)] = 2
    sofa_renal[valid_cr & (cr_vals >= 1.2) & (cr_vals < 2.0)] = 1
    no_cr = xp.isnan(cr_vals)
    valid_uo = ~xp.isnan(uo_vals)
    sofa_renal[no_cr & valid_uo & (uo_vals < 34)] = 4
    sofa_renal[no_cr & valid_uo & (uo_vals >= 34) & (uo_vals < 84)] = 3

    df['sofa_resp'] = backend.to_cpu(sofa_resp).astype(int)
    df['sofa_coag'] = backend.to_cpu(sofa_coag).astype(int)
    df['sofa_liver'] = backend.to_cpu(sofa_liver).astype(int)
    df['sofa_cv'] = backend.to_cpu(sofa_cv).astype(int)
    df['sofa_cns'] = backend.to_cpu(sofa_cns).astype(int)
    df['sofa_renal'] = backend.to_cpu(sofa_renal).astype(int)

    df['sofa_score'] = df['sofa_resp'] + df['sofa_coag'] + df['sofa_liver'] + \
                       df['sofa_cv'] + df['sofa_cns'] + df['sofa_renal']

    # SIRS
    sirs = xp.zeros(n, dtype=xp.int32)

    if 'temp_C' in df.columns:
        tc = get_col('temp_C')
        valid = ~xp.isnan(tc)
        sirs[valid & ((tc >= 38) | (tc <= 36))] += 1

    if 'heart_rate' in df.columns:
        hr = get_col('heart_rate')
        valid = ~xp.isnan(hr)
        sirs[valid & (hr > 90)] += 1

    rr = get_col('respiratory_rate')
    co2 = get_col('arterial_co2_pressure')
    valid_rr = ~xp.isnan(rr)
    valid_co2 = ~xp.isnan(co2)
    resp_crit = (valid_rr & (rr >= 20)) | (valid_co2 & (co2 <= 32))
    sirs[resp_crit] += 1

    if 'wbc' in df.columns:
        wbc = get_col('wbc')
        valid = ~xp.isnan(wbc)
        sirs[valid & ((wbc >= 12) | (wbc < 4))] += 1

    df['sirs_score'] = backend.to_cpu(sirs).astype(int)

    # ------------------------------------------------------------------ #
    # NEWS2 Score (National Early Warning Score 2) — CPU path             #
    # Uses Scale 1 SpO2 (standard; not hypercapnic respiratory failure)   #
    # ------------------------------------------------------------------ #
    news2 = np.zeros(len(df), dtype=int)

    if 'respiratory_rate' in df.columns:
        rr_n = df['respiratory_rate'].values
        valid = ~np.isnan(rr_n)
        news2[valid & (rr_n <= 8)] += 3
        news2[valid & (rr_n >= 9) & (rr_n <= 11)] += 1
        news2[valid & (rr_n >= 21) & (rr_n <= 24)] += 2
        news2[valid & (rr_n >= 25)] += 3

    if 'spo2' in df.columns:
        sp = df['spo2'].values
        valid = ~np.isnan(sp)
        news2[valid & (sp <= 91)] += 3
        news2[valid & (sp >= 92) & (sp <= 93)] += 2
        news2[valid & (sp >= 94) & (sp <= 95)] += 1

    if 'fio2' in df.columns:
        fio2_n = df['fio2'].values
        valid = ~np.isnan(fio2_n)
        news2[valid & (fio2_n > 21)] += 2

    if 'sbp_arterial' in df.columns:
        sbp = df['sbp_arterial'].values
        valid = ~np.isnan(sbp)
        news2[valid & (sbp <= 90)] += 3
        news2[valid & (sbp >= 91) & (sbp <= 100)] += 2
        news2[valid & (sbp >= 101) & (sbp <= 110)] += 1
        news2[valid & (sbp >= 220)] += 3

    if 'heart_rate' in df.columns:
        hr_n = df['heart_rate'].values
        valid = ~np.isnan(hr_n)
        news2[valid & (hr_n <= 40)] += 3
        news2[valid & (hr_n >= 41) & (hr_n <= 50)] += 1
        news2[valid & (hr_n >= 91) & (hr_n <= 110)] += 1
        news2[valid & (hr_n >= 111) & (hr_n <= 130)] += 2
        news2[valid & (hr_n >= 131)] += 3

    if 'gcs' in df.columns:
        gcs_n = df['gcs'].values
        valid = ~np.isnan(gcs_n)
        news2[valid & (gcs_n < 15)] += 3

    if 'temp_C' in df.columns:
        tc_n = df['temp_C'].values
        valid = ~np.isnan(tc_n)
        news2[valid & (tc_n <= 35.0)] += 3
        news2[valid & (tc_n >= 35.1) & (tc_n <= 36.0)] += 1
        news2[valid & (tc_n >= 38.1) & (tc_n <= 39.0)] += 1
        news2[valid & (tc_n >= 39.1)] += 2

    df['news2_score'] = news2

    for comp in ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_cns', 'sofa_renal']:
        print(f"\n{comp} distribution:")
        print(df[comp].value_counts().sort_index())

    print(f"\nnews2_score distribution:")
    print(pd.Series(news2).value_counts().sort_index())

    return df


# ============================================================
# EXCLUSION CRITERIA
# ============================================================

def apply_exclusion_criteria(df, noise_ratio=0.10):
    """Multi-score gate + controlled noise injection (first copy).

    Patients are retained if they reach a clinical threshold on AT LEAST ONE of:
      - SOFA >= 2  (organ dysfunction / Sepsis-3)
      - SIRS >= 2  (systemic inflammation)
      - NEWS2 >= 5 (medium clinical risk)

    A controlled fraction (noise_ratio) of non-scoring patients is then
    re-injected as negative examples to improve model robustness.
    """
    print('Applying exclusion criteria')

    initial_patients = df['stay_id'].nunique()
    excluded_counts = {}

    # Extreme UO
    extreme_uo_stays = df.loc[df['uo_step'] > 12000, 'stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_uo_stays)]
    excluded_counts['extreme_uo'] = len(extreme_uo_stays)

    # Extreme fluid
    extreme_fluid_stays = df.loc[df['fluid_step'] > 10000, 'stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_fluid_stays)]
    excluded_counts['extreme_fluid'] = len(extreme_fluid_stays)

    # Early deaths
    patient_time_range = df.groupby('stay_id').agg(
        min_time=('timestamp', 'min'),
        max_time=('timestamp', 'max'),
        morta_hosp=('morta_hosp', 'first')
    )
    patient_time_range['duration_hours'] = (patient_time_range['max_time'] - patient_time_range['min_time']) / 3600
    early_death_stays = patient_time_range[
        (patient_time_range['morta_hosp'] == 1) & (patient_time_range['duration_hours'] <= 24)
    ].index.values
    df = df[~df['stay_id'].isin(early_death_stays)]
    excluded_counts['early_death'] = len(early_death_stays)

    # Multi-score gate: keep patients that score on at least one system
    score_cols = {}
    if 'sofa_score' in df.columns:
        score_cols['max_sofa'] = ('sofa_score', 'max')
    if 'sirs_score' in df.columns:
        score_cols['max_sirs'] = ('sirs_score', 'max')
    if 'news2_score' in df.columns:
        score_cols['max_news2'] = ('news2_score', 'max')

    if score_cols:
        max_scores = df.groupby('stay_id').agg(**score_cols)

        eligible_mask = pd.Series(False, index=max_scores.index)
        if 'max_sofa' in max_scores.columns:
            eligible_mask |= (max_scores['max_sofa'] >= 2)
        if 'max_sirs' in max_scores.columns:
            eligible_mask |= (max_scores['max_sirs'] >= 2)
        if 'max_news2' in max_scores.columns:
            eligible_mask |= (max_scores['max_news2'] >= 5)

        eligible_stays   = max_scores[eligible_mask].index.values
        ineligible_stays = max_scores[~eligible_mask].index.values

        noise_n = max(1, int(len(eligible_stays) * noise_ratio))
        rng = np.random.default_rng(42)
        noise_stays = rng.choice(
            ineligible_stays,
            size=min(noise_n, len(ineligible_stays)),
            replace=False
        ) if len(ineligible_stays) > 0 else np.array([], dtype=ineligible_stays.dtype)

        keep_stays = np.concatenate([eligible_stays, noise_stays])
        df = df[df['stay_id'].isin(keep_stays)]

        excluded_counts['no_score_threshold'] = len(ineligible_stays) - len(noise_stays)
        excluded_counts['noise_injected']      = len(noise_stays)
    else:
        print("WARNING: No score columns found — skipping multi-score gate")

    final_patients = df['stay_id'].nunique()
    print("\nExclusion Statistics:")
    print("-" * 50)
    print(f"Initial patient count: {initial_patients}")
    for reason, count in excluded_counts.items():
        print(f"  {reason}: {count}")
    print(f"Final patient count: {final_patients}")
    print(f"Total excluded (net): {initial_patients - final_patients}")
    print("-" * 50)

    # Free memory
    gc.collect()

    return df


# ============================================================
# SEPSIS / SHOCK FLAGS
# ============================================================

def add_sepsis_flag(df):
    """Memory-efficient sepsis flag."""
    print('Adding sepsis flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['sepsis'] = 0

    # Find first SOFA >= 2 per patient
    sepsis_mask = df['sofa_score'] >= 2
    sepsis_first = df[sepsis_mask].groupby('stay_id').head(1)

    df.loc[sepsis_first.index, 'sepsis'] = 1

    # Mark censored using numpy for speed
    stay_ids_arr = df['stay_id'].values
    idx_arr = np.arange(len(df))
    onset_dict = dict(zip(sepsis_first['stay_id'].values, sepsis_first.index.values))

    for stay_id, onset_idx in onset_dict.items():
        mask = (stay_ids_arr == stay_id) & (idx_arr > onset_idx)
        df.loc[mask, 'sepsis'] = 2

    total_patients = df['stay_id'].nunique()
    sepsis_patients = len(onset_dict)

    print("\nSepsis Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing sepsis: {sepsis_patients} ({sepsis_patients / max(total_patients, 1) * 100:.1f}%)")
    print(f"Timesteps with sepsis onset: {(df['sepsis'] == 1).sum()}")
    print(f"Censored timesteps: {(df['sepsis'] == 2).sum()}")
    print("-" * 50)

    return df


def add_septic_shock_flag(df):
    """Memory-efficient septic shock flag."""
    print('Adding septic shock flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['septic_shock'] = 0

    TIMESTEP_SIZE = 4
    FLUID_WINDOW = 12
    WINDOW_STEPS = max(1, FLUID_WINDOW // TIMESTEP_SIZE)
    MIN_FLUID_THRESHOLD = 2000
    MAP_THRESHOLD = 65
    LACTATE_THRESHOLD = 2

    print(f"  Rolling window: {WINDOW_STEPS} steps, fluid threshold: {MIN_FLUID_THRESHOLD}mL")

    stay_ids_arr = df['stay_id'].values
    fluid_arr = df['fluid_step'].values.astype(np.float64)
    map_arr = df['map'].values.astype(np.float64) if 'map' in df.columns else np.full(len(df), np.nan)
    lactic_arr = df['lactic_acid'].values.astype(np.float64) if 'lactic_acid' in df.columns else np.full(len(df), np.nan)

    # Group boundaries
    stay_change = np.concatenate([[True], stay_ids_arr[1:] != stay_ids_arr[:-1]])
    group_starts = np.where(stay_change)[0]
    group_ends = np.concatenate([group_starts[1:], [len(df)]])

    shock_values = df['septic_shock'].values

    for g_start, g_end in zip(group_starts, group_ends):
        group_len = g_end - g_start
        group_fluid = fluid_arr[g_start:g_end]

        # Rolling sum
        rolling_fluid = np.zeros(group_len)
        for k in range(group_len):
            start_k = max(0, k - WINDOW_STEPS + 1)
            rolling_fluid[k] = group_fluid[start_k:k + 1].sum()

        group_map = map_arr[g_start:g_end]
        group_lactic = lactic_arr[g_start:g_end]

        shock_cond = (
            (rolling_fluid >= MIN_FLUID_THRESHOLD) &
            (group_map < MAP_THRESHOLD) &
            (group_lactic > LACTATE_THRESHOLD)
        )

        if shock_cond.any():
            first_shock_pos = int(np.argmax(shock_cond))
            shock_values[g_start + first_shock_pos] = 1
            shock_values[g_start + first_shock_pos + 1:g_end] = 2

    df['septic_shock'] = shock_values

    total_patients = df['stay_id'].nunique()
    shock_patients = df.loc[df['septic_shock'] == 1, 'stay_id'].nunique()

    print("\nSeptic Shock Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing shock: {shock_patients} ({shock_patients / max(total_patients, 1) * 100:.1f}%)")
    print(f"Timesteps with shock onset: {(df['septic_shock'] == 1).sum()}")
    print(f"Censored timesteps: {(df['septic_shock'] == 2).sum()}")
    print("-" * 50)

    return df


# ============================================================
# MAIN
# ============================================================

import gc

def sample_and_hold(df, vitalslab_hold):
    """Memory-efficient sample and hold - processes columns sequentially."""
    print(f'Performing sample and hold interpolation ({"Numba" if NUMBA_AVAILABLE else "Python"})')

    df = df.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    cols_to_process = [col for col in vitalslab_hold if col in df.columns
                       and np.issubdtype(df[col].dtype, np.number)]

    if not cols_to_process:
        return df

    # Extract these once (shared across all columns)
    stay_ids = df['stay_id'].values
    charttimes = df['charttime'].values.astype(np.float64)

    # Find group boundaries once
    stay_change = np.concatenate([[True], stay_ids[1:] != stay_ids[:-1]])
    group_starts = np.where(stay_change)[0].astype(np.int64)
    group_ends = np.concatenate([group_starts[1:], [len(df)]]).astype(np.int64)

    print(f'  {len(cols_to_process)} columns, {len(group_starts)} groups, {len(df)} rows')

    for col_idx, col in enumerate(cols_to_process):
        if (col_idx + 1) % 10 == 0:
            print(f'  Column {col_idx + 1}/{len(cols_to_process)}: {col}')

        hold_period = float(vitalslab_hold[col] * 3600)
        col_values = df[col].to_numpy(dtype=np.float64, copy=True)
        col_values = _sample_hold_numba(col_values, charttimes, group_starts, group_ends, hold_period)
        df[col] = col_values

    return df


def handle_outliers(df):
    """Handle outliers - memory efficient version without Polars conversion."""
    print('Handling outliers in patient timeseries data')

    # Skip Polars conversion for large DataFrames to save memory
    # Use numpy directly instead

    outlier_rules = [
        ('weight_kg', 300, None),
        ('weight_lb', 660, None),
        ('heart_rate', 250, None),
        ('sbp_arterial', 300, None),
        ('map', 200, 0),
        ('dbp_arterial', 200, 0),
        ('respiratory_rate', 80, None),
        ('spo2', 150, None),
        ('oxygen_flow', 70, None),
        ('peep', 40, 0),
        ('tidal_volume', 1800, None),
        ('minute_volume', 50, None),
        ('potassium', 15, 1),
        ('sodium', 178, 95),
        ('chloride', 150, 70),
        ('glucose', 1000, 1),
        ('creatinine', 150, None),
        ('magnesium', 10, None),
        ('calcium_total', 20, None),
        ('calcium_ionized', 5, None),
        ('total_co2', 120, None),
        ('ast', 10000, None),
        ('alt', 10000, None),
        ('hemoglobin', 20, None),
        ('hematocrit', 65, None),
        ('wbc', 500, None),
        ('platelets', 2000, None),
        ('inr', 20, None),
        ('ph_arterial', 8, 6.7),
        ('arterial_o2_pressure', 700, None),
        ('arterial_co2_pressure', 200, None),
        ('arterial_base_excess', None, -50),
        ('lactic_acid', 30, None),
        ('bilirubin_total', 30, None),
    ]

    for col, upper, lower in outlier_rules:
        if col not in df.columns:
            continue
        arr = df[col].values  # No copy, works on underlying array
        if upper is not None:
            arr[arr > upper] = np.nan
        if lower is not None:
            arr[arr < lower] = np.nan

    # SpO2 cap at 100
    if 'spo2' in df.columns:
        arr = df['spo2'].values
        arr[arr > 100] = 100

    # Temperature
    if 'temp_C' in df.columns and 'temp_F' in df.columns:
        tc = df['temp_C'].values
        tf = df['temp_F'].values
        mask = (tc > 90) & np.isnan(tf)
        tf[mask] = tc[mask]
        tc[tc > 90] = np.nan

    # FiO2: (1) >100→NaN, (2) <1→*100, (3) <20→NaN
    if 'fio2' in df.columns:
        arr = df['fio2'].values
        arr[arr > 100] = np.nan
        mask_frac = (~np.isnan(arr)) & (arr < 1)
        arr[mask_frac] = arr[mask_frac] * 100
        mask_low = (~np.isnan(arr)) & (arr < 20)
        arr[mask_low] = np.nan

    return df


def estimate_gcs_from_rass(df):
    """Memory-efficient GCS estimation - no Polars conversion."""
    if 'gcs' not in df.columns:
        df['gcs'] = np.nan
    if 'richmond_ras' not in df.columns:
        return df

    rass_to_gcs = {4: 15, 3: 15, 2: 15, 1: 15, 0: 15,
                   -1: 14, -2: 12, -3: 11, -4: 6, -5: 3}

    gcs = df['gcs'].values
    rass = df['richmond_ras'].values

    gcs_missing = np.isnan(gcs)

    for rass_val, gcs_val in rass_to_gcs.items():
        mask = gcs_missing & (rass == rass_val)
        gcs[mask] = gcs_val

    df['gcs'] = gcs
    return df


def estimate_fio2(df):
    """Memory-efficient FiO2 estimation - no Polars conversion."""
    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    existing_flow_cols = [c for c in flow_columns if c in df.columns]

    if existing_flow_cols:
        combined_flow = df[existing_flow_cols].bfill(axis=1).iloc[:, 0].values
    else:
        combined_flow = np.full(len(df), np.nan)

    if 'fio2' not in df.columns:
        df['fio2'] = np.nan

    if 'oxygen_flow_device' not in df.columns:
        return df

    fio2 = df['fio2'].values.astype(np.float64)
    device = df['oxygen_flow_device'].astype(str).values
    flow = combined_flow

    fio2_missing = np.isnan(fio2)
    has_flow = ~np.isnan(flow)

    # Case 1: nasal cannula with flow
    mask1 = fio2_missing & has_flow & np.isin(device, ['0', '2'])
    if mask1.any():
        f = flow[mask1]
        vals = np.full(mask1.sum(), 70.0)
        thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        values = [24, 28, 32, 36, 40, 44, 50, 55, 62, 70]
        for t, v in zip(thresholds, values):
            vals[f <= t] = v
        fio2[mask1] = vals

    # Case 2: no flow, nasal → room air
    mask2 = fio2_missing & (~has_flow) & np.isin(device, ['0', '2'])
    fio2[mask2] = 21

    fio2_missing = np.isnan(fio2)

    # Case 3: face mask
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12']
    mask3 = fio2_missing & has_flow & np.isin(device, face_mask_types)
    if mask3.any():
        f = flow[mask3]
        vals = np.full(mask3.sum(), 75.0)
        thresholds = [4, 6, 8, 10, 12, 15]
        values = [36, 40, 58, 66, 69, 75]
        for t, v in zip(thresholds, values):
            vals[f <= t] = v
        fio2[mask3] = vals

    fio2_missing = np.isnan(fio2)

    # Case 4: non-rebreather
    mask4 = fio2_missing & has_flow & (device == '7')
    if mask4.any():
        f = flow[mask4]
        fio2[mask4] = np.where(f >= 15, 100,
                     np.where(f >= 10, 90,
                     np.where(f > 8, 80,
                     np.where(f > 6, 70, 60))))

    fio2_missing = np.isnan(fio2)

    # Case 5: CPAP/BiPAP
    mask5 = fio2_missing & has_flow & (device == '13')
    if mask5.any():
        f = flow[mask5]
        fio2[mask5] = np.where(f >= 15, 100, np.where(f >= 10, 80, 60))

    fio2_missing = np.isnan(fio2)

    # Case 6: Oxymizer
    mask6 = fio2_missing & has_flow & (device == '14')
    if mask6.any():
        f = flow[mask6]
        fio2[mask6] = np.where(f >= 10, 80, np.where(f >= 5, 60, 40))

    df['fio2'] = fio2
    return df


def handle_unit_conversions(df):
    """Memory-efficient unit conversions - no Polars conversion."""
    if 'temp_F' in df.columns and 'temp_C' in df.columns:
        tc = df['temp_C'].values.astype(np.float64)
        tf = df['temp_F'].values.astype(np.float64)

        # tempF 25-45 looks like Celsius
        mask = (tf > 25) & (tf < 45)
        tc[mask] = tf[mask]
        tf[mask] = np.nan

        # tempC > 70 looks like Fahrenheit
        mask = tc > 70
        tf[mask] = tc[mask]
        tc[mask] = np.nan

        # C → F
        mask = (~np.isnan(tc)) & np.isnan(tf)
        tf[mask] = tc[mask] * 1.8 + 32

        # F → C
        mask = (~np.isnan(tf)) & np.isnan(tc)
        tc[mask] = (tf[mask] - 32) / 1.8

        df['temp_C'] = tc
        df['temp_F'] = tf

    if 'hemoglobin' in df.columns and 'hematocrit' in df.columns:
        hgb = df['hemoglobin'].values.astype(np.float64)
        hct = df['hematocrit'].values.astype(np.float64)

        mask = (~np.isnan(hgb)) & np.isnan(hct)
        hct[mask] = hgb[mask] * 2.862 + 1.216

        mask = (~np.isnan(hct)) & np.isnan(hgb)
        hgb[mask] = (hct[mask] - 1.216) / 2.862

        df['hemoglobin'] = hgb
        df['hematocrit'] = hct

    if 'bilirubin_total' in df.columns and 'bilirubin_direct' in df.columns:
        bt = df['bilirubin_total'].values.astype(np.float64)
        bd = df['bilirubin_direct'].values.astype(np.float64)

        mask = (~np.isnan(bt)) & np.isnan(bd)
        bd[mask] = bt[mask] * 0.6934 - 0.1752

        mask = (~np.isnan(bd)) & np.isnan(bt)
        bt[mask] = (bd[mask] + 0.1752) / 0.6934

        df['bilirubin_total'] = bt
        df['bilirubin_direct'] = bd

    return df


def apply_exclusion_criteria(df, noise_ratio=0.10):
    """Multi-score gate + controlled noise injection (second copy — Polars pipeline path).

    Patients are retained if they reach a clinical threshold on AT LEAST ONE of:
      - SOFA >= 2  (organ dysfunction / Sepsis-3)
      - SIRS >= 2  (systemic inflammation)
      - NEWS2 >= 5 (medium clinical risk)

    A controlled fraction (noise_ratio) of non-scoring patients is then
    re-injected as negative examples to improve model robustness.
    """
    print('Applying exclusion criteria')

    initial_patients = df['stay_id'].nunique()
    excluded_counts = {}

    # Extreme UO
    extreme_uo_stays = df.loc[df['uo_step'] > 12000, 'stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_uo_stays)]
    excluded_counts['extreme_uo'] = len(extreme_uo_stays)

    # Extreme fluid
    extreme_fluid_stays = df.loc[df['fluid_step'] > 10000, 'stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_fluid_stays)]
    excluded_counts['extreme_fluid'] = len(extreme_fluid_stays)

    # Early deaths
    patient_time_range = df.groupby('stay_id').agg(
        min_time=('timestamp', 'min'),
        max_time=('timestamp', 'max'),
        morta_hosp=('morta_hosp', 'first')
    )
    patient_time_range['duration_hours'] = (patient_time_range['max_time'] - patient_time_range['min_time']) / 3600
    early_death_stays = patient_time_range[
        (patient_time_range['morta_hosp'] == 1) & (patient_time_range['duration_hours'] <= 24)
    ].index.values
    df = df[~df['stay_id'].isin(early_death_stays)]
    excluded_counts['early_death'] = len(early_death_stays)

    # Multi-score gate: keep patients that score on at least one system
    score_cols = {}
    if 'sofa_score' in df.columns:
        score_cols['max_sofa'] = ('sofa_score', 'max')
    if 'sirs_score' in df.columns:
        score_cols['max_sirs'] = ('sirs_score', 'max')
    if 'news2_score' in df.columns:
        score_cols['max_news2'] = ('news2_score', 'max')

    if score_cols:
        max_scores = df.groupby('stay_id').agg(**score_cols)

        eligible_mask = pd.Series(False, index=max_scores.index)
        if 'max_sofa' in max_scores.columns:
            eligible_mask |= (max_scores['max_sofa'] >= 2)
        if 'max_sirs' in max_scores.columns:
            eligible_mask |= (max_scores['max_sirs'] >= 2)
        if 'max_news2' in max_scores.columns:
            eligible_mask |= (max_scores['max_news2'] >= 5)

        eligible_stays   = max_scores[eligible_mask].index.values
        ineligible_stays = max_scores[~eligible_mask].index.values

        noise_n = max(1, int(len(eligible_stays) * noise_ratio))
        rng = np.random.default_rng(42)
        noise_stays = rng.choice(
            ineligible_stays,
            size=min(noise_n, len(ineligible_stays)),
            replace=False
        ) if len(ineligible_stays) > 0 else np.array([], dtype=ineligible_stays.dtype)

        keep_stays = np.concatenate([eligible_stays, noise_stays])
        df = df[df['stay_id'].isin(keep_stays)]

        excluded_counts['no_score_threshold'] = len(ineligible_stays) - len(noise_stays)
        excluded_counts['noise_injected']      = len(noise_stays)
    else:
        print("WARNING: No score columns found — skipping multi-score gate")

    final_patients = df['stay_id'].nunique()
    print("\nExclusion Statistics:")
    print("-" * 50)
    print(f"Initial patient count: {initial_patients}")
    for reason, count in excluded_counts.items():
        print(f"  {reason}: {count}")
    print(f"Final patient count: {final_patients}")
    print(f"Total excluded (net): {initial_patients - final_patients}")
    print("-" * 50)

    # Free memory
    gc.collect()

    return df


def add_sepsis_flag(df):
    """Memory-efficient sepsis flag."""
    print('Adding sepsis flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['sepsis'] = 0

    # Find first SOFA >= 2 per patient
    sepsis_mask = df['sofa_score'] >= 2
    sepsis_first = df[sepsis_mask].groupby('stay_id').head(1)

    df.loc[sepsis_first.index, 'sepsis'] = 1

    # Mark censored using numpy for speed
    stay_ids_arr = df['stay_id'].values
    idx_arr = np.arange(len(df))
    onset_dict = dict(zip(sepsis_first['stay_id'].values, sepsis_first.index.values))

    for stay_id, onset_idx in onset_dict.items():
        mask = (stay_ids_arr == stay_id) & (idx_arr > onset_idx)
        df.loc[mask, 'sepsis'] = 2

    total_patients = df['stay_id'].nunique()
    sepsis_patients = len(onset_dict)

    print("\nSepsis Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing sepsis: {sepsis_patients} ({sepsis_patients / max(total_patients, 1) * 100:.1f}%)")
    print(f"Timesteps with sepsis onset: {(df['sepsis'] == 1).sum()}")
    print(f"Censored timesteps: {(df['sepsis'] == 2).sum()}")
    print("-" * 50)

    return df


def add_septic_shock_flag(df):
    """Memory-efficient septic shock flag."""
    print('Adding septic shock flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['septic_shock'] = 0

    TIMESTEP_SIZE = 4
    FLUID_WINDOW = 12
    WINDOW_STEPS = max(1, FLUID_WINDOW // TIMESTEP_SIZE)
    MIN_FLUID_THRESHOLD = 2000
    MAP_THRESHOLD = 65
    LACTATE_THRESHOLD = 2

    print(f"  Rolling window: {WINDOW_STEPS} steps, fluid threshold: {MIN_FLUID_THRESHOLD}mL")

    stay_ids_arr = df['stay_id'].values
    fluid_arr = df['fluid_step'].values.astype(np.float64)
    map_arr = df['map'].values.astype(np.float64) if 'map' in df.columns else np.full(len(df), np.nan)
    lactic_arr = df['lactic_acid'].values.astype(np.float64) if 'lactic_acid' in df.columns else np.full(len(df), np.nan)

    # Group boundaries
    stay_change = np.concatenate([[True], stay_ids_arr[1:] != stay_ids_arr[:-1]])
    group_starts = np.where(stay_change)[0]
    group_ends = np.concatenate([group_starts[1:], [len(df)]])

    shock_values = df['septic_shock'].values

    for g_start, g_end in zip(group_starts, group_ends):
        group_len = g_end - g_start
        group_fluid = fluid_arr[g_start:g_end]

        # Rolling sum
        rolling_fluid = np.zeros(group_len)
        for k in range(group_len):
            start_k = max(0, k - WINDOW_STEPS + 1)
            rolling_fluid[k] = group_fluid[start_k:k + 1].sum()

        group_map = map_arr[g_start:g_end]
        group_lactic = lactic_arr[g_start:g_end]

        shock_cond = (
            (rolling_fluid >= MIN_FLUID_THRESHOLD) &
            (group_map < MAP_THRESHOLD) &
            (group_lactic > LACTATE_THRESHOLD)
        )

        if shock_cond.any():
            first_shock_pos = int(np.argmax(shock_cond))
            shock_values[g_start + first_shock_pos] = 1
            shock_values[g_start + first_shock_pos + 1:g_end] = 2

    df['septic_shock'] = shock_values

    total_patients = df['stay_id'].nunique()
    shock_patients = df.loc[df['septic_shock'] == 1, 'stay_id'].nunique()

    print("\nSeptic Shock Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing shock: {shock_patients} ({shock_patients / max(total_patients, 1) * 100:.1f}%)")
    print(f"Timesteps with shock onset: {(df['septic_shock'] == 1).sum()}")
    print(f"Censored timesteps: {(df['septic_shock'] == 2).sum()}")
    print("-" * 50)

    return df


def main():
    global backend
    args = parse_args()

    backend = ComputeBackend(use_gpu=args.gpu, device_id=args.gpu_device)

    # Load data as pandas
    data = load_processed_files()
    measurements, code_to_concept, hold_times = load_measurement_mappings()

    onset = data['onset']

    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, random_state=42)

    onset = onset.sort_values('stay_id').reset_index(drop=True)

    # Load non-onset (control) patients
    non_onset = data['non_onset']
    if args.non_onset_cap is not None:
        non_onset = non_onset.sample(
            n=min(args.non_onset_cap, len(non_onset)), random_state=42
        )
    non_onset = non_onset.sort_values('stay_id').reset_index(drop=True)
    print(f'  onset patients        : {len(onset)}')
    print(f'  non-onset patients    : {len(non_onset)}')

    # Pre-group
    print('Pre-indexing data by stay_id...')
    ce_groups = dict(list(data['ce'].groupby('stay_id')))
    print(f'  ce: {len(ce_groups)} groups')
    labU_groups = dict(list(data['labU'].groupby('stay_id')))
    print(f'  labU: {len(labU_groups)} groups')
    MV_groups = dict(list(data['MV'].groupby('stay_id')))
    print(f'  MV: {len(MV_groups)} groups')

    # Free original large DataFrames
    del data['ce'], data['labU'], data['MV']
    gc.collect()

    # Process each patient
    print('Processing patient timeseries data')
    def _process_cohort_polars(cohort_df, label):
        """Process a cohort (onset or non-onset) through the measurement pipeline."""
        patient_data = []
        count = 0
        total = len(cohort_df)
        print(f'Processing {label} timeseries data ({total} patients)')
        for _, row in cohort_df.iterrows():
            count += 1
            if count % 500 == 0:
                print(f'  Processed {count}/{total} {label} patients')
            icustayid = row['stay_id']
            onset_time = row['onset_time']
            if onset_time > 0:
                patient_df = process_patient_measurements_fast(
                    ce_groups.get(icustayid),
                    labU_groups.get(icustayid),
                    MV_groups.get(icustayid),
                    code_to_concept,
                    icustayid, onset_time,
                    winb4=args.window_before,
                    winaft=args.window_after
                )
                if patient_df is not None:
                    patient_data.append(patient_df)
        return patient_data

    # Process onset (infected) patients
    onset_data = _process_cohort_polars(onset, 'onset')

    # Process non-onset (control) patients through identical pipeline
    non_onset_data = _process_cohort_polars(non_onset, 'non-onset')

    # Free groups
    del ce_groups, labU_groups, MV_groups
    gc.collect()

    init_traj = pd.concat(onset_data + non_onset_data, ignore_index=True)
    del onset_data, non_onset_data
    gc.collect()
    print(f'  Total rows: {len(init_traj)}, patients: {init_traj["stay_id"].nunique()}')

    # Pipeline - ALL memory-efficient (no Polars conversion for large DFs)
    init_traj = handle_outliers(init_traj)
    gc.collect()

    init_traj = estimate_gcs_from_rass(init_traj)
    init_traj = estimate_fio2(init_traj)
    init_traj = handle_unit_conversions(init_traj)

    init_traj = sample_and_hold(init_traj, hold_times)
    gc.collect()

    init_traj = standardize_patient_trajectories(
        init_traj, data,
        timestep=args.timestep,
        window_before=args.window_before,
        window_after=args.window_after
    )
    gc.collect()

    init_traj = handle_missing_values(init_traj, args.missing_threshold)
    gc.collect()

    if 'fio2' in init_traj.columns:
        print(f"FiO2 zeros after handling missing values: {(init_traj['fio2'] == 0).sum()}")

    init_traj = calculate_derived_variables(init_traj)
    init_traj = apply_exclusion_criteria(init_traj, noise_ratio=args.noise_ratio)
    init_traj = add_septic_shock_flag(init_traj)
    init_traj = add_sepsis_flag(init_traj)

    # Print missingness
    missing_pct = (init_traj.isna().sum() / len(init_traj)) * 100
    print("\nMissing value percentages:")
    for col, pct in missing_pct.sort_values(ascending=False).items():
        if pct > 0:
            print(f"{col}: {pct:.1f}%")

    # Save standard output — DETERMINISTIC: sorted output
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"{args.output_dir}/patient_timeseries_{current_time}.csv"
    init_traj = init_traj.sort_values(by=['stay_id', 'timestamp']).reset_index(drop=True)
    init_traj.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

    # Optionally write a balanced version
    if args.balance:
        from data_processor import balance_dataframe
        balanced = balance_dataframe(init_traj, score_cols=['sofa_score', 'sirs_score', 'news2_score'])
        balanced_path = f"{args.output_dir}/patient_timeseries_{current_time}_balanced.csv"
        balanced.to_csv(balanced_path, index=False)
        print(f"Saved balanced data to {balanced_path}")



if __name__ == "__main__":
    main()
