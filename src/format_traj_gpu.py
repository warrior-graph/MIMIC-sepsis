import argparse
import json
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer
import math
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Try to import GPU libraries
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available (CuPy)")
except ImportError:
    print("GPU not available, using CPU")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true',
                        help="If specified, additionally save trajectories without normalized features")
    parser.add_argument("--output_dir", type=str, default="processed_files",
                        help="Directory to save processed files")
    parser.add_argument("--missing_threshold", type=float, default=0.8,
                        help="Threshold for dropping columns with missing values")
    parser.add_argument("--low_missing_threshold", type=float, default=0.05,
                        help="Threshold for using linear interpolation instead of KNN")
    parser.add_argument("--knn_neighbors", type=int, default=1,
                        help="Number of neighbors to use for KNN imputation")
    parser.add_argument("--knn_chunk_size", type=int, default=9999,
                        help="Chunk size for KNN imputation processing")
    parser.add_argument("--fluid_window", type=int, default=12,
                        help="Window in hours for fluid calculation in septic shock detection")
    parser.add_argument("--min_fluid_threshold", type=float, default=2000,
                        help="Minimum fluid threshold in mL for septic shock detection")
    parser.add_argument("--map_threshold", type=float, default=65,
                        help="MAP threshold for septic shock detection")
    parser.add_argument("--lactate_threshold", type=float, default=2,
                        help="Lactate threshold for septic shock detection")
    parser.add_argument("--timestep", type=int, default=4,
                        help="Size of timestep in hours")
    parser.add_argument("--window_before", type=int, default=24,
                        help="Hours to include before onset time")
    parser.add_argument("--window_after", type=int, default=72,
                        help="Hours to include after onset time")
    parser.add_argument("--notes_dir", type=str, default="processed_files",
                        help="Directory containing processed notes files")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of subjects to sample for testing")
    parser.add_argument("--gpu", action='store_true', default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--gpu_device", type=int, default=0,
                        help="GPU device ID to use (default: 0)")
    parser.add_argument("--balance", action='store_true', default=False,
                        help="If specified, also write a class-balanced version of the output CSV")
    parser.add_argument("--noise_ratio", type=float, default=0.10,
                        help="Fraction of non-scoring patients to inject as noise relative to primary cohort (default: 0.10)")
    return parser.parse_args()


class ComputeBackend:
    """Abstraction layer for CPU/GPU computation - guaranteed identical results"""

    def __init__(self, use_gpu=False, device_id=0):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device_id = device_id

        if self.use_gpu:
            cp.cuda.Device(device_id).use()
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            print(f"Using GPU device {device_id}: {props['name'].decode()}")
            meminfo = cp.cuda.runtime.memGetInfo()
            print(f"  Free memory: {meminfo[0] / 1e9:.2f} GB / {meminfo[1] / 1e9:.2f} GB total")
        else:
            print("Using CPU backend")

    @property
    def xp(self):
        """Return the array module (cupy or numpy)"""
        return cp if self.use_gpu else np

    def to_gpu(self, arr):
        """Move numpy array to GPU"""
        if self.use_gpu:
            return cp.asarray(arr)
        return arr

    def to_cpu(self, arr):
        """Move GPU array to CPU"""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)


# Global backend
backend = None


def load_processed_files():
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
        'onset': 'onset.csv'
    }

    data = {}
    for key, filename in files.items():
        data[key] = pd.read_csv(f'processed_files/{filename}', sep='|')

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


def process_patient_measurements_fast(data_indexed, code_to_concept, icustayid, onset_time, winb4=24, winaft=72):
    """Fast version using pre-grouped data - identical results to original"""
    t_start = onset_time - winb4 * 3600
    t_end = onset_time + winaft * 3600

    temp = data_indexed['ce_groups'].get(icustayid)
    temp2 = data_indexed['labU_groups'].get(icustayid)
    temp3 = data_indexed['MV_groups'].get(icustayid)

    frames = []

    if temp is not None:
        temp = temp[(temp['charttime'] >= t_start) & (temp['charttime'] < t_end)]
        if len(temp) > 0:
            temp_mapped = temp[['charttime', 'itemid', 'valuenum']].copy()
            temp_mapped['itemid'] = temp_mapped['itemid'].astype(int).astype(str)
            temp_mapped['concept'] = temp_mapped['itemid'].map(code_to_concept)
            temp_mapped = temp_mapped.dropna(subset=['concept'])
            if len(temp_mapped) > 0:
                pivot_ce = temp_mapped.pivot_table(
                    index='charttime', columns='concept', values='valuenum', aggfunc='last'
                )
                frames.append(pivot_ce)

    if temp2 is not None:
        temp2 = temp2[(temp2['charttime'] >= t_start) & (temp2['charttime'] < t_end)]
        if len(temp2) > 0:
            temp2_mapped = temp2[['charttime', 'itemid', 'valuenum']].copy()
            temp2_mapped['itemid'] = temp2_mapped['itemid'].astype(int).astype(str)
            temp2_mapped['concept'] = temp2_mapped['itemid'].map(code_to_concept)
            temp2_mapped = temp2_mapped.dropna(subset=['concept'])
            if len(temp2_mapped) > 0:
                pivot_lab = temp2_mapped.pivot_table(
                    index='charttime', columns='concept', values='valuenum', aggfunc='last'
                )
                frames.append(pivot_lab)

    if temp3 is not None:
        temp3 = temp3[(temp3['charttime'] >= t_start) & (temp3['charttime'] < t_end)]
        if len(temp3) > 0:
            mv_data = temp3[['charttime', 'mechvent']].drop_duplicates(subset=['charttime'], keep='last')
            mv_data = mv_data.set_index('charttime')
            frames.append(mv_data)

    if not frames:
        return None

    patient_df = pd.concat(frames, axis=1)
    if patient_df.columns.duplicated().any():
        patient_df = patient_df.loc[:, ~patient_df.columns.duplicated(keep='last')]

    patient_df.index.name = 'charttime'
    patient_df = patient_df.reset_index()
    patient_df['stay_id'] = icustayid

    return patient_df


def handle_outliers(df):
    """Handle outliers - uses GPU for masking if available, identical results guaranteed"""
    global backend
    print(f'Handling outliers ({"GPU" if backend.use_gpu else "CPU"})')

    xp = backend.xp

    outlier_rules_nan = [
        ('weight_kg', 300, None),
        ('weight_lb', 660, None),
        ('heart_rate', 250, None),
        ('sbp_arterial', 300, None),
        ('map', 200, 0),  # (max, min) - values outside [min, max] -> NaN
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
        ('arterial_base_excess', None, -50),  # only lower bound
        ('lactic_acid', 30, None),
        ('bilirubin_total', 30, None),
    ]

    for col, upper, lower in outlier_rules_nan:
        if col not in df.columns:
            continue

        arr = df[col].values.astype(np.float64)

        if backend.use_gpu:
            arr_gpu = backend.to_gpu(arr)
            if upper is not None:
                arr_gpu[arr_gpu > upper] = xp.nan
            if lower is not None:
                arr_gpu[arr_gpu < lower] = xp.nan
            df[col] = backend.to_cpu(arr_gpu)
        else:
            if upper is not None:
                arr[arr > upper] = np.nan
            if lower is not None:
                arr[arr < lower] = np.nan
            df[col] = arr

    # SpO2 cap at 100
    if 'spo2' in df.columns:
        arr = df['spo2'].values.astype(np.float64)
        if backend.use_gpu:
            arr_gpu = backend.to_gpu(arr)
            arr_gpu[arr_gpu > 100] = 100
            df['spo2'] = backend.to_cpu(arr_gpu)
        else:
            arr[arr > 100] = 100
            df['spo2'] = arr

    # Temperature: move misplaced values
    if 'temp_C' in df.columns and 'temp_F' in df.columns:
        tc = df['temp_C'].values.astype(np.float64)
        tf = df['temp_F'].values.astype(np.float64)

        if backend.use_gpu:
            tc_g = backend.to_gpu(tc)
            tf_g = backend.to_gpu(tf)
            mask = (tc_g > 90) & xp.isnan(tf_g)
            tf_g[mask] = tc_g[mask]
            tc_g[tc_g > 90] = xp.nan
            df['temp_C'] = backend.to_cpu(tc_g)
            df['temp_F'] = backend.to_cpu(tf_g)
        else:
            mask = (tc > 90) & np.isnan(tf)
            tf[mask] = tc[mask]
            tc[tc > 90] = np.nan
            df['temp_C'] = tc
            df['temp_F'] = tf

    # FiO2: exact same order as original
    # 1. > 100 → NaN
    # 2. < 1 → multiply by 100
    # 3. < 20 → NaN
    if 'fio2' in df.columns:
        arr = df['fio2'].values.astype(np.float64)
        if backend.use_gpu:
            arr_g = backend.to_gpu(arr)
            arr_g[arr_g > 100] = xp.nan
            mask_frac = (~xp.isnan(arr_g)) & (arr_g < 1)
            arr_g[mask_frac] = arr_g[mask_frac] * 100
            arr_g[(~xp.isnan(arr_g)) & (arr_g < 20)] = xp.nan
            df['fio2'] = backend.to_cpu(arr_g)
        else:
            arr[arr > 100] = np.nan
            mask_frac = (~np.isnan(arr)) & (arr < 1)
            arr[mask_frac] = arr[mask_frac] * 100
            arr[(~np.isnan(arr)) & (arr < 20)] = np.nan
            df['fio2'] = arr

    return df


def estimate_gcs_from_rass(df):
    """Vectorized GCS estimation from RASS - identical on CPU/GPU"""
    if 'gcs' not in df.columns:
        df['gcs'] = np.nan
    if 'richmond_ras' not in df.columns:
        return df

    rass_to_gcs = {4: 15, 3: 15, 2: 15, 1: 15, 0: 15,
                   -1: 14, -2: 12, -3: 11, -4: 6, -5: 3}

    gcs_missing = df['gcs'].isna()
    rass_values = df['richmond_ras']

    for rass_val, gcs_val in rass_to_gcs.items():
        mask = gcs_missing & (rass_values == rass_val)
        df.loc[mask, 'gcs'] = gcs_val

    return df


def estimate_fio2(df):
    """Estimate FiO2 - identical results CPU/GPU"""
    df = df.copy()

    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    existing_flow_cols = [c for c in flow_columns if c in df.columns]
    if existing_flow_cols:
        df['combined_o2_flow'] = df[existing_flow_cols].bfill(axis=1).iloc[:, 0]
    else:
        df['combined_o2_flow'] = np.nan

    if 'fio2' not in df.columns:
        df['fio2'] = np.nan

    if 'oxygen_flow_device' not in df.columns:
        df.drop('combined_o2_flow', axis=1, errors='ignore')
        return df

    fio2_missing = df['fio2'].isna()
    has_flow = df['combined_o2_flow'].notna()
    device = df['oxygen_flow_device'].astype(str)
    flow = df['combined_o2_flow'].values

    # Case 1: nasal cannula / none with flow
    mask1 = (fio2_missing & has_flow & device.isin(['0', '2'])).values
    if mask1.any():
        f = flow[mask1]
        fio2_vals = np.full(len(f), 70.0)
        thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        values = [24, 28, 32, 36, 40, 44, 50, 55, 62, 70]
        for t, v in zip(thresholds, values):
            fio2_vals[f <= t] = v
        df.loc[mask1, 'fio2'] = fio2_vals

    # Case 2: no flow, nasal cannula / none -> room air
    mask2 = (fio2_missing & (~has_flow) & device.isin(['0', '2'])).values
    df.loc[mask2, 'fio2'] = 21

    fio2_missing = df['fio2'].isna()

    # Case 3: face mask
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12']
    mask3 = (fio2_missing & has_flow & device.isin(face_mask_types)).values
    if mask3.any():
        f = flow[mask3]
        fio2_vals = np.full(len(f), 75.0)
        thresholds = [4, 6, 8, 10, 12, 15]
        values = [36, 40, 58, 66, 69, 75]
        for t, v in zip(thresholds, values):
            fio2_vals[f <= t] = v
        df.loc[mask3, 'fio2'] = fio2_vals

    fio2_missing = df['fio2'].isna()

    # Case 4: non-rebreather
    mask4 = (fio2_missing & has_flow & (device == '7')).values
    if mask4.any():
        f = flow[mask4]
        fio2_vals = np.where(f >= 15, 100,
                   np.where(f >= 10, 90,
                   np.where(f > 8, 80,
                   np.where(f > 6, 70, 60))))
        df.loc[mask4, 'fio2'] = fio2_vals

    fio2_missing = df['fio2'].isna()

    # Case 5: CPAP/BiPAP
    mask5 = (fio2_missing & has_flow & (device == '13')).values
    if mask5.any():
        f = flow[mask5]
        fio2_vals = np.where(f >= 15, 100, np.where(f >= 10, 80, 60))
        df.loc[mask5, 'fio2'] = fio2_vals

    fio2_missing = df['fio2'].isna()

    # Case 6: Oxymizer
    mask6 = (fio2_missing & has_flow & (device == '14')).values
    if mask6.any():
        f = flow[mask6]
        fio2_vals = np.where(f >= 10, 80, np.where(f >= 5, 60, 40))
        df.loc[mask6, 'fio2'] = fio2_vals

    df.drop('combined_o2_flow', axis=1, inplace=True)
    return df


def handle_unit_conversions(df):
    """Handle unit conversions - identical CPU/GPU"""
    if 'temp_F' in df.columns and 'temp_C' in df.columns:
        mask = (df['temp_F'] > 25) & (df['temp_F'] < 45)
        df.loc[mask, 'temp_C'] = df.loc[mask, 'temp_F']
        df.loc[mask, 'temp_F'] = np.nan

        mask = df['temp_C'] > 70
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[mask, 'temp_C'] = np.nan

        mask = df['temp_C'].notna() & df['temp_F'].isna()
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C'] * 1.8 + 32

        mask = df['temp_F'].notna() & df['temp_C'].isna()
        df.loc[mask, 'temp_C'] = (df.loc[mask, 'temp_F'] - 32) / 1.8

    if 'hemoglobin' in df.columns and 'hematocrit' in df.columns:
        mask = df['hemoglobin'].notna() & df['hematocrit'].isna()
        df.loc[mask, 'hematocrit'] = df.loc[mask, 'hemoglobin'] * 2.862 + 1.216

        mask = df['hematocrit'].notna() & df['hemoglobin'].isna()
        df.loc[mask, 'hemoglobin'] = (df.loc[mask, 'hematocrit'] - 1.216) / 2.862

    if 'bilirubin_total' in df.columns and 'bilirubin_direct' in df.columns:
        mask = df['bilirubin_total'].notna() & df['bilirubin_direct'].isna()
        df.loc[mask, 'bilirubin_direct'] = df.loc[mask, 'bilirubin_total'] * 0.6934 - 0.1752

        mask = df['bilirubin_direct'].notna() & df['bilirubin_total'].isna()
        df.loc[mask, 'bilirubin_total'] = (df.loc[mask, 'bilirubin_direct'] + 0.1752) / 0.6934

    return df


def sample_and_hold(df, vitalslab_hold):
    """Sample and hold - sequential, identical results CPU/GPU
    GPU is used to accelerate the NaN checking if available"""
    global backend
    print(f'Performing sample and hold interpolation ({"GPU-assisted" if backend.use_gpu else "CPU"})')

    df = df.copy()
    df = df.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    cols_to_process = [col for col in vitalslab_hold if col in df.columns
                       and np.issubdtype(df[col].dtype, np.number)]

    if not cols_to_process:
        return df

    stay_ids = df['stay_id'].values
    charttimes = df['charttime'].values.astype(np.float64)

    # Find group boundaries
    stay_change = np.concatenate([[True], stay_ids[1:] != stay_ids[:-1]])
    group_starts = np.where(stay_change)[0]
    group_ends = np.concatenate([group_starts[1:], [len(df)]])

    for col in cols_to_process:
        hold_period = vitalslab_hold[col] * 3600
        col_values = df[col].values.astype(np.float64).copy()

        for g_start, g_end in zip(group_starts, group_ends):
            last_value = np.nan
            last_time = 0.0

            for i in range(g_start, g_end):
                if not np.isnan(col_values[i]):
                    last_value = col_values[i]
                    last_time = charttimes[i]
                elif not np.isnan(last_value) and (charttimes[i] - last_time) <= hold_period:
                    col_values[i] = last_value

        df[col] = col_values

    return df


def combine_patient_data(patient_data, timestep=4, window_before=24, window_after=72):
    """Combines measurement sources into unified time series - identical results"""

    start_time = patient_data['start_time']
    stay_id = patient_data['stay_id']
    measurements = patient_data['measurements']
    fluid_data = patient_data['fluid']
    vaso_data = patient_data['vasopressors']
    uo_data = patient_data['urine_output']
    abx_data = patient_data['antibiotics']
    demographics = patient_data['demographics']

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

    # Precompute arrays
    if abx_data is not None and len(abx_data) > 0:
        abx_data_c = abx_data.copy()
        abx_data_c['stay_id'] = abx_data_c['stay_id'].astype('int64')
        first_abx_time = abx_data_c['starttime'].min()
        abx_starts = abx_data_c['starttime'].values
        abx_stops = abx_data_c['stoptime'].values
        abx_drugs = abx_data_c['drug'].values
    else:
        first_abx_time = None
        abx_starts = np.array([])
        abx_stops = np.array([])
        abx_drugs = np.array([])

    if fluid_data is not None and len(fluid_data) > 0:
        fluid_starts = fluid_data['starttime'].values
        fluid_ends = fluid_data['endtime'].values
        fluid_amounts = fluid_data['amount'].values
    else:
        fluid_starts = np.array([])
        fluid_ends = np.array([])
        fluid_amounts = np.array([])

    if vaso_data is not None and len(vaso_data) > 0:
        vaso_starts = vaso_data['starttime'].values
        vaso_ends = vaso_data['endtime'].values
        vaso_rates = vaso_data['rate_std'].values
    else:
        vaso_starts = np.array([])
        vaso_ends = np.array([])
        vaso_rates = np.array([])

    if uo_data is not None and len(uo_data) > 0:
        uo_times = uo_data['charttime'].values
        uo_values_arr = uo_data['value'].values
    else:
        uo_times = np.array([])
        uo_values_arr = np.array([])

    processed_rows = []

    for idx in range(num_timesteps):
        ws = window_starts[idx]
        we = window_ends[idx]

        # Measurements
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
            fluid_step = 0
            fluid_total = 0

        # Vasopressors
        if len(vaso_starts) > 0:
            vmask = (vaso_starts <= we) & (vaso_ends >= ws)
            if vmask.any():
                window_rates = vaso_rates[vmask]
                vaso_median = float(np.nanmedian(window_rates))
                vaso_max = float(np.nanmax(window_rates))
            else:
                vaso_median = 0
                vaso_max = 0
        else:
            vaso_median = 0
            vaso_max = 0

        # Urine output
        if len(uo_times) > 0:
            uo_mask = (uo_times >= ws) & (uo_times < we)
            uo_step = float(uo_values_arr[uo_mask].sum())
            uo_total_mask = uo_times < we
            uo_total = float(uo_values_arr[uo_total_mask].sum())
        else:
            uo_step = 0
            uo_total = 0

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


def standardize_patient_trajectories(init_traj, data_dict, timestep=4, window_before=24, window_after=72):
    print('Processing all patients with fixed time windows')
    all_patient_data = []

    onset_lookup = data_dict['onset'].set_index('stay_id')['onset_time'].to_dict()
    demog_lookup = data_dict['demog'].set_index('stay_id')

    fluid_groups = dict(list(data_dict['fluid'].groupby('stay_id')))
    vaso_groups = dict(list(data_dict['vaso'].groupby('stay_id')))
    uo_groups = dict(list(data_dict['UO'].groupby('stay_id')))
    abx_groups = dict(list(data_dict['abx'].groupby('stay_id')))

    stay_ids = init_traj['stay_id'].unique()
    traj_groups = dict(list(init_traj.groupby('stay_id')))

    print(f'Processing {len(stay_ids)} patients')
    count = 0
    for stay_id in stay_ids:
        count += 1
        if count % 500 == 0:
            print(f'  Processed {count}/{len(stay_ids)} patients')

        if stay_id not in onset_lookup:
            continue

        start_time = onset_lookup[stay_id]

        try:
            demog_row = demog_lookup.loc[stay_id]
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
            patient_data,
            timestep=timestep,
            window_before=window_before,
            window_after=window_after
        )
        if processed_patient is not None:
            all_patient_data.append(processed_patient)

    return pd.concat(all_patient_data, ignore_index=True)


def fixgaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolates gaps - always uses numpy for exact results"""
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


from sklearn.impute import KNNImputer

def handle_missing_values(df, missing_threshold=0.8):
    """Handle missing values through interpolation and KNN imputation"""
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
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns
    excluded_numeric_cols = [col for col in all_numeric_cols if col not in measurement_cols]

    miss_stats_meas = df[measurement_cols].isna().sum() / len(df)
    miss_stats_excl = df[excluded_numeric_cols].isna().sum() / len(df)

    miss_stats_meas = miss_stats_meas.sort_values(ascending=False)
    miss_stats_excl = miss_stats_excl.sort_values(ascending=False)

    print("Measurement columns to be imputed:")
    print(f"{'Variable':<30} {'Missing %':>10}")
    print("-" * 50)
    for var, miss_pct in miss_stats_meas.items():
        print(f"{var:<30} {miss_pct:>10.1%}")

    print("\nExcluded numeric columns:")
    print(f"{'Variable':<30} {'Missing %':>10}")
    print("-" * 50)
    for var, miss_pct in miss_stats_excl.items():
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
    cols_to_keep = [
        col for col in miss.index
        if miss[col] < missing_threshold or col in PROTECTED_SCORE_COLS
    ]
    non_meas_cols = [c for c in df.columns if c not in measurement_cols]
    df = df[non_meas_cols + cols_to_keep]

    protected_kept = [c for c in PROTECTED_SCORE_COLS if c in df.columns]
    print(f"Protected score columns retained: {len(protected_kept)} -> {protected_kept}")

    # Linear interpolation for low-missing columns
    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    low_missing_cols = [c for c in low_missing_cols if c in df.columns]
    for col in low_missing_cols:
        df[col] = fixgaps(df[col].values)

    # KNN imputation for remaining missing values (includes protected cols above threshold)
    cols_for_knn = [c for c in cols_to_keep if c not in low_missing_cols and c in df.columns]
    if cols_for_knn:
        ref = df[cols_for_knn].values.astype(np.float64)

        chunk_size = 9999
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        print(f'KNN imputation: {total_chunks} chunks')

        imputer = KNNImputer(n_neighbors=1, weights='uniform')

        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            chunk = ref[i:chunk_end, :]

            # Only run imputer if there are missing values in this chunk
            if np.isnan(chunk).any():
                ref[i:chunk_end, :] = imputer.fit_transform(chunk)

            chunk_idx = i // chunk_size + 1
            if chunk_idx % 10 == 0 or chunk_idx == total_chunks:
                print(f'  Chunk {chunk_idx}/{total_chunks}')

        df[cols_for_knn] = ref

    return df

def calculate_derived_variables(df):
    """Calculate derived variables - GPU accelerated with identical results guaranteed"""
    global backend
    print(f'Computing derived variables ({"GPU" if backend.use_gpu else "CPU"})')

    df = df.copy()

    # Fix demographics
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

    # Extract arrays and optionally move to GPU
    def get_col(col_name, default_nan=True):
        if col_name in df.columns:
            arr = df[col_name].values.astype(np.float64)
        else:
            arr = np.full(n, np.nan) if default_nan else np.zeros(n)
        return backend.to_gpu(arr) if backend.use_gpu else arr

    # P/F ratio
    if 'arterial_o2_pressure' in df.columns and 'fio2' in df.columns:
        ao2 = get_col('arterial_o2_pressure')
        fio2 = get_col('fio2')
        pf_ratio = ao2 / (fio2 / 100)
        df['pf_ratio'] = backend.to_cpu(pf_ratio)
    else:
        df['pf_ratio'] = np.nan
        pf_ratio = get_col('pf_ratio')

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

    # Get arrays for SOFA
    pf = get_col('pf_ratio')
    plt_vals = get_col('platelets')
    bili = get_col('bilirubin_total')
    map_vals = get_col('map')
    vaso_max_arr = get_col('vaso_max')
    gcs_vals = get_col('gcs')
    cr_vals = get_col('creatinine')
    uo_vals = get_col('uo_step')

    # SOFA Respiratory
    sofa_resp = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(pf)
    sofa_resp[valid & (pf < 100)] = 4
    sofa_resp[valid & (pf >= 100) & (pf < 200)] = 3
    sofa_resp[valid & (pf >= 200) & (pf < 300)] = 2
    sofa_resp[valid & (pf >= 300) & (pf < 400)] = 1

    # SOFA Coagulation
    sofa_coag = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(plt_vals)
    sofa_coag[valid & (plt_vals < 20)] = 4
    sofa_coag[valid & (plt_vals >= 20) & (plt_vals < 50)] = 3
    sofa_coag[valid & (plt_vals >= 50) & (plt_vals < 100)] = 2
    sofa_coag[valid & (plt_vals >= 100) & (plt_vals < 150)] = 1

    # SOFA Liver
    sofa_liver = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(bili)
    sofa_liver[valid & (bili >= 12)] = 4
    sofa_liver[valid & (bili >= 6) & (bili < 12)] = 3
    sofa_liver[valid & (bili >= 2) & (bili < 6)] = 2
    sofa_liver[valid & (bili >= 1.2) & (bili < 2)] = 1

    # SOFA Cardiovascular - exact same logic as original
    sofa_cv = xp.zeros(n, dtype=xp.int32)
    valid_map = ~xp.isnan(map_vals)
    map_na = xp.isnan(map_vals)
    vaso_na = xp.isnan(vaso_max_arr)
    sofa_cv[valid_map & (map_vals < 70) & (map_vals >= 65)] = 1
    sofa_cv[valid_map & (map_vals < 65)] = 2
    sofa_cv[map_na & ~vaso_na & (vaso_max_arr <= 0.1)] = 3
    sofa_cv[map_na & ~vaso_na & (vaso_max_arr > 0.1)] = 4

    # SOFA CNS
    sofa_cns = xp.zeros(n, dtype=xp.int32)
    valid = ~xp.isnan(gcs_vals)
    sofa_cns[valid & (gcs_vals <= 5)] = 4
    sofa_cns[valid & (gcs_vals > 5) & (gcs_vals <= 9)] = 3
    sofa_cns[valid & (gcs_vals > 9) & (gcs_vals <= 12)] = 2
    sofa_cns[valid & (gcs_vals > 12) & (gcs_vals <= 14)] = 1

    # SOFA Renal
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

    # Move results to CPU
    df['sofa_resp'] = backend.to_cpu(sofa_resp).astype(int)
    df['sofa_coag'] = backend.to_cpu(sofa_coag).astype(int)
    df['sofa_liver'] = backend.to_cpu(sofa_liver).astype(int)
    df['sofa_cv'] = backend.to_cpu(sofa_cv).astype(int)
    df['sofa_cns'] = backend.to_cpu(sofa_cns).astype(int)
    df['sofa_renal'] = backend.to_cpu(sofa_renal).astype(int)

    df['sofa_score'] = df['sofa_resp'] + df['sofa_coag'] + df['sofa_liver'] + \
                       df['sofa_cv'] + df['sofa_cns'] + df['sofa_renal']

    # SIRS - GPU accelerated
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

    # Respiratory rate
    if 'respiratory_rate' in df.columns:
        rr_n = df['respiratory_rate'].values
        valid = ~np.isnan(rr_n)
        news2[valid & (rr_n <= 8)] += 3
        news2[valid & (rr_n >= 9) & (rr_n <= 11)] += 1
        news2[valid & (rr_n >= 21) & (rr_n <= 24)] += 2
        news2[valid & (rr_n >= 25)] += 3

    # SpO2 (Scale 1)
    if 'spo2' in df.columns:
        sp = df['spo2'].values
        valid = ~np.isnan(sp)
        news2[valid & (sp <= 91)] += 3
        news2[valid & (sp >= 92) & (sp <= 93)] += 2
        news2[valid & (sp >= 94) & (sp <= 95)] += 1

    # Supplemental oxygen: fio2 > 21% → +2
    if 'fio2' in df.columns:
        fio2_n = df['fio2'].values
        valid = ~np.isnan(fio2_n)
        news2[valid & (fio2_n > 21)] += 2

    # Systolic BP
    if 'sbp_arterial' in df.columns:
        sbp = df['sbp_arterial'].values
        valid = ~np.isnan(sbp)
        news2[valid & (sbp <= 90)] += 3
        news2[valid & (sbp >= 91) & (sbp <= 100)] += 2
        news2[valid & (sbp >= 101) & (sbp <= 110)] += 1
        news2[valid & (sbp >= 220)] += 3

    # Heart rate
    if 'heart_rate' in df.columns:
        hr_n = df['heart_rate'].values
        valid = ~np.isnan(hr_n)
        news2[valid & (hr_n <= 40)] += 3
        news2[valid & (hr_n >= 41) & (hr_n <= 50)] += 1
        news2[valid & (hr_n >= 91) & (hr_n <= 110)] += 1
        news2[valid & (hr_n >= 111) & (hr_n <= 130)] += 2
        news2[valid & (hr_n >= 131)] += 3

    # Consciousness: GCS 15 = Alert (0), GCS < 15 = CVPU (+3)
    if 'gcs' in df.columns:
        gcs_n = df['gcs'].values
        valid = ~np.isnan(gcs_n)
        news2[valid & (gcs_n < 15)] += 3

    # Temperature (Celsius)
    if 'temp_C' in df.columns:
        tc_n = df['temp_C'].values
        valid = ~np.isnan(tc_n)
        news2[valid & (tc_n <= 35.0)] += 3
        news2[valid & (tc_n >= 35.1) & (tc_n <= 36.0)] += 1
        news2[valid & (tc_n >= 38.1) & (tc_n <= 39.0)] += 1
        news2[valid & (tc_n >= 39.1)] += 2

    df['news2_score'] = news2

    # Print distributions
    for comp in ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_cns', 'sofa_renal']:
        print(f"\n{comp} distribution:")
        print(df[comp].value_counts().sort_index())

    print(f"\nnews2_score distribution:")
    print(pd.Series(news2).value_counts().sort_index())

    return df


def apply_exclusion_criteria(df, noise_ratio=0.10):
    """Apply exclusion criteria with multi-score gate and controlled noise injection.

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

    extreme_uo_stays = df.loc[df['uo_step'] > 12000, 'stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_uo_stays)]
    excluded_counts['extreme_uo'] = len(extreme_uo_stays)

    extreme_fluid_stays = df.loc[df['fluid_step'] > 10000, 'stay_id'].unique()
    df = df[~df['stay_id'].isin(extreme_fluid_stays)]
    excluded_counts['extreme_fluid'] = len(extreme_fluid_stays)

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

        # Inject a controlled fraction of zero-score patients as noise
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

    return df


def add_sepsis_flag(df):
    """Add sepsis flag - identical results"""
    print('Adding sepsis flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['sepsis'] = 0

    sepsis_mask = df['sofa_score'] >= 2
    sepsis_first = df[sepsis_mask].groupby('stay_id').head(1)

    df.loc[sepsis_first.index, 'sepsis'] = 1

    # Mark censored
    for stay_id in sepsis_first['stay_id'].unique():
        onset_idx = sepsis_first.loc[sepsis_first['stay_id'] == stay_id].index[0]
        mask = (df['stay_id'] == stay_id) & (df.index > onset_idx)
        df.loc[mask, 'sepsis'] = 2

    total_patients = df['stay_id'].nunique()
    sepsis_patients = len(sepsis_first['stay_id'].unique()) if len(sepsis_first) > 0 else 0

    print("\nSepsis Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing sepsis: {sepsis_patients} ({sepsis_patients / max(total_patients, 1) * 100:.1f}%)")
    print(f"Timesteps with sepsis onset: {(df['sepsis'] == 1).sum()}")
    print(f"Censored timesteps: {(df['sepsis'] == 2).sum()}")
    print("-" * 50)

    return df


def add_septic_shock_flag(df):
    """Add septic shock flag - identical results"""
    print('Adding septic shock flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['septic_shock'] = 0

    TIMESTEP_SIZE = 4
    FLUID_WINDOW = 12
    WINDOW_STEPS = max(1, FLUID_WINDOW // TIMESTEP_SIZE)
    MIN_FLUID_THRESHOLD = 2000
    MAP_THRESHOLD = 65
    LACTATE_THRESHOLD = 2

    print(f"Using rolling window of {WINDOW_STEPS} timesteps ({WINDOW_STEPS * TIMESTEP_SIZE} hours)")
    print(f"Minimum fluid threshold: {MIN_FLUID_THRESHOLD}mL over {FLUID_WINDOW} hours")

    for stay_id, group in df.groupby('stay_id'):
        group_sorted = group.sort_values('timestamp')
        rolling_fluid = group_sorted['fluid_step'].rolling(window=WINDOW_STEPS, min_periods=1).sum()

        lactic = group_sorted['lactic_acid'].values if 'lactic_acid' in group_sorted.columns else np.full(len(group_sorted), np.nan)
        map_v = group_sorted['map'].values if 'map' in group_sorted.columns else np.full(len(group_sorted), np.nan)

        shock_cond = (
            (rolling_fluid.values >= MIN_FLUID_THRESHOLD) &
            (map_v < MAP_THRESHOLD) &
            (lactic > LACTATE_THRESHOLD)
        )

        if shock_cond.any():
            first_shock_pos = np.argmax(shock_cond)
            shock_idx = group_sorted.index[first_shock_pos]
            df.loc[shock_idx, 'septic_shock'] = 1

            subsequent = group_sorted.index[first_shock_pos + 1:]
            if len(subsequent) > 0:
                df.loc[subsequent, 'septic_shock'] = 2

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

    # Initialize compute backend
    backend = ComputeBackend(use_gpu=args.gpu, device_id=args.gpu_device)

    # Load data
    data = load_processed_files()
    measurements, code_to_concept, hold_times = load_measurement_mappings()

    onset = data['onset']

    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, random_state=42)

    # Pre-index data by stay_id
    print('Pre-indexing data by stay_id...')
    data_indexed = {
        'ce_groups': dict(list(data['ce'].groupby('stay_id'))),
        'labU_groups': dict(list(data['labU'].groupby('stay_id'))),
        'MV_groups': dict(list(data['MV'].groupby('stay_id'))),
    }

    # Process each patient
    print('Processing patient timeseries data')
    all_patient_data = []

    count = 0
    total = len(onset)
    for _, row in onset.iterrows():
        count += 1
        if count % 500 == 0:
            print(f'  Processed {count}/{total} patients')

        icustayid = row['stay_id']
        onset_time = row['onset_time']
        if onset_time > 0:
            patient_df = process_patient_measurements_fast(
                data_indexed, code_to_concept,
                icustayid, onset_time,
                winb4=args.window_before,
                winaft=args.window_after
            )
            if patient_df is not None:
                all_patient_data.append(patient_df)

    init_traj = pd.concat(all_patient_data, ignore_index=True)

    # Pipeline
    init_traj = handle_outliers(init_traj)
    init_traj = estimate_gcs_from_rass(init_traj)
    init_traj = estimate_fio2(init_traj)
    init_traj = handle_unit_conversions(init_traj)
    init_traj = sample_and_hold(init_traj, hold_times)

    init_traj = standardize_patient_trajectories(
        init_traj, data,
        timestep=args.timestep,
        window_before=args.window_before,
        window_after=args.window_after
    )

    init_traj = handle_missing_values(init_traj, args.missing_threshold)

    print(f"FiO2 zeros after handling missing values: {(init_traj['fio2'] == 0).sum()}" if 'fio2' in init_traj.columns else "")

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

    # Save standard output — filename includes timestamp for versioning
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    init_traj = init_traj.sort_values(by=['stay_id', 'timestamp']).reset_index(drop=True)
    output_path = f"{args.output_dir}/patient_timeseries_{current_time}.csv"
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
