import argparse
import json
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from fancyimpute import KNN
import math
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true',
                        help="If specified, additionally save trajectories without normalized features")
    parser.add_argument("--output_dir", type=str, default="processed_files",
                        help="Directory to save processed files")
    parser.add_argument("--missing_threshold", type=float, default=0.8,
                        help="Threshold for dropping columns with missing values (default: 0.7)")
    parser.add_argument("--low_missing_threshold", type=float, default=0.05,
                        help="Threshold for using linear interpolation instead of KNN (default: 0.05)")
    parser.add_argument("--knn_neighbors", type=int, default=1,
                        help="Number of neighbors to use for KNN imputation (default: 1)")
    parser.add_argument("--knn_chunk_size", type=int, default=9999,
                        help="Chunk size for KNN imputation processing (default: 9999)")
    parser.add_argument("--fluid_window", type=int, default=12,
                        help="Window in hours for fluid calculation in septic shock detection (default: 12)")
    parser.add_argument("--min_fluid_threshold", type=float, default=2000,
                        help="Minimum fluid threshold in mL for septic shock detection (default: 2000)")
    parser.add_argument("--map_threshold", type=float, default=65,
                        help="MAP threshold for septic shock detection (default: 65)")
    parser.add_argument("--lactate_threshold", type=float, default=2,
                        help="Lactate threshold for septic shock detection (default: 2)")
    parser.add_argument("--timestep", type=int, default=4,
                        help="Size of timestep in hours (default: 4)")
    parser.add_argument("--window_before", type=int, default=24,
                        help="Hours to include before onset time (default: 24)")
    parser.add_argument("--window_after", type=int, default=72,
                        help="Hours to include after onset time (default: 72)")
    parser.add_argument("--notes_dir", type=str, default="processed_files",
                        help="Directory containing processed notes files")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of subjects to sample for testing (default: None, use all subjects)")
    return parser.parse_args()


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
    """Load the measurement mappings from JSON file"""
    print('Loading measurement mappings')
    with open("src/ReferenceFiles/measurement_mappings.json", "r") as f:
        measurements = json.load(f)

    # Create reverse mapping (code to concept)
    code_to_concept = {}
    for concept, info in measurements.items():
        for code in info['codes']:
            code_to_concept[code] = concept

    # Create hold times mapping
    hold_times = {}
    for concept, info in measurements.items():
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']

    return measurements, code_to_concept, hold_times


def process_patient_measurements(data, measurements, code_to_concept, icustayid, onset_time, winb4=24, winaft=72):
    """Process data for a single patient - vectorized version"""
    # Get relevant data for this patient
    temp = data['ce'][data['ce']['stay_id'] == icustayid]
    temp2 = data['labU'][data['labU']['stay_id'] == icustayid]
    temp3 = data['MV'][data['MV']['stay_id'] == icustayid]

    # Filter for time window
    t_start = onset_time - winb4 * 3600
    t_end = onset_time + winaft * 3600

    temp = temp[(temp['charttime'] >= t_start) & (temp['charttime'] < t_end)]
    temp2 = temp2[(temp2['charttime'] >= t_start) & (temp2['charttime'] < t_end)]
    temp3 = temp3[(temp3['charttime'] >= t_start) & (temp3['charttime'] < t_end)]

    # Get unique timestamps
    all_times = np.concatenate([
        temp['charttime'].values,
        temp2['charttime'].values,
        temp3['charttime'].values
    ])

    if len(all_times) == 0:
        return None

    t = np.unique(all_times)

    # Vectorized pivot for chartevents
    frames = []

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

    if len(temp3) > 0:
        mv_data = temp3[['charttime', 'mechvent']].drop_duplicates(subset=['charttime'], keep='last')
        mv_data = mv_data.set_index('charttime')
        frames.append(mv_data)

    if not frames:
        return None

    # Combine all frames
    patient_df = pd.concat(frames, axis=1)
    # Handle duplicate columns by taking last non-null
    patient_df = patient_df.groupby(level=0, axis=1).last()

    # Reindex to all unique timestamps
    patient_df = patient_df.reindex(t)
    patient_df.index.name = 'charttime'
    patient_df = patient_df.reset_index()
    patient_df['stay_id'] = icustayid

    return patient_df


def handle_outliers(df):
    """Handle outliers - corrected to match original logic exactly"""
    print('Handling outliers in patient timeseries data')

    # All simple "set to NaN" rules
    outlier_rules_nan = [
        ('weight_kg', lambda s: s > 300),
        ('weight_lb', lambda s: s > 660),
        ('heart_rate', lambda s: s > 250),
        ('sbp_arterial', lambda s: s > 300),
        ('map', lambda s: (s < 0) | (s > 200)),
        ('dbp_arterial', lambda s: (s < 0) | (s > 200)),
        ('respiratory_rate', lambda s: s > 80),
        ('spo2', lambda s: s > 150),
        ('oxygen_flow', lambda s: s > 70),
        ('peep', lambda s: (s < 0) | (s > 40)),
        ('tidal_volume', lambda s: s > 1800),
        ('minute_volume', lambda s: s > 50),
        ('potassium', lambda s: (s < 1) | (s > 15)),
        ('sodium', lambda s: (s < 95) | (s > 178)),
        ('chloride', lambda s: (s < 70) | (s > 150)),
        ('glucose', lambda s: (s < 1) | (s > 1000)),
        ('creatinine', lambda s: s > 150),
        ('magnesium', lambda s: s > 10),
        ('calcium_total', lambda s: s > 20),
        ('calcium_ionized', lambda s: s > 5),
        ('total_co2', lambda s: s > 120),
        ('ast', lambda s: s > 10000),
        ('alt', lambda s: s > 10000),
        ('hemoglobin', lambda s: s > 20),
        ('hematocrit', lambda s: s > 65),
        ('wbc', lambda s: s > 500),
        ('platelets', lambda s: s > 2000),
        ('inr', lambda s: s > 20),
        ('ph_arterial', lambda s: (s < 6.7) | (s > 8)),
        ('arterial_o2_pressure', lambda s: s > 700),
        ('arterial_co2_pressure', lambda s: s > 200),
        ('arterial_base_excess', lambda s: s < -50),
        ('lactic_acid', lambda s: s > 30),
        ('bilirubin_total', lambda s: s > 30),
    ]

    for col, cond_func in outlier_rules_nan:
        if col in df.columns:
            df.loc[cond_func(df[col]), col] = np.nan

    # SpO2: cap at 100 (after removing > 150)
    if 'spo2' in df.columns:
        df.loc[df['spo2'] > 100, 'spo2'] = 100

    # Temperature: move misplaced values from temp_C to temp_F
    if 'temp_C' in df.columns and 'temp_F' in df.columns:
        mask = (df['temp_C'] > 90) & (df['temp_F'].isna())
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[df['temp_C'] > 90, 'temp_C'] = np.nan

    # FiO2: MUST match original order exactly:
    # 1. > 100 → NaN
    # 2. < 1 → multiply by 100
    # 3. < 20 → NaN
    if 'fio2' in df.columns:
        df.loc[df['fio2'] > 100, 'fio2'] = np.nan
        mask_fraction = df['fio2'] < 1
        df.loc[mask_fraction, 'fio2'] = df.loc[mask_fraction, 'fio2'] * 100
        df.loc[df['fio2'] < 20, 'fio2'] = np.nan

    return df


def estimate_gcs_from_rass(df):
    """Vectorized GCS estimation from RASS"""
    if 'gcs' not in df.columns:
        df['gcs'] = np.nan

    if 'richmond_ras' not in df.columns:
        return df

    # RASS to GCS mapping
    rass_to_gcs = {4: 15, 3: 15, 2: 15, 1: 15, 0: 15,
                   -1: 14, -2: 12, -3: 11, -4: 6, -5: 3}

    gcs_missing = df['gcs'].isna()
    rass_values = df['richmond_ras']

    for rass_val, gcs_val in rass_to_gcs.items():
        mask = gcs_missing & (rass_values == rass_val)
        df.loc[mask, 'gcs'] = gcs_val

    return df


def estimate_fio2(df):
    """Estimate FiO2 values based on O2 flow rate - optimized"""
    df = df.copy()

    # Combine all oxygen flow measurements
    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    existing_flow_cols = [c for c in flow_columns if c in df.columns]
    if existing_flow_cols:
        df['combined_o2_flow'] = df[existing_flow_cols].bfill(axis=1).iloc[:, 0]
    else:
        df['combined_o2_flow'] = np.nan

    fio2_missing = df['fio2'].isna() if 'fio2' in df.columns else pd.Series(True, index=df.index)
    if 'fio2' not in df.columns:
        df['fio2'] = np.nan

    has_flow = df['combined_o2_flow'].notna()

    if 'oxygen_flow_device' not in df.columns:
        df.drop('combined_o2_flow', axis=1, errors='ignore')
        return df

    device = df['oxygen_flow_device'].astype(str)
    flow = df['combined_o2_flow']

    # Case 1: nasal cannula / none
    mask1 = fio2_missing & has_flow & device.isin(['0', '2'])
    if mask1.any():
        fio2_vals = np.full(mask1.sum(), np.nan)
        f = flow[mask1].values
        thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        values = [24, 28, 32, 36, 40, 44, 50, 55, 62, 70]
        fio2_vals[:] = 70  # default for > 15
        for t, v in zip(thresholds, values):
            fio2_vals[f <= t] = v
        df.loc[mask1, 'fio2'] = fio2_vals

    # Case 2: no flow, nasal cannula / none -> room air
    mask2 = fio2_missing & (~has_flow) & device.isin(['0', '2'])
    df.loc[mask2, 'fio2'] = 21

    # Recalculate fio2_missing after above assignments
    fio2_missing = df['fio2'].isna()

    # Case 3: face mask types
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12']
    mask3 = fio2_missing & has_flow & device.isin(face_mask_types)
    if mask3.any():
        fio2_vals = np.full(mask3.sum(), np.nan)
        f = flow[mask3].values
        thresholds = [4, 6, 8, 10, 12, 15]
        values = [36, 40, 58, 66, 69, 75]
        fio2_vals[:] = 75
        for t, v in zip(thresholds, values):
            fio2_vals[f <= t] = v
        df.loc[mask3, 'fio2'] = fio2_vals

    fio2_missing = df['fio2'].isna()

    # Case 4: non-rebreather
    mask4 = fio2_missing & has_flow & (device == '7')
    if mask4.any():
        f = flow[mask4].values
        fio2_vals = np.where(f >= 15, 100,
                   np.where(f >= 10, 90,
                   np.where(f > 8, 80,
                   np.where(f > 6, 70, 60))))
        df.loc[mask4, 'fio2'] = fio2_vals

    fio2_missing = df['fio2'].isna()

    # Case 5: CPAP/BiPAP
    mask5 = fio2_missing & has_flow & (device == '13')
    if mask5.any():
        f = flow[mask5].values
        fio2_vals = np.where(f >= 15, 100,
                   np.where(f >= 10, 80, 60))
        df.loc[mask5, 'fio2'] = fio2_vals

    fio2_missing = df['fio2'].isna()

    # Case 6: Oxymizer
    mask6 = fio2_missing & has_flow & (device == '14')
    if mask6.any():
        f = flow[mask6].values
        fio2_vals = np.where(f >= 10, 80,
                   np.where(f >= 5, 60, 40))
        df.loc[mask6, 'fio2'] = fio2_vals

    df.drop('combined_o2_flow', axis=1, inplace=True)
    return df


def handle_unit_conversions(df):
    """Handle various unit conversions - vectorized"""
    if 'temp_F' in df.columns and 'temp_C' in df.columns:
        # tempF values that look like Celsius
        mask = (df['temp_F'] > 25) & (df['temp_F'] < 45)
        df.loc[mask, 'temp_C'] = df.loc[mask, 'temp_F']
        df.loc[mask, 'temp_F'] = np.nan

        # tempC values that look like Fahrenheit
        mask = df['temp_C'] > 70
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[mask, 'temp_C'] = np.nan

        # Convert C to F
        mask = df['temp_C'].notna() & df['temp_F'].isna()
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C'] * 1.8 + 32

        # Convert F to C
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
    """Optimized sample and hold using pandas groupby + ffill with limits"""
    print('Performing sample and hold interpolation')

    df = df.copy()
    df = df.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    # For each column, calculate the max number of rows that correspond to hold_time
    # We need to estimate time gaps per patient, so we use a row-based approach with groupby

    cols_to_process = [col for col in vitalslab_hold if col in df.columns
                       and np.issubdtype(df[col].dtype, np.number)]

    if not cols_to_process:
        return df

    # Pre-compute time differences within each stay
    # For sample-and-hold, we need to respect hold_time in seconds
    # Strategy: use groupby forward fill with a time-based limit

    # Convert charttime to timedelta for time-aware ffill
    # pandas ffill with limit based on time requires a DatetimeIndex or manual approach

    # Efficient approach: iterate by stay_id groups and use numpy
    stay_ids = df['stay_id'].values
    charttimes = df['charttime'].values.astype(float)

    # Find group boundaries
    stay_change = np.concatenate([[True], stay_ids[1:] != stay_ids[:-1]])
    group_starts = np.where(stay_change)[0]
    group_ends = np.concatenate([group_starts[1:], [len(df)]])

    for col in cols_to_process:
        hold_period = vitalslab_hold[col] * 3600
        col_values = df[col].values.astype(float).copy()

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
    """Combines multiple measurement sources into unified time series with fixed timesteps."""

    start_time = patient_data['start_time']
    stay_id = patient_data['stay_id']
    measurements = patient_data['measurements']
    fluid_data = patient_data['fluid']
    vaso_data = patient_data['vasopressors']
    uo_data = patient_data['urine_output']
    abx_data = patient_data['antibiotics']
    demographics = patient_data['demographics']

    # Get actual data time range
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

    # Precompute window boundaries
    window_starts = first_time + np.arange(num_timesteps) * timestep * 3600
    window_ends = window_starts + timestep * 3600

    # Pre-filter and sort data
    meas_times = measurements['charttime'].values
    meas_cols = [c for c in measurements.columns if c not in ['stay_id']]

    # Vectorized measurement aggregation
    processed_rows = []

    # Precompute abx info
    if abx_data is not None and len(abx_data) > 0:
        abx_data = abx_data.copy()
        abx_data['stay_id'] = abx_data['stay_id'].astype('int64')
        first_abx_time = abx_data['starttime'].min()
        abx_starts = abx_data['starttime'].values
        abx_stops = abx_data['stoptime'].values
        abx_drugs = abx_data['drug'].values
    else:
        first_abx_time = None

    # Precompute fluid data arrays
    if fluid_data is not None and len(fluid_data) > 0:
        fluid_starts = fluid_data['starttime'].values
        fluid_ends = fluid_data['endtime'].values
        fluid_amounts = fluid_data['amount'].values
    else:
        fluid_starts = np.array([])

    # Precompute vaso data arrays
    if vaso_data is not None and len(vaso_data) > 0:
        vaso_starts = vaso_data['starttime'].values
        vaso_ends = vaso_data['endtime'].values
        vaso_rates = vaso_data['rate_std'].values
    else:
        vaso_starts = np.array([])

    # Precompute UO data arrays
    if uo_data is not None and len(uo_data) > 0:
        uo_times = uo_data['charttime'].values
        uo_values = uo_data['value'].values
    else:
        uo_times = np.array([])

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
            fluid_step = fluid_amounts[step_mask].sum()
            total_mask = fluid_ends < we
            fluid_total = fluid_amounts[total_mask].sum()
        else:
            fluid_step = 0
            fluid_total = 0

        # Vasopressors
        if len(vaso_starts) > 0:
            vmask = (vaso_starts <= we) & (vaso_ends >= ws)
            if vmask.any():
                window_rates = vaso_rates[vmask]
                vaso_median = np.nanmedian(window_rates)
                vaso_max = np.nanmax(window_rates)
            else:
                vaso_median = 0
                vaso_max = 0
        else:
            vaso_median = 0
            vaso_max = 0

        # Urine output
        if len(uo_times) > 0:
            uo_mask = (uo_times >= ws) & (uo_times < we)
            uo_step = uo_values[uo_mask].sum()
            uo_total_mask = uo_times < we
            uo_total = uo_values[uo_total_mask].sum()
        else:
            uo_step = 0
            uo_total = 0

        # Antibiotics
        if first_abx_time is not None:
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

    # Pre-index data for fast lookup
    onset_lookup = data_dict['onset'].set_index('stay_id')['onset_time'].to_dict()
    demog_lookup = data_dict['demog'].set_index('stay_id')

    # Pre-group supplementary data
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
    """Linearly interpolates gaps (NaN values) in a time series."""
    y = np.copy(x)
    nan_mask = np.isnan(x)
    valid_indices = np.where(~nan_mask)[0]

    if len(valid_indices) < 2:
        return y

    # Only interpolate between first and last valid
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
    """Handle missing values through interpolation and KNN imputation"""
    print('Handling missing values...')

    measurement_cols = [col for col in df.columns if col not in [
        'timestep', 'stay_id', 'timestamp', 'gender', 'age',
        'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 'fluid_total', 'fluid_step',
        'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]]

    # Print missingness statistics
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

    # Calculate missingness per column
    miss = df[measurement_cols].isna().sum() / len(df)

    # Drop columns with too much missing
    cols_to_keep = miss[miss < missing_threshold].index.tolist()
    non_meas_cols = [c for c in df.columns if c not in measurement_cols]
    df = df[non_meas_cols + cols_to_keep]

    # Linear interpolation for columns with <5% missing
    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    low_missing_cols = [c for c in low_missing_cols if c in df.columns]
    for col in low_missing_cols:
        df[col] = fixgaps(df[col].values)

    # KNN imputation for remaining
    cols_for_knn = [c for c in cols_to_keep if c not in low_missing_cols and c in df.columns]
    if cols_for_knn:
        ref = df[cols_for_knn].values

        chunk_size = 9999
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        print(f'KNN imputation: {total_chunks} chunks')
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            ref[i:chunk_end, :] = KNN(k=1, verbose=0).fit_transform(ref[i:chunk_end, :])

        df[cols_for_knn] = ref

    return df


def calculate_derived_variables(df):
    """Calculate derived variables - corrected SOFA CV to match original"""
    print('Computing derived variables: P/F ratio, Shock Index, SOFA, SIRS...')

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

    # P/F ratio
    if 'arterial_o2_pressure' in df.columns and 'fio2' in df.columns:
        df['pf_ratio'] = df['arterial_o2_pressure'] / (df['fio2'] / 100)
    else:
        df['pf_ratio'] = np.nan

    # Shock Index
    if 'heart_rate' in df.columns and 'sbp_arterial' in df.columns:
        df['shock_index'] = df['heart_rate'] / df['sbp_arterial']
        df.loc[np.isinf(df['shock_index']), 'shock_index'] = np.nan
        df['shock_index'] = df['shock_index'].fillna(df['shock_index'].mean())
    else:
        df['shock_index'] = np.nan

    # Extract arrays
    pf = df['pf_ratio'].values if 'pf_ratio' in df.columns else np.full(len(df), np.nan)
    plt_vals = df['platelets'].values if 'platelets' in df.columns else np.full(len(df), np.nan)
    bili = df['bilirubin_total'].values if 'bilirubin_total' in df.columns else np.full(len(df), np.nan)
    map_vals = df['map'].values if 'map' in df.columns else np.full(len(df), np.nan)
    vaso_max = df['vaso_max'].values
    gcs_vals = df['gcs'].values if 'gcs' in df.columns else np.full(len(df), np.nan)
    cr_vals = df['creatinine'].values if 'creatinine' in df.columns else np.full(len(df), np.nan)
    uo_vals = df['uo_step'].values if 'uo_step' in df.columns else np.full(len(df), np.nan)

    # SOFA Respiratory
    sofa_resp = np.zeros(len(df), dtype=int)
    valid = ~np.isnan(pf)
    sofa_resp[valid & (pf < 100)] = 4
    sofa_resp[valid & (pf >= 100) & (pf < 200)] = 3
    sofa_resp[valid & (pf >= 200) & (pf < 300)] = 2
    sofa_resp[valid & (pf >= 300) & (pf < 400)] = 1
    df['sofa_resp'] = sofa_resp

    # SOFA Coagulation
    sofa_coag = np.zeros(len(df), dtype=int)
    valid = ~np.isnan(plt_vals)
    sofa_coag[valid & (plt_vals < 20)] = 4
    sofa_coag[valid & (plt_vals >= 20) & (plt_vals < 50)] = 3
    sofa_coag[valid & (plt_vals >= 50) & (plt_vals < 100)] = 2
    sofa_coag[valid & (plt_vals >= 100) & (plt_vals < 150)] = 1
    df['sofa_coag'] = sofa_coag

    # SOFA Liver
    sofa_liver = np.zeros(len(df), dtype=int)
    valid = ~np.isnan(bili)
    sofa_liver[valid & (bili >= 12)] = 4
    sofa_liver[valid & (bili >= 6) & (bili < 12)] = 3
    sofa_liver[valid & (bili >= 2) & (bili < 6)] = 2
    sofa_liver[valid & (bili >= 1.2) & (bili < 2)] = 1
    df['sofa_liver'] = sofa_liver

    # SOFA Cardiovascular - MATCHES ORIGINAL LOGIC EXACTLY
    # Original: checks MAP first, then vaso only if MAP conditions don't apply
    # if map >= 70: 0
    # elif map >= 65: 1
    # elif map < 65: 2
    # elif vaso <= 0.1: 3
    # elif vaso > 0.1: 4
    # else: 0
    #
    # Note: in the original, once map < 65 is True, it returns 2
    # and never checks vaso. Vaso is only checked if map is NaN.
    sofa_cv = np.zeros(len(df), dtype=int)
    valid_map = ~np.isnan(map_vals)
    map_na = np.isnan(map_vals)
    vaso_na = np.isnan(vaso_max)

    # If both NaN: 0 (default)
    # If MAP valid and >= 70: 0
    # If MAP valid and >= 65 and < 70: 1
    # If MAP valid and < 65: 2
    sofa_cv[valid_map & (map_vals < 70) & (map_vals >= 65)] = 1
    sofa_cv[valid_map & (map_vals < 65)] = 2

    # If MAP is NaN but vaso is valid:
    sofa_cv[map_na & ~vaso_na & (vaso_max <= 0.1)] = 3
    sofa_cv[map_na & ~vaso_na & (vaso_max > 0.1)] = 4

    df['sofa_cv'] = sofa_cv

    # SOFA CNS
    sofa_cns = np.zeros(len(df), dtype=int)
    valid = ~np.isnan(gcs_vals)
    sofa_cns[valid & (gcs_vals <= 5)] = 4
    sofa_cns[valid & (gcs_vals > 5) & (gcs_vals <= 9)] = 3
    sofa_cns[valid & (gcs_vals > 9) & (gcs_vals <= 12)] = 2
    sofa_cns[valid & (gcs_vals > 12) & (gcs_vals <= 14)] = 1
    df['sofa_cns'] = sofa_cns

    # SOFA Renal
    sofa_renal = np.zeros(len(df), dtype=int)
    valid_cr = ~np.isnan(cr_vals)
    sofa_renal[valid_cr & (cr_vals >= 5)] = 4
    sofa_renal[valid_cr & (cr_vals >= 3.5) & (cr_vals < 5)] = 3
    sofa_renal[valid_cr & (cr_vals >= 2.0) & (cr_vals < 3.5)] = 2
    sofa_renal[valid_cr & (cr_vals >= 1.2) & (cr_vals < 2.0)] = 1
    # UO-based only where creatinine is NaN
    no_cr = np.isnan(cr_vals)
    valid_uo = ~np.isnan(uo_vals)
    sofa_renal[no_cr & valid_uo & (uo_vals < 34)] = 4
    sofa_renal[no_cr & valid_uo & (uo_vals >= 34) & (uo_vals < 84)] = 3
    df['sofa_renal'] = sofa_renal

    df['sofa_score'] = df['sofa_resp'] + df['sofa_coag'] + df['sofa_liver'] + \
                       df['sofa_cv'] + df['sofa_cns'] + df['sofa_renal']

    for comp in ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cv', 'sofa_cns', 'sofa_renal']:
        print(f"\n{comp} distribution:")
        print(df[comp].value_counts().sort_index())

    # Vectorized SIRS
    sirs = np.zeros(len(df), dtype=int)

    if 'temp_C' in df.columns:
        tc = df['temp_C'].values
        valid = ~np.isnan(tc)
        sirs[valid & ((tc >= 38) | (tc <= 36))] += 1

    if 'heart_rate' in df.columns:
        hr = df['heart_rate'].values
        valid = ~np.isnan(hr)
        sirs[valid & (hr > 90)] += 1

    rr = df['respiratory_rate'].values if 'respiratory_rate' in df.columns else np.full(len(df), np.nan)
    co2 = df['arterial_co2_pressure'].values if 'arterial_co2_pressure' in df.columns else np.full(len(df), np.nan)
    valid_rr = ~np.isnan(rr)
    valid_co2 = ~np.isnan(co2)
    resp_crit = (valid_rr & (rr >= 20)) | (valid_co2 & (co2 <= 32))
    sirs[resp_crit] += 1

    if 'wbc' in df.columns:
        wbc = df['wbc'].values
        valid = ~np.isnan(wbc)
        sirs[valid & ((wbc >= 12) | (wbc < 4))] += 1

    df['sirs_score'] = sirs

    return df

def apply_exclusion_criteria(df):
    """Apply exclusion criteria - vectorized"""
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

    # Early deaths (vectorized)
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

    # Non-sepsis (SOFA < 2 throughout)
    max_sofa = df.groupby('stay_id')['sofa_score'].max()
    non_sepsis_stays = max_sofa[max_sofa < 2].index.values
    df = df[~df['stay_id'].isin(non_sepsis_stays)]
    excluded_counts['non_sepsis'] = len(non_sepsis_stays)

    final_patients = df['stay_id'].nunique()
    print("\nExclusion Statistics:")
    print("-" * 50)
    print(f"Initial patient count: {initial_patients}")
    for reason, count in excluded_counts.items():
        print(f"Excluded due to {reason}: {count}")
    print(f"Final patient count: {final_patients}")
    print(f"Total excluded: {initial_patients - final_patients}")
    print("-" * 50)

    return df


def add_sepsis_flag(df):
    """Add sepsis flag - vectorized"""
    print('Adding sepsis flags to trajectories')

    df = df.sort_values(['stay_id', 'timestamp']).reset_index(drop=True)
    df['sepsis'] = 0

    # Find first occurrence of SOFA >= 2 per patient
    sepsis_mask = df['sofa_score'] >= 2
    sepsis_df = df[sepsis_mask].groupby('stay_id').head(1)

    # Mark onset
    df.loc[sepsis_df.index, 'sepsis'] = 1

    # Mark censored (all rows after onset for each patient)
    for stay_id, onset_idx in sepsis_df.groupby('stay_id').apply(lambda x: x.index[0]).items():
        subsequent = (df['stay_id'] == stay_id) & (df.index > onset_idx)
        df.loc[subsequent, 'sepsis'] = 2

    total_patients = df['stay_id'].nunique()
    sepsis_patients = len(sepsis_df['stay_id'].unique())

    print("\nSepsis Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing sepsis: {sepsis_patients} ({sepsis_patients / total_patients * 100:.1f}%)")
    print(f"Timesteps with sepsis onset: {(df['sepsis'] == 1).sum()}")
    print(f"Censored timesteps: {(df['sepsis'] == 2).sum()}")
    print("-" * 50)

    return df


def add_septic_shock_flag(df):
    """Add septic shock flag - vectorized"""
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

    # Process per patient using groupby
    for stay_id, group in df.groupby('stay_id'):
        group_sorted = group.sort_values('timestamp')
        rolling_fluid = group_sorted['fluid_step'].rolling(window=WINDOW_STEPS, min_periods=1).sum()

        shock_cond = (
            (rolling_fluid.values >= MIN_FLUID_THRESHOLD) &
            (group_sorted['map'].values < MAP_THRESHOLD) &
            (group_sorted['lactic_acid'].values > LACTATE_THRESHOLD)
        )

        if shock_cond.any():
            first_shock_pos = np.argmax(shock_cond)
            shock_idx = group_sorted.index[first_shock_pos]
            df.loc[shock_idx, 'septic_shock'] = 1

            subsequent = group_sorted.index[first_shock_pos + 1:]
            df.loc[subsequent, 'septic_shock'] = 2

    total_patients = df['stay_id'].nunique()
    shock_patients = df.loc[df['septic_shock'] == 1, 'stay_id'].nunique()

    print("\nSeptic Shock Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing shock: {shock_patients} ({shock_patients / total_patients * 100:.1f}%)")
    print(f"Timesteps with shock onset: {(df['septic_shock'] == 1).sum()}")
    print(f"Censored timesteps: {(df['septic_shock'] == 2).sum()}")
    print("-" * 50)

    return df


def main():
    args = parse_args()

    # Load all required data
    data = load_processed_files()
    measurements, code_to_concept, hold_times = load_measurement_mappings()

    # Load onset data
    onset = data['onset']

    # Sample subjects if specified
    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, random_state=42)

    # Pre-index data by stay_id for fast lookup
    print('Pre-indexing data by stay_id...')
    ce_groups = dict(list(data['ce'].groupby('stay_id')))
    labU_groups = dict(list(data['labU'].groupby('stay_id')))
    MV_groups = dict(list(data['MV'].groupby('stay_id')))

    # Create a modified data dict for per-patient lookup
    data_indexed = {
        'ce_groups': ce_groups,
        'labU_groups': labU_groups,
        'MV_groups': MV_groups,
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

    # Combine all patient data
    init_traj = pd.concat(all_patient_data, ignore_index=True)

    # Handle outliers
    init_traj = handle_outliers(init_traj)

    # Estimate GCS from RASS
    init_traj = estimate_gcs_from_rass(init_traj)

    # Estimate FiO2 from O2 flow
    init_traj = estimate_fio2(init_traj)

    # Handle unit conversions
    init_traj = handle_unit_conversions(init_traj)

    # Sample and hold
    init_traj = sample_and_hold(init_traj, hold_times)

    # Standardize trajectories
    init_traj = standardize_patient_trajectories(
        init_traj,
        data,
        timestep=args.timestep,
        window_before=args.window_before,
        window_after=args.window_after
    )

    # Handle missing values
    init_traj = handle_missing_values(init_traj, args.missing_threshold)

    # Calculate derived variables
    init_traj = calculate_derived_variables(init_traj)

    # Apply exclusion criteria
    init_traj = apply_exclusion_criteria(init_traj)

    # Add septic shock flags
    init_traj = add_septic_shock_flag(init_traj)

    # Add sepsis flags
    init_traj = add_sepsis_flag(init_traj)

    # Print missingness statistics
    missing_pct = (init_traj.isna().sum() / len(init_traj)) * 100
    print("\nMissing value percentages:")
    for col, pct in missing_pct.sort_values(ascending=False).items():
        if pct > 0:
            print(f"{col}: {pct:.1f}%")

    # Save
    output_path = f"{args.output_dir}/patient_timeseries_v4.csv"
    init_traj.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def process_patient_measurements_fast(data_indexed, code_to_concept, icustayid, onset_time, winb4=24, winaft=72):
    """Fast version using pre-grouped data"""
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

    # Combine all frames
    patient_df = pd.concat(frames, axis=1)
    # Handle duplicate columns
    if patient_df.columns.duplicated().any():
        patient_df = patient_df.loc[:, ~patient_df.columns.duplicated(keep='last')]

    patient_df.index.name = 'charttime'
    patient_df = patient_df.reset_index()
    patient_df['stay_id'] = icustayid

    return patient_df


if __name__ == "__main__":
    main()