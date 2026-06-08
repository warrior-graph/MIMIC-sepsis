import argparse
import json
import numpy as np
import polars as pl
import pyprind
import os
import gc
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer
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
                       help="Threshold for dropping columns with missing values (default: 0.8)")
    parser.add_argument("--low_missing_threshold", type=float, default=0.05,
                       help="Threshold for using linear interpolation instead of KNN (default: 0.05)")
    parser.add_argument("--knn_neighbors", type=int, default=1,
                       help="Number of neighbors to use for KNN imputation (default: 1)")
    parser.add_argument("--knn_chunk_size", type=int, default=5000,
                       help="Chunk size for KNN imputation processing (default: 5000)")
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
    parser.add_argument("--window_before", type=int, default = 24,
                       help="Hours to include before onset time (default: 24)")
    parser.add_argument("--window_after", type=int, default= 72,
                       help="Hours to include after onset time (default: 72)")
    parser.add_argument("--notes_dir", type=str, default="processed_files",
                       help="Directory containing processed notes files")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of subjects to sample for testing (default: None, use all subjects)")
    # Added chunk size parameter for memory control
    parser.add_argument("--patient_chunk_size", type=int, default=2500,
                       help="Number of patients to process in RAM at one time (default: 2500)")
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
        # Handle dynamic paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Depending on where you run this, you may need to adjust the directory traversal
        # Assuming processed_files is in the same directory you run the script from
        filepath = os.path.join(os.getcwd(), 'processed_files', filename)
        
        with open(filepath, 'r') as f:
            first_line = f.readline()
            header = first_line.strip().split('|') if first_line else []
            
        overrides = {}
        if 'dose_val_rx' in header:
            overrides['dose_val_rx'] = pl.String
            
        df = pl.read_csv(
            filepath, 
            separator='|',
            infer_schema_length=10000,
            schema_overrides=overrides,
            ignore_errors=True
        )
        
        # MEMORY OPTIMIZATION 1: Downcast all 64-bit floats to 32-bit floats
        float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Float64]
        if float_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
            
        # UNIVERSAL TYPE ENFORCEMENT
        if 'stay_id' in df.columns:
            df = df.with_columns(pl.col('stay_id').cast(pl.Int64))
            
        data[key] = df
        
    return data

def load_measurement_mappings():
    print('Loading measurement mappings')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(script_dir, "ReferenceFiles", "measurement_mappings.json")
    
    # Fallback if running from root dir
    if not os.path.exists(mapping_path):
        mapping_path = os.path.join(os.getcwd(), "ReferenceFiles", "measurement_mappings.json")
        
    with open(mapping_path, "r") as f:
        measurements = json.load(f)
    
    code_to_concept = {}
    hold_times = {}
    for concept, info in measurements.items():
        for code in info['codes']:
            code_to_concept[code] = concept
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']
            
    return measurements, code_to_concept, hold_times

def build_raw_measurements(data, onset, code_to_concept, winb4, winaft):
    bounds = onset.select([
        pl.col('stay_id'),
        (pl.col('onset_time') - winb4 * 3600).alias('start_bound'),
        (pl.col('onset_time') + winaft * 3600).alias('end_bound')
    ])
    
    mapping_df = pl.DataFrame({
        'itemid': [int(k) for k in code_to_concept.keys()],
        'concept': list(code_to_concept.values())
    }).with_columns(pl.col('itemid').cast(pl.Int64))
    
    ce = data['ce'].join(bounds, on='stay_id', how='inner')
    ce = ce.filter((pl.col('charttime') >= pl.col('start_bound')) & (pl.col('charttime') < pl.col('end_bound')))
    ce = ce.join(mapping_df, on='itemid', how='inner')
    ce_mapped = ce.select(['stay_id', 'charttime', 'concept', pl.col('valuenum').cast(pl.Float32)])
    
    labU = data['labU'].join(bounds, on='stay_id', how='inner')
    labU = labU.filter((pl.col('charttime') >= pl.col('start_bound')) & (pl.col('charttime') < pl.col('end_bound')))
    labU = labU.join(mapping_df, on='itemid', how='inner')
    labU_mapped = labU.select(['stay_id', 'charttime', 'concept', pl.col('valuenum').cast(pl.Float32)])
    
    MV = data['MV'].join(bounds, on='stay_id', how='inner')
    MV = MV.filter((pl.col('charttime') >= pl.col('start_bound')) & (pl.col('charttime') < pl.col('end_bound')))
    MV_mapped = MV.with_columns(pl.lit('mechvent').alias('concept')).rename({'mechvent': 'valuenum'})
    MV_mapped = MV_mapped.select(['stay_id', 'charttime', 'concept', pl.col('valuenum').cast(pl.Float32)])
    
    all_events = pl.concat([ce_mapped, labU_mapped, MV_mapped])
    
    raw = all_events.group_by(['stay_id', 'charttime', 'concept']).agg(pl.col('valuenum').first())
    
    # FIX: Changed 'columns' keyword to 'on' to fix the DeprecationWarning
    raw_pivoted = raw.pivot(values='valuenum', index=['stay_id', 'charttime'], on='concept')
    
    for c in code_to_concept.values():
        if c not in raw_pivoted.columns:
            raw_pivoted = raw_pivoted.with_columns(pl.lit(None).cast(pl.Float32).alias(c))
            
    if 'mechvent' not in raw_pivoted.columns:
        raw_pivoted = raw_pivoted.with_columns(pl.lit(None).cast(pl.Float32).alias('mechvent'))
        
    return raw_pivoted, bounds

def _cap_outlier(expr, max_val=None, min_val=None, replace_with=None):
    if max_val is not None:
        expr = pl.when(expr > max_val).then(replace_with).otherwise(expr)
    if min_val is not None:
        expr = pl.when(expr < min_val).then(replace_with).otherwise(expr)
    return expr

def handle_outliers(df):
    exprs = []
    cols = df.columns
    
    def safe_cap(col, max_val=None, min_val=None, replace_with=None):
        if col in cols:
            exprs.append(_cap_outlier(pl.col(col), max_val, min_val, replace_with).alias(col))

    safe_cap('weight_kg', max_val=300)
    safe_cap('weight_lb', max_val=660)
    safe_cap('heart_rate', max_val=250)
    safe_cap('sbp_arterial', max_val=300)
    safe_cap('map', min_val=0, max_val=200)
    safe_cap('dbp_arterial', min_val=0, max_val=200)
    safe_cap('respiratory_rate', max_val=80)
    
    if 'spo2' in cols:
        exprs.append(
            pl.when(pl.col('spo2') > 150).then(None)
              .when(pl.col('spo2') > 100).then(100)
              .otherwise(pl.col('spo2')).alias('spo2')
        )
        
    if 'temp_C' in cols and 'temp_F' in cols:
        mask = (pl.col('temp_C') > 90) & pl.col('temp_F').is_null()
        exprs.append(pl.when(mask).then(pl.col('temp_C')).otherwise(pl.col('temp_F')).alias('temp_F'))
        
    if 'temp_C' in cols:
        exprs.append(pl.when(pl.col('temp_C') > 90).then(None).otherwise(pl.col('temp_C')).alias('temp_C'))
        
    if 'fio2' in cols:
        fio2_expr = pl.when(pl.col('fio2') > 100).then(None)\
                      .when(pl.col('fio2') < 1).then(pl.col('fio2') * 100)\
                      .otherwise(pl.col('fio2'))
        fio2_expr = pl.when(fio2_expr < 20).then(None).otherwise(fio2_expr)
        exprs.append(fio2_expr.alias('fio2'))

    safe_cap('oxygen_flow', max_val=70)
    safe_cap('peep', min_val=0, max_val=40)
    safe_cap('tidal_volume', max_val=1800)
    safe_cap('minute_volume', max_val=50)
    safe_cap('potassium', min_val=1, max_val=15)
    safe_cap('sodium', min_val=95, max_val=178)
    safe_cap('chloride', min_val=70, max_val=150)
    safe_cap('glucose', min_val=1, max_val=1000)
    safe_cap('creatinine', max_val=150)
    safe_cap('magnesium', max_val=10)
    safe_cap('calcium_total', max_val=20)
    safe_cap('calcium_ionized', max_val=5)
    safe_cap('total_co2', max_val=120)
    safe_cap('ast', max_val=10000)
    safe_cap('alt', max_val=10000)
    safe_cap('hemoglobin', max_val=20)
    safe_cap('hematocrit', max_val=65)
    safe_cap('wbc', max_val=500)
    safe_cap('platelets', max_val=2000)
    safe_cap('inr', max_val=20)
    safe_cap('ph_arterial', min_val=6.7, max_val=8)
    safe_cap('arterial_o2_pressure', max_val=700)
    safe_cap('arterial_co2_pressure', max_val=200)
    
    if 'arterial_base_excess' in cols:
        exprs.append(pl.when(pl.col('arterial_base_excess') < -50).then(None).otherwise(pl.col('arterial_base_excess')).alias('arterial_base_excess'))
        
    safe_cap('lactic_acid', max_val=30)
    safe_cap('bilirubin_total', max_val=30)
    
    if exprs:
        df = df.with_columns(exprs)
        
    return df

def estimate_gcs_from_rass(df):
    if 'gcs' not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias('gcs'))
        
    if 'richmond_ras' in df.columns:
        gcs = pl.col('gcs')
        rass = pl.col('richmond_ras')
        
        df = df.with_columns(
            pl.when(gcs.is_null() & rass.is_in([0, 1, 2, 3, 4])).then(15)
              .when(gcs.is_null() & (rass == -1)).then(14)
              .when(gcs.is_null() & (rass == -2)).then(12)
              .when(gcs.is_null() & (rass == -3)).then(11)
              .when(gcs.is_null() & (rass == -4)).then(6)
              .when(gcs.is_null() & (rass == -5)).then(3)
              .otherwise(gcs).alias('gcs')
        )
    return df

def estimate_fio2(df):
    if 'fio2' not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float32).alias('fio2'))
        
    cols = df.columns
    flow_cols = [c for c in ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate'] if c in cols]
    
    if not flow_cols or 'oxygen_flow_device' not in cols:
        return df

    df = df.with_columns([
        pl.coalesce(flow_cols).alias('combined_o2_flow'),
        pl.col('oxygen_flow_device').cast(pl.Utf8)
    ])
    
    fio2 = pl.col('fio2')
    flow = pl.col('combined_o2_flow')
    dev = pl.col('oxygen_flow_device')
    
    is_missing = fio2.is_null()
    has_flow = flow.is_not_null()
    no_flow = flow.is_null()
    
    def cascade_fio2(base_cond, thresholds, values, current_expr):
        expr = current_expr
        sorted_pairs = sorted(zip(thresholds, values), key=lambda x: x[0])
        for t, v in sorted_pairs:
            expr = pl.when(base_cond & (flow <= t)).then(v).otherwise(expr)
        return expr

    new_fio2 = fio2
    
    c1 = is_missing & has_flow & dev.is_in(['0', '2'])
    new_fio2 = cascade_fio2(c1, [15, 12, 10, 8, 6, 5, 4, 3, 2, 1], [70, 62, 55, 50, 44, 40, 36, 32, 28, 24], new_fio2)

    c2 = is_missing & no_flow & dev.is_in(['0', '2'])
    new_fio2 = pl.when(c2).then(21).otherwise(new_fio2)

    c3 = is_missing & has_flow & dev.is_in(['3', '4', '5', '6', '8', '9', '10', '11', '12'])
    new_fio2 = cascade_fio2(c3, [15, 12, 10, 8, 6, 4], [75, 69, 66, 58, 40, 36], new_fio2)

    c4 = is_missing & has_flow & (dev == '7')
    new_fio2 = pl.when(c4 & (flow >= 15)).then(100) \
                 .when(c4 & (flow >= 10) & (flow < 15)).then(90) \
                 .when(c4 & (flow > 8) & (flow < 10)).then(80) \
                 .when(c4 & (flow > 6) & (flow <= 8)).then(70) \
                 .when(c4 & (flow <= 6)).then(60) \
                 .otherwise(new_fio2)

    c5 = is_missing & has_flow & (dev == '13')
    new_fio2 = pl.when(c5 & (flow >= 15)).then(100) \
                 .when(c5 & (flow >= 10) & (flow < 15)).then(80) \
                 .when(c5 & (flow < 10)).then(60) \
                 .otherwise(new_fio2)

    c6 = is_missing & has_flow & (dev == '14')
    new_fio2 = pl.when(c6 & (flow >= 10)).then(80) \
                 .when(c6 & (flow >= 5) & (flow < 10)).then(60) \
                 .when(c6 & (flow < 5)).then(40) \
                 .otherwise(new_fio2)

    df = df.with_columns(new_fio2.alias('fio2')).drop('combined_o2_flow')
    return df

def handle_unit_conversions(df):
    cols = df.columns
    exprs = []
    
    if 'temp_F' in cols and 'temp_C' in cols:
        cond_to_c = (pl.col('temp_F') > 25) & (pl.col('temp_F') < 45)
        cond_to_f = (pl.col('temp_C') > 70)
        
        c_expr = pl.when(cond_to_c).then(pl.col('temp_F')) \
                   .when(cond_to_f).then(None) \
                   .when(pl.col('temp_C').is_null() & pl.col('temp_F').is_not_null()).then((pl.col('temp_F') - 32) / 1.8) \
                   .otherwise(pl.col('temp_C'))
                   
        f_expr = pl.when(cond_to_f).then(pl.col('temp_C')) \
                   .when(cond_to_c).then(None) \
                   .when(pl.col('temp_F').is_null() & pl.col('temp_C').is_not_null()).then(pl.col('temp_C') * 1.8 + 32) \
                   .otherwise(pl.col('temp_F'))
                   
        exprs.extend([c_expr.alias('temp_C'), f_expr.alias('temp_F')])

    if 'hemoglobin' in cols and 'hematocrit' in cols:
        exprs.append(
            pl.when(pl.col('hematocrit').is_null() & pl.col('hemoglobin').is_not_null())
              .then((pl.col('hemoglobin') * 2.862) + 1.216)
              .otherwise(pl.col('hematocrit')).alias('hematocrit')
        )
        exprs.append(
            pl.when(pl.col('hemoglobin').is_null() & pl.col('hematocrit').is_not_null())
              .then((pl.col('hematocrit') - 1.216) / 2.862)
              .otherwise(pl.col('hemoglobin')).alias('hemoglobin')
        )

    if 'bilirubin_total' in cols and 'bilirubin_direct' in cols:
        exprs.append(
            pl.when(pl.col('bilirubin_direct').is_null() & pl.col('bilirubin_total').is_not_null())
              .then((pl.col('bilirubin_total') * 0.6934) - 0.1752)
              .otherwise(pl.col('bilirubin_direct')).alias('bilirubin_direct')
        )
        exprs.append(
            pl.when(pl.col('bilirubin_total').is_null() & pl.col('bilirubin_direct').is_not_null())
              .then((pl.col('bilirubin_direct') + 0.1752) / 0.6934)
              .otherwise(pl.col('bilirubin_total')).alias('bilirubin_total')
        )
        
    if exprs:
        df = df.with_columns(exprs)
        
    return df

def sample_and_hold(df, vitalslab_hold):
    cols_to_process = [c for c in vitalslab_hold if c in df.columns and df.schema[c].is_numeric()]
    if not cols_to_process:
        return df

    df = df.sort(['stay_id', 'charttime'])
    exprs = []
    
    for col in cols_to_process:
        hold_h = vitalslab_hold[col]
        last_time = pl.when(pl.col(col).is_not_null()).then(pl.col('charttime')).otherwise(None)
        
        ffilled_time = last_time.forward_fill().over('stay_id')
        ffilled_val = pl.col(col).forward_fill().over('stay_id')
        
        valid_mask = (pl.col('charttime') - ffilled_time) <= (hold_h * 3600)
        exprs.append(
            pl.when(valid_mask).then(ffilled_val).otherwise(pl.col(col)).alias(col)
        )
        
    return df.with_columns(exprs)

def standardize_patient_trajectories(raw_pivoted, bounds, data_dict, timestep):
    patient_times = raw_pivoted.group_by('stay_id').agg([
        pl.col('charttime').min().alias('first_charttime'),
        pl.col('charttime').max().alias('last_charttime')
    ]).join(bounds, on='stay_id')
    
    patient_times = patient_times.with_columns([
        pl.max_horizontal('first_charttime', 'start_bound').alias('first_time'),
        pl.min_horizontal('last_charttime', 'end_bound').alias('last_time')
    ])
    
    patient_times = patient_times.with_columns(
        ((pl.col('last_time') - pl.col('first_time')) / (timestep * 3600)).ceil().cast(pl.Int32).alias('num_timesteps')
    )
    
    master_grid = patient_times.select(['stay_id', 'num_timesteps', 'first_time']).with_columns(
        pl.int_ranges(0, pl.col('num_timesteps')).alias('timestep_idx')
    ).explode('timestep_idx')
    
    master_grid = master_grid.with_columns([
        (pl.col('timestep_idx') + 1).alias('timestep'),
        (pl.col('first_time') + pl.col('timestep_idx') * timestep * 3600).alias('timestamp'),
        (pl.col('first_time') + (pl.col('timestep_idx') + 1) * timestep * 3600).alias('window_end')
    ])
    
    raw = raw_pivoted.join(patient_times.select(['stay_id', 'first_time', 'last_time']), on='stay_id')
    raw = raw.filter((pl.col('charttime') >= pl.col('first_time')) & (pl.col('charttime') <= pl.col('last_time')))
    raw = raw.with_columns(
        ((pl.col('charttime') - pl.col('first_time')) / (timestep * 3600)).floor().cast(pl.Int32).alias('timestep_idx')
    )
    
    agg_cols = [c for c in raw_pivoted.columns if c not in ['stay_id', 'charttime', 'first_time', 'last_time']]
    meas_agg = raw.group_by(['stay_id', 'timestep_idx']).agg([pl.col(c).mean() for c in agg_cols])
    grid = master_grid.join(meas_agg, on=['stay_id', 'timestep_idx'], how='left')
    
    step_sec = timestep * 3600
    
    # FIX: Added end_col parameter to handle MIMIC schema inconsistencies ('endtime' vs 'stoptime')
    def get_intervals(df, agg_exprs, end_col='endtime'):
        df_jn = df.join(patient_times.select(['stay_id', 'first_time', 'num_timesteps']), on='stay_id', how='inner')
        df_jn = df_jn.with_columns([
            pl.max_horizontal(0, ((pl.col('starttime') - pl.col('first_time')) / step_sec).floor().cast(pl.Int32)).alias('start_bin'),
            pl.min_horizontal(pl.col('num_timesteps') - 1, ((pl.col(end_col) - 0.001 - pl.col('first_time')) / step_sec).floor().cast(pl.Int32)).alias('end_bin')
        ]).filter(pl.col('start_bin') <= pl.col('end_bin'))
        
        df_exp = df_jn.with_columns(pl.int_ranges(pl.col('start_bin'), pl.col('end_bin') + 1).alias('timestep_idx')).explode('timestep_idx')
        return df_exp.group_by(['stay_id', 'timestep_idx']).agg(agg_exprs)
    
    fluid_step = get_intervals(data_dict['fluid'], [pl.col('amount').sum().alias('fluid_step')])
    vaso_agg = get_intervals(data_dict['vaso'], [pl.col('rate_std').median().alias('vaso_median'), pl.col('rate_std').max().alias('vaso_max')])
    
    # FIX: Pass 'stoptime' explicitly for antibiotics
    abx_agg = get_intervals(data_dict['abx'], [pl.lit(1).alias('abx_given'), pl.col('drug').n_unique().alias('num_abx')], end_col='stoptime')
    
    fluid_comp = data_dict['fluid'].join(patient_times, on='stay_id', how='inner')
    fluid_comp = fluid_comp.with_columns(((pl.col('endtime') - 0.001 - pl.col('first_time')) / step_sec).floor().cast(pl.Int32).alias('comp_bin'))
    fluid_comp = fluid_comp.filter((pl.col('comp_bin') >= 0) & (pl.col('comp_bin') < pl.col('num_timesteps')))
    fluid_total_agg = fluid_comp.group_by(['stay_id', 'comp_bin']).agg(pl.col('amount').sum().alias('amount_ended'))
    
    uo = data_dict['UO'].join(patient_times, on='stay_id', how='inner')
    uo = uo.with_columns(((pl.col('charttime') - pl.col('first_time')) / step_sec).floor().cast(pl.Int32).alias('timestep_idx'))
    uo = uo.filter((pl.col('timestep_idx') >= 0) & (pl.col('timestep_idx') < pl.col('num_timesteps')))
    uo_step = uo.group_by(['stay_id', 'timestep_idx']).agg(pl.col('value').sum().alias('uo_step'))
    
    first_abx = data_dict['abx'].group_by('stay_id').agg(pl.col('starttime').min().alias('first_abx_time'))
    
    grid = grid.join(fluid_step, on=['stay_id', 'timestep_idx'], how='left')
    grid = grid.join(fluid_total_agg.rename({'comp_bin': 'timestep_idx'}), on=['stay_id', 'timestep_idx'], how='left')
    grid = grid.join(vaso_agg, on=['stay_id', 'timestep_idx'], how='left')
    grid = grid.join(abx_agg, on=['stay_id', 'timestep_idx'], how='left')
    grid = grid.join(uo_step, on=['stay_id', 'timestep_idx'], how='left')
    grid = grid.join(first_abx, on='stay_id', how='left')
    
    grid = grid.sort(['stay_id', 'timestep_idx'])
    grid = grid.with_columns([
        pl.col('fluid_step').fill_null(0),
        pl.col('amount_ended').fill_null(0).cum_sum().over('stay_id').alias('fluid_total'),
        pl.col('uo_step').fill_null(0),
        pl.col('uo_step').fill_null(0).cum_sum().over('stay_id').alias('uo_total'),
        pl.col('vaso_median').fill_null(0.0),
        pl.col('vaso_max').fill_null(0.0),
        pl.col('abx_given').fill_null(0),
        pl.col('num_abx').fill_null(0)
    ])
    
    grid = grid.with_columns([
        (pl.col('fluid_total') - pl.col('uo_total')).alias('balance'),
        pl.when(pl.col('first_abx_time').is_not_null()).then((pl.col('window_end') - pl.col('first_abx_time')) / 3600).otherwise(None).alias('hours_since_first_abx')
    ])
    
    grid = grid.drop(['amount_ended', 'first_abx_time', 'window_end', 'timestep_idx', 'num_timesteps', 'first_time'])
    grid = grid.join(data_dict['demog'], on='stay_id', how='left')
    
    return grid

def fixgaps(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    nan_mask = np.isnan(x)
    valid_indices = np.arange(len(x))[~nan_mask]
    
    if len(valid_indices) == 0:
        return y
        
    nan_mask[:valid_indices[0]] = False
    nan_mask[valid_indices[-1]+1:] = False
    
    y[nan_mask] = interp1d(
        valid_indices,
        x[valid_indices]
    )(np.arange(len(x))[nan_mask])
    
    return y

def handle_missing_values(df, missing_threshold=0.8, chunk_size=5000):
    excluded_cols = [
        'timestep', 'stay_id', 'timestamp', 'gender', 'age', 
        'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 'fluid_total', 'fluid_step',
        'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]
    
    measurement_cols = [col for col in df.columns if col not in excluded_cols]
    
    total_len = len(df)
    if total_len == 0: return df
    
    miss_stats_meas = {col: df[col].null_count() / total_len for col in measurement_cols}
    cols_to_keep = [col for col, miss in miss_stats_meas.items() if miss < missing_threshold]
    final_cols = [c for c in df.columns if c in excluded_cols] + cols_to_keep
    df = df.select(final_cols)
    
    low_missing_cols = [col for col in cols_to_keep if 0 < miss_stats_meas[col] < 0.05]
    for col in low_missing_cols:
        arr = fixgaps(df[col].to_numpy())
        df = df.with_columns(pl.Series(col, arr))
        
    cols_for_knn = [col for col in cols_to_keep if col not in low_missing_cols]
    if cols_for_knn:
        # FIX: Directly convert to NumPy and enforce .copy() to make it writable
        ref = df.select(cols_for_knn).to_numpy().copy()
        imputer = KNNImputer(n_neighbors=1)
        
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            ref[i:chunk_end] = imputer.fit_transform(ref[i:chunk_end])
            
        df = df.with_columns([
            pl.Series(name, ref[:, j]) for j, name in enumerate(cols_for_knn)
        ])
        
    return df

def calculate_derived_variables(df):
    df = df.with_columns([
        (pl.col('gender') - 1).alias('gender'),
        pl.when(pl.col('age') > 150).then(91.4).otherwise(pl.col('age')).alias('age'),
        pl.when(pl.col('mechvent').fill_null(0) > 0).then(1).otherwise(0).alias('mechvent')
    ])
    
    median_cci = df['charlson_comorbidity_index'].median()
    mean_shock = (df['heart_rate'] / df['sbp_arterial']).mean()
    
    df = df.with_columns([
        pl.col('charlson_comorbidity_index').fill_null(median_cci),
        pl.col('vaso_median').fill_null(0),
        pl.col('vaso_max').fill_null(0),
        (pl.col('arterial_o2_pressure') / (pl.col('fio2') / 100)).alias('pf_ratio'),
    ])
    
    shock_idx = pl.col('heart_rate') / pl.col('sbp_arterial')
    df = df.with_columns(
        pl.when(shock_idx.is_infinite() | shock_idx.is_null()).then(mean_shock).otherwise(shock_idx).alias('shock_index')
    )

    sofa_resp = (pl.when(pl.col('pf_ratio').is_null()).then(0)
                 .when(pl.col('pf_ratio') >= 400).then(0)
                 .when(pl.col('pf_ratio') >= 300).then(1)
                 .when(pl.col('pf_ratio') >= 200).then(2)
                 .when(pl.col('pf_ratio') >= 100).then(3)
                 .otherwise(4))
    sofa_coag = (pl.when(pl.col('platelets').is_null()).then(0)
                 .when(pl.col('platelets') >= 150).then(0)
                 .when(pl.col('platelets') >= 100).then(1)
                 .when(pl.col('platelets') >= 50).then(2)
                 .when(pl.col('platelets') >= 20).then(3)
                 .otherwise(4))
    sofa_liver = (pl.when(pl.col('bilirubin_total').is_null()).then(0)
                  .when(pl.col('bilirubin_total') < 1.2).then(0)
                  .when(pl.col('bilirubin_total') < 2.0).then(1)
                  .when(pl.col('bilirubin_total') < 6.0).then(2)
                  .when(pl.col('bilirubin_total') < 12.0).then(3)
                  .otherwise(4))
    sofa_cv = (pl.when(pl.col('map').is_null() & pl.col('vaso_max').is_null()).then(0)
               .when(pl.col('vaso_max').is_not_null() & (pl.col('vaso_max') > 0.1)).then(4)
               .when(pl.col('vaso_max').is_not_null() & (pl.col('vaso_max') <= 0.1) & (pl.col('vaso_max') > 0)).then(3)
               .when(pl.col('map').is_not_null() & (pl.col('map') < 65)).then(2)
               .when(pl.col('map').is_not_null() & (pl.col('map') >= 65) & (pl.col('map') < 70)).then(1)
               .otherwise(0))
    sofa_cns = (pl.when(pl.col('gcs').is_null() | (pl.col('gcs') > 14)).then(0)
                .when(pl.col('gcs') > 12).then(1)
                .when(pl.col('gcs') > 9).then(2)
                .when(pl.col('gcs') > 5).then(3)
                .otherwise(4))
    sofa_renal = (pl.when(pl.col('creatinine').is_null() & pl.col('uo_step').is_null()).then(0)
                  .when(pl.col('creatinine').is_not_null() & (pl.col('creatinine') >= 5.0)).then(4)
                  .when(pl.col('creatinine').is_not_null() & (pl.col('creatinine') >= 3.5)).then(3)
                  .when(pl.col('creatinine').is_not_null() & (pl.col('creatinine') >= 2.0)).then(2)
                  .when(pl.col('creatinine').is_not_null() & (pl.col('creatinine') >= 1.2)).then(1)
                  .when(pl.col('creatinine').is_not_null() & (pl.col('creatinine') < 1.2)).then(0)
                  .when(pl.col('uo_step') < 34).then(4)
                  .when(pl.col('uo_step') < 84).then(3)
                  .otherwise(0))

    df = df.with_columns([
        sofa_resp.alias('sofa_resp'), sofa_coag.alias('sofa_coag'),
        sofa_liver.alias('sofa_liver'), sofa_cv.alias('sofa_cv'),
        sofa_cns.alias('sofa_cns'), sofa_renal.alias('sofa_renal')
    ])
    
    df = df.with_columns(
        (pl.col('sofa_resp') + pl.col('sofa_coag') + pl.col('sofa_liver') + 
         pl.col('sofa_cv') + pl.col('sofa_cns') + pl.col('sofa_renal')).alias('sofa_score')
    )

    sirs_temp = pl.when(pl.col('temp_C').is_not_null() & ((pl.col('temp_C') >= 38) | (pl.col('temp_C') <= 36))).then(1).otherwise(0)
    sirs_hr = pl.when(pl.col('heart_rate').is_not_null() & (pl.col('heart_rate') > 90)).then(1).otherwise(0)
    sirs_resp = pl.when((pl.col('respiratory_rate').is_not_null() & (pl.col('respiratory_rate') >= 20)) | 
                        (pl.col('arterial_co2_pressure').is_not_null() & (pl.col('arterial_co2_pressure') <= 32))).then(1).otherwise(0)
    sirs_wbc = pl.when(pl.col('wbc').is_not_null() & ((pl.col('wbc') >= 12) | (pl.col('wbc') < 4))).then(1).otherwise(0)
    
    df = df.with_columns((sirs_temp + sirs_hr + sirs_resp + sirs_wbc).alias('sirs_score'))
    return df

def apply_exclusion_criteria(df):
    extreme_uo_stays = df.filter(pl.col('uo_step') > 12000).select('stay_id').unique()
    # Anti-join: Keep rows in df that DO NOT exist in extreme_uo_stays
    df = df.join(extreme_uo_stays, on='stay_id', how='anti')
    
    extreme_fluid_stays = df.filter(pl.col('fluid_step') > 10000).select('stay_id').unique()
    df = df.join(extreme_fluid_stays, on='stay_id', how='anti')
    
    if len(df) == 0: return df
    
    patient_stats = df.group_by('stay_id').agg([
        pl.col('timestamp').min().alias('start_time'),
        pl.col('timestamp').max().alias('end_time'),
        pl.col('morta_hosp').first().alias('morta')
    ])
    early_death_stays = patient_stats.filter(
        (pl.col('morta') == 1) & ((pl.col('end_time') - pl.col('start_time')) / 3600 <= 24)
    ).select('stay_id')
    df = df.join(early_death_stays, on='stay_id', how='anti')

    sepsis_stays = df.filter(pl.col('sofa_score') >= 2).select('stay_id').unique()
    # Semi-join: Keep rows in df that DO exist in sepsis_stays
    df = df.join(sepsis_stays, on='stay_id', how='semi')
    
    return df

def add_sepsis_flag(df):
    if len(df) == 0: return df
    df = df.sort(['stay_id', 'timestamp'])
    
    df = df.with_columns(
        pl.when(pl.col('sofa_score') >= 2).then(1).otherwise(0).alias('sepsis_trigger')
    ).with_columns(
        pl.col('sepsis_trigger').cum_max().over('stay_id').alias('sepsis_cum_max')
    ).with_columns(
        pl.col('sepsis_cum_max').cum_sum().over('stay_id').alias('sepsis_phase')
    ).with_columns(
        pl.when(pl.col('sepsis_phase') == 0).then(0)
          .when(pl.col('sepsis_phase') == 1).then(1)
          .otherwise(2).alias('sepsis')
    ).drop(['sepsis_trigger', 'sepsis_cum_max', 'sepsis_phase'])
    return df

def add_septic_shock_flag(df, fluid_window, timestep):
    if len(df) == 0: return df
    WINDOW_STEPS = max(1, fluid_window // timestep)  
    MIN_FLUID_THRESHOLD = 2000  
    MAP_THRESHOLD = 65
    LACTATE_THRESHOLD = 2 
    
    df = df.sort(['stay_id', 'timestamp'])
    df = df.with_columns(
        # FIX: Renamed min_periods to min_samples
        pl.col('fluid_step').rolling_sum(window_size=WINDOW_STEPS, min_samples=1).over('stay_id').alias('rolling_fluid')
    )
    
    shock_condition = (
        (pl.col('rolling_fluid') >= MIN_FLUID_THRESHOLD) &
        (pl.col('map') < MAP_THRESHOLD) &
        (pl.col('lactic_acid') > LACTATE_THRESHOLD)
    )
    
    df = df.with_columns(
        shock_condition.cast(pl.Int32).alias('shock_trigger')
    ).with_columns(
        pl.col('shock_trigger').cum_max().over('stay_id').alias('shock_cum_max')
    ).with_columns(
        pl.col('shock_cum_max').cum_sum().over('stay_id').alias('shock_phase')
    ).with_columns(
        pl.when(pl.col('shock_phase') == 0).then(0)
          .when(pl.col('shock_phase') == 1).then(1)
          .otherwise(2).alias('septic_shock')
    ).drop(['rolling_fluid', 'shock_trigger', 'shock_cum_max', 'shock_phase'])
    return df

def main():
    args = parse_args()
    data = load_processed_files()
    measurements, code_to_concept, hold_times = load_measurement_mappings()
    onset = data['onset']
    
    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, seed=42)
        
    unique_stays = onset['stay_id'].unique().to_list()
    chunk_size = args.patient_chunk_size
    
    output_path = f"{args.output_dir}/patient_timeseries_v4.csv"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Remove existing file if present so we can start fresh and append
    if os.path.exists(output_path):
        os.remove(output_path)

    total_chunks = (len(unique_stays) - 1) // chunk_size + 1
    print(f"\n--- Starting Batched Processing ---")
    print(f"Total Patients: {len(unique_stays)}")
    print(f"Batch Size: {chunk_size} patients per cycle")
    print(f"Total Batches: {total_chunks}")
    print("-" * 35)
    
    for i in range(0, len(unique_stays), chunk_size):
        chunk_stays = unique_stays[i:i + chunk_size]
        batch_num = (i // chunk_size) + 1
        print(f"Processing Batch {batch_num}/{total_chunks} ({len(chunk_stays)} patients)...")
        
        # 1. Slice all data dictionaries strictly to the current patient batch
        chunk_data = {
            k: (v.filter(pl.col('stay_id').is_in(chunk_stays)) if 'stay_id' in v.columns else v)
            for k, v in data.items()
        }
        chunk_onset = chunk_data['onset']
        
        # 2. Execute Pipeline
        raw_pivoted, bounds = build_raw_measurements(chunk_data, chunk_onset, code_to_concept, args.window_before, args.window_after)
        
        if raw_pivoted.height == 0:
            continue
            
        raw_pivoted = handle_outliers(raw_pivoted)
        raw_pivoted = estimate_gcs_from_rass(raw_pivoted)
        raw_pivoted = estimate_fio2(raw_pivoted)
        raw_pivoted = handle_unit_conversions(raw_pivoted)
        raw_pivoted = sample_and_hold(raw_pivoted, hold_times) 

        init_traj = standardize_patient_trajectories(raw_pivoted, bounds, chunk_data, timestep=args.timestep)
        init_traj = handle_missing_values(init_traj, args.missing_threshold, chunk_size=args.knn_chunk_size)
        init_traj = calculate_derived_variables(init_traj)    
        init_traj = apply_exclusion_criteria(init_traj)
        
        if init_traj.height > 0:
            init_traj = add_septic_shock_flag(init_traj, fluid_window=args.fluid_window, timestep=args.timestep)
            init_traj = add_sepsis_flag(init_traj)

            # 3. Append to Disk Immediately
            with open(output_path, 'a') as f:
                # Include headers only on the very first batch
                init_traj.write_csv(f, include_header=(batch_num == 1))
        
        # 4. EXTREME GARBAGE COLLECTION 
        # Delete large variables explicitly to forcefully clear RAM before the next iteration
        del chunk_data, chunk_onset, raw_pivoted, bounds, init_traj
        gc.collect()

    print(f"\n✅ Processing Complete! Saved batch-processed data to {output_path}")

if __name__ == "__main__":
    main()