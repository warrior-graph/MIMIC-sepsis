import argparse
import json
import numpy as np
import pandas as pd
import pyprind
import os
import concurrent.futures
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer

import math
import warnings 

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.string_storage = "pyarrow"

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
    # TODO: double check the unit to be ml
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
    # Multiprocessing option
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of CPU cores to use for multiprocessing (default: 1)")
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
        data[key] = pd.read_csv(f'processed_files/{filename}', sep='|', engine="pyarrow", dtype_backend="pyarrow")
        
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
    
    # Create hold times mapping, used for sample and hold
    hold_times = {}
    for concept, info in measurements.items():
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']
            
    return measurements, code_to_concept, hold_times

def process_patient_measurements(data, measurements, code_to_concept, icustayid, onset_time, winb4=24, winaft=72):
    """Process data for a single patient"""
    
    # Helper function for fast O(1) index lookup
    def get_patient(df, stay_id):
        if stay_id in df.index:
            res = df.loc[stay_id]
            # Ensure it always returns a DataFrame, even if there's only 1 row
            return res.to_frame().T.copy() if isinstance(res, pd.Series) else res.copy()
        return pd.DataFrame(columns=df.columns)

    # Get relevant data for this patient instantly
    temp = get_patient(data['ce'], icustayid)
    temp2 = get_patient(data['labU'], icustayid)
    temp3 = get_patient(data['MV'], icustayid)
    
    # Filter for time window
    time_window = lambda df: (df['charttime'] >= onset_time - winb4*3600) & \
                           (df['charttime'] < onset_time + winaft*3600)
    
    temp = temp[time_window(temp)].copy()
    temp2 = temp2[time_window(temp2)].copy()
    temp3 = temp3[time_window(temp3)].copy()
    
    if temp.empty and temp2.empty and temp3.empty:
        return None
        
    # Process chartevents
    df_ce = pd.DataFrame()
    if not temp.empty:
        temp['concept'] = temp['itemid'].astype(int).astype(str).map(code_to_concept)
        temp = temp.dropna(subset=['concept'])
        if not temp.empty:
            df_ce = temp.pivot_table(index='charttime', columns='concept', values='valuenum', aggfunc='last')
            
    # Process lab values
    df_lab = pd.DataFrame()
    if not temp2.empty:
        temp2['concept'] = temp2['itemid'].astype(int).astype(str).map(code_to_concept)
        temp2 = temp2.dropna(subset=['concept'])
        if not temp2.empty:
            df_lab = temp2.pivot_table(index='charttime', columns='concept', values='valuenum', aggfunc='last')
            
    # Process mechanical ventilation
    df_mv = pd.DataFrame()
    if not temp3.empty:
        df_mv = temp3.groupby('charttime')['mechvent'].last().to_frame()
        
    patient_df = pd.concat([df_ce, df_lab, df_mv], axis=1).reset_index()
    if patient_df.empty:
        return None
        
    patient_df.rename(columns={'index': 'charttime'}, inplace=True)
    patient_df['stay_id'] = icustayid
    
    return patient_df

def handle_outliers(df):
    """Handle outliers in the patient timeseries data based on clinical thresholds
    """
    print('Handling outliers in patient timeseries data')
    
    # Weight
    df.loc[df['weight_kg'] > 300, 'weight_kg'] = np.nan
    df.loc[df['weight_lb'] > 660, 'weight_lb'] = np.nan
    
    # Heart Rate
    df.loc[df['heart_rate'] > 250, 'heart_rate'] = np.nan
    
    # Blood Pressure
    df.loc[df['sbp_arterial'] > 300, 'sbp_arterial'] = np.nan
    df.loc[df['map'] < 0, 'map'] = np.nan
    df.loc[df['map'] > 200, 'map'] = np.nan
    df.loc[df['dbp_arterial'] < 0, 'dbp_arterial'] = np.nan
    df.loc[df['dbp_arterial'] > 200, 'dbp_arterial'] = np.nan
    
    # Respiratory Rate
    df.loc[df['respiratory_rate'] > 80, 'respiratory_rate'] = np.nan
    
    # SpO2
    df.loc[df['spo2'] > 150, 'spo2'] = np.nan
    df.loc[df['spo2'] > 100, 'spo2'] = 100
    
    # Temperature
    mask = (df['temp_C'] > 90) & (df['temp_F'].isna())
    df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
    df.loc[df['temp_C'] > 90, 'temp_C'] = np.nan
    
    # FiO2
    df.loc[df['fio2'] > 100, 'fio2'] = np.nan
    df.loc[df['fio2'] < 1, 'fio2'] *= 100
    df.loc[df['fio2'] < 20, 'fio2'] = np.nan
    
    # O2 Flow
    df.loc[df['oxygen_flow'] > 70, 'oxygen_flow'] = np.nan
    
    # PEEP
    df.loc[df['peep'] < 0, 'peep'] = np.nan
    df.loc[df['peep'] > 40, 'peep'] = np.nan
    
    # Tidal Volume
    df.loc[df['tidal_volume'] > 1800, 'tidal_volume'] = np.nan
    
    # Minute Ventilation
    df.loc[df['minute_volume'] > 50, 'minute_volume'] = np.nan
    
    # Lab Values
    df.loc[df['potassium'] < 1, 'potassium'] = np.nan
    df.loc[df['potassium'] > 15, 'potassium'] = np.nan
    
    df.loc[df['sodium'] < 95, 'sodium'] = np.nan
    df.loc[df['sodium'] > 178, 'sodium'] = np.nan
    
    df.loc[df['chloride'] < 70, 'chloride'] = np.nan
    df.loc[df['chloride'] > 150, 'chloride'] = np.nan
    
    df.loc[df['glucose'] < 1, 'glucose'] = np.nan
    df.loc[df['glucose'] > 1000, 'glucose'] = np.nan
    
    df.loc[df['creatinine'] > 150, 'creatinine'] = np.nan
    df.loc[df['magnesium'] > 10, 'magnesium'] = np.nan
    df.loc[df['calcium_total'] > 20, 'calcium_total'] = np.nan
    df.loc[df['calcium_ionized'] > 5, 'calcium_ionized'] = np.nan

    df.loc[df['total_co2'] > 120, 'total_co2'] = np.nan
    
    df.loc[df['ast'] > 10000, 'ast'] = np.nan
    df.loc[df['alt'] > 10000, 'alt'] = np.nan
    
    df.loc[df['hemoglobin'] > 20, 'hemoglobin'] = np.nan
    df.loc[df['hematocrit'] > 65, 'hematocrit'] = np.nan
    df.loc[df['wbc'] > 500, 'wbc'] = np.nan
    df.loc[df['platelets'] > 2000, 'platelets'] = np.nan
    
    df.loc[df['inr'] > 20, 'inr'] = np.nan
    
    df.loc[df['ph_arterial'] < 6.7, 'ph_arterial'] = np.nan
    df.loc[df['ph_arterial'] > 8, 'ph_arterial'] = np.nan
    
    df.loc[df['arterial_o2_pressure'] > 700, 'arterial_o2_pressure'] = np.nan
    df.loc[df['arterial_co2_pressure'] > 200, 'arterial_co2_pressure'] = np.nan
    df.loc[df['arterial_base_excess'] < -50, 'arterial_base_excess'] = np.nan
    df.loc[df['lactic_acid'] > 30, 'lactic_acid'] = np.nan
    
    df.loc[df['bilirubin_total'] > 30, 'bilirubin_total'] = np.nan
    
    return df

def estimate_gcs_from_rass(df):
    """
    Estimate Glasgow Coma Scale (GCS) from Richmond Agitation-Sedation Scale (RASS)
    Based on data from Wesley JAMA 2003
    
    If GCS column doesn't exist, creates it and initializes with NaN values.
    """
    # Create GCS column if it doesn't exist
    if 'gcs' not in df.columns:
        df['gcs'] = np.nan
    
    # RASS +4 (Combative) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 4), 'gcs'] = 15
    
    # RASS +3 (Pulls tubes) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 3), 'gcs'] = 15
    
    # RASS +2 (Fights ventilator) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 2), 'gcs'] = 15
    
    # RASS +1 (Anxious) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 1), 'gcs'] = 15
    
    # RASS 0 (Alert and calm) -> GCS = 15
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == 0), 'gcs'] = 15
    
    # RASS -1 (Awakens to voice >10s) -> GCS = 14
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -1), 'gcs'] = 14
    
    # RASS -2 (Light sedation, awakens <10s) -> GCS = 12
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -2), 'gcs'] = 12
    
    # RASS -3 (Moderate sedation) -> GCS = 11
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -3), 'gcs'] = 11
    
    # RASS -4 (Deep sedation) -> GCS = 6
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -4), 'gcs'] = 6
    
    # RASS -5 (Unarousable) -> GCS = 3
    df.loc[(df['gcs'].isna()) & (df['richmond_ras'] == -5), 'gcs'] = 3
    
    return df

def estimate_fio2(df):
    """
    Estimate FiO2 values based on O2 flow rate. FiO2 is a critical measurement needed to
    Calculate the P/F ratio (PaO2/FiO2) which is used in SOFA scoring
    
    Args:
        df: pandas DataFrame containing patient data with columns:
            - oxygen_flow_device: type of oxygen delivery device (numeric codes)
            - oxygen_flow, oxygen_flow_cannula_rate, oxygen_flow_rate: different flow measurements
            - fio2: Fraction of inspired oxygen
    Returns:
        DataFrame with estimated fio2 values
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Combine all oxygen flow measurements into a single column
    # Taking the first non-null value from any of the flow measurements, (horizontal fill)
    flow_columns = ['oxygen_flow', 'oxygen_flow_cannula_rate', 'oxygen_flow_rate']
    df['combined_o2_flow'] = df[flow_columns].bfill(axis=1).iloc[:, 0]
    
    # Helper function to set FiO2 based on O2 flow thresholds
    def set_fio2_by_flow(mask, flow_thresholds, fio2_values):
        df_subset = df[mask].copy()
        df_subset['fio2'] = None 
        for threshold, fio2 in zip(flow_thresholds, fio2_values):
            flow_mask = df_subset['combined_o2_flow'] <= threshold
            df_subset.loc[flow_mask, 'fio2'] = fio2
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Case 1: No FiO2, Yes O2 flow, No interface or nasal cannula
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'].isin(['0', '2']))  # None or Nasal cannula
    
    if mask.any():
        flow_thresholds = [15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
        fio2_values = [70, 62, 55, 50, 44, 40, 36, 32, 28, 24]
        set_fio2_by_flow(mask, flow_thresholds, fio2_values)

    # Case 2: No FiO2, No O2 flow, No interface or nasal cannula
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].isna()) & \
           (df['oxygen_flow_device'].isin(['0', '2']))  # None or Nasal cannula
    df.loc[mask, 'fio2'] = 21  # Room air

    # Case 3: No FiO2, Yes O2 flow, Face mask or similar devices
    face_mask_types = ['3', '4', '5', '6', '8', '9', '10', '11', '12']  # Face tent through T-piece
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'].isin(face_mask_types))
    
    if mask.any():
        flow_thresholds = [15, 12, 10, 8, 6, 4]
        fio2_values = [75, 69, 66, 58, 40, 36]
        set_fio2_by_flow(mask, flow_thresholds, fio2_values)

    # Case 4: No FiO2, Yes O2 flow, Non-rebreather mask
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'] == '7')  # Non-rebreather
    
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        
        df_subset.loc[flow >= 15, 'fio2'] = 100
        df_subset.loc[(flow >= 10) & (flow < 15), 'fio2'] = 90
        df_subset.loc[(flow < 10) & (flow > 8), 'fio2'] = 80
        df_subset.loc[(flow <= 8) & (flow > 6), 'fio2'] = 70
        df_subset.loc[flow <= 6, 'fio2'] = 60
        
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Case 5: No FiO2, Yes O2 flow, CPAP/BiPAP mask
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'] == '13')  # CPAP/BiPAP mask
    
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        
        df_subset.loc[flow >= 15, 'fio2'] = 100
        df_subset.loc[(flow >= 10) & (flow < 15), 'fio2'] = 80
        df_subset.loc[flow < 10, 'fio2'] = 60
        
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Case 6: No FiO2, Yes O2 flow, Oxymizer
    mask = (df['fio2'].isna()) & \
           (df['combined_o2_flow'].notna()) & \
           (df['oxygen_flow_device'] == '14')  # Oxymizer
    
    if mask.any():
        df_subset = df[mask].copy()
        flow = df_subset['combined_o2_flow']
        
        df_subset.loc[flow >= 10, 'fio2'] = 80
        df_subset.loc[(flow >= 5) & (flow < 10), 'fio2'] = 60
        df_subset.loc[flow < 5, 'fio2'] = 40
        
        df.loc[mask, 'fio2'] = df_subset['fio2']

    # Clean up temporary column
    df = df.drop('combined_o2_flow', axis=1)

    return df

def handle_unit_conversions(df):
    """
    Handle various unit conversions and fix incorrectly recorded values:
    - Temperature (Celsius/Fahrenheit)
    - Hemoglobin/Hematocrit
    - Bilirubin (Total/Direct)
    """
    # Some values recorded in wrong column
    mask = (df['temp_F'] > 25) & (df['temp_F'] < 45)  # tempF close to 37deg
    if mask.any():
        df.loc[mask, 'temp_C'] = df.loc[mask, 'temp_F']
        df.loc[mask, 'temp_F'] = None

    # Values likely recorded in Fahrenheit but in Celsius column
    mask = df['temp_C'] > 70
    if mask.any():
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C']
        df.loc[mask, 'temp_C'] = None

    # Convert Celsius to Fahrenheit where missing
    mask = (~df['temp_C'].isna()) & (df['temp_F'].isna())
    if mask.any():
        df.loc[mask, 'temp_F'] = df.loc[mask, 'temp_C'] * 1.8 + 32

    # Convert Fahrenheit to Celsius where missing
    mask = (~df['temp_F'].isna()) & (df['temp_C'].isna())
    if mask.any():
        df.loc[mask, 'temp_C'] = (df.loc[mask, 'temp_F'] - 32) / 1.8

    # Handle Hemoglobin/Hematocrit conversions
    mask = (~df['hemoglobin'].isna()) & (df['hematocrit'].isna())
    if mask.any():
        df.loc[mask, 'hematocrit'] = (df.loc[mask, 'hemoglobin'] * 2.862) + 1.216

    mask = (~df['hematocrit'].isna()) & (df['hemoglobin'].isna())
    if mask.any():
        df.loc[mask, 'hemoglobin'] = (df.loc[mask, 'hematocrit'] - 1.216) / 2.862

    # Handle Bilirubin conversions
    mask = (~df['bilirubin_total'].isna()) & (df['bilirubin_direct'].isna())
    if mask.any():
        df.loc[mask, 'bilirubin_direct'] = (df.loc[mask, 'bilirubin_total'] * 0.6934) - 0.1752

    mask = (~df['bilirubin_direct'].isna()) & (df['bilirubin_total'].isna())
    if mask.any():
        df.loc[mask, 'bilirubin_total'] = (df.loc[mask, 'bilirubin_direct'] + 0.1752) / 0.6934

    return df

def sample_and_hold(df, vitalslab_hold):
    print('Performing sample and hold interpolation')
    
    # We only need the 1D arrays for time tracking
    stay_ids = df['stay_id'].to_numpy()
    charttimes = df['charttime'].to_numpy()
    
    cols_to_process = [col for col in vitalslab_hold if col in df.columns]
    bar = pyprind.ProgBar(len(cols_to_process), title='Processing columns')
    
    for col in cols_to_process:
        # MEMORY FIX: Use pandas-native type checking to support PyArrow engine
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Skipping non-numeric column: {col}")
            continue
            
        hold_period = vitalslab_hold[col] * 3600
        
        # Isolate just this column as a fast float array
        col_data = df[col].to_numpy(dtype='float64', na_value=np.nan, copy=True)
        
        last_charttime = 0
        last_value = np.nan
        current_stay_id = stay_ids[0]
        
        for i in range(len(col_data)):
            if stay_ids[i] != current_stay_id:
                last_charttime = 0
                last_value = np.nan
                current_stay_id = stay_ids[i]
            
            val = col_data[i]
            if not np.isnan(val):
                last_charttime = charttimes[i]
                last_value = val
            elif (charttimes[i] - last_charttime) <= hold_period and not np.isnan(last_value):
                col_data[i] = last_value
                
        # Assign optimized array back to the dataframe
        df[col] = col_data
        bar.update()
        
    return df

def combine_patient_data(patient_data, timestep=4, window_before=24, window_after=72):
    """
    Combines multiple measurement sources into a unified time series format with fixed timesteps.
    
    Args:
        patient_data: Dictionary containing patient measurements and metadata
        timestep: Size of timestep in hours (default: 4)
        window_before: Hours to include before onset time (default: up to 24)
        window_after: Hours to include after onset time (default: up to 72)
    """
    def process_antibiotics_data(start_time, end_time, abx_data):
        """Process antibiotics data within a time window"""
        # Create an explicit copy to avoid SettingWithCopyWarning
        abx_data = abx_data.copy()
        
        # Now it's safe to modify
        abx_data['stay_id'] = abx_data['stay_id'].astype('int64')
        
        # Get first antibiotics time (looking at all data, not just window)
        first_abx_time = abx_data['starttime'].min()
        
        # Find antibiotics active in the window
        mask = (abx_data['starttime'] <= end_time) & (abx_data['stoptime'] >= start_time)
        window_abx = abx_data[mask]
        
        return {
            'abx_given': 1 if len(window_abx) > 0 else 0,
            'hours_since_first_abx': (end_time - first_abx_time) / 3600 if first_abx_time else None,
            'num_abx': len(window_abx['drug'].unique()) if len(window_abx) > 0 else 0
        }
    
    def process_fluid_data(start_time, end_time, fluid_data):
        """Calculate fluid intake between two timepoints
        """
        if fluid_data is None or len(fluid_data) == 0:
            return 0, 0
        
        # For step fluids: entries that overlap with the window
        step_mask = (fluid_data['starttime'] < end_time) & (fluid_data['endtime'] >= start_time)
        step_fluids = fluid_data[step_mask]['amount'].sum()
        
        # For total fluids: entries that completed before the window end
        total_mask = fluid_data['endtime'] < end_time
        total_fluids = fluid_data[total_mask]['amount'].sum()
        
        return total_fluids, step_fluids

    def process_vasopressor_data(start_time, end_time, vaso_data):
        """Calculate vasopressor doses between two timepoints"""
        if vaso_data is None or len(vaso_data) == 0:
            return 0, 0
        
        # Find vasopressor entries that overlap with the window
        mask = (vaso_data['starttime'] <= end_time) & (vaso_data['endtime'] >= start_time)
        window_data = vaso_data[mask]
        
        if len(window_data) == 0:
            return 0, 0
        
        return window_data['rate_std'].median(), window_data['rate_std'].max()

    def process_urine_output(start_time, end_time, uo_data):
        """Calculate urine output between two timepoints"""
        if uo_data is None or len(uo_data) == 0:
            return 0, 0
        mask = (uo_data['charttime'] >= start_time) & (uo_data['charttime'] < end_time)
        step_uo = uo_data[mask]['value'].sum()  # Changed from 'urineoutput' to 'value'
        total_uo = uo_data[uo_data['charttime'] < end_time]['value'].sum()  # Changed from 'urineoutput' to 'value'
        return total_uo, step_uo
    
    def get_window_measurements(measurements, start_time, end_time):
        """Get all measurements within a time window"""
        if measurements is None or len(measurements) == 0:
            # For empty measurements, return dictionary with NaN values for all columns
            # and window midpoint as charttime
            dummy_row = {col: np.nan for col in measurements.columns if col not in ['stay_id']}
            dummy_row['charttime'] = (start_time + end_time) / 2
            return dummy_row
        
        mask = (measurements['charttime'] >= start_time) & (measurements['charttime'] < end_time)
        window_data = measurements[mask]
        
        if len(window_data) == 0:
            # If no data in this window, return NaN for all columns except charttime
            dummy_row = {col: np.nan for col in measurements.columns if col not in ['stay_id']}
            dummy_row['charttime'] = (start_time + end_time) / 2
            return dummy_row
        
        # Use mean for aggregation, but don't fill missing values
        return window_data.mean(axis=0, skipna=True).to_dict()
    
    # Initialize output dataframe with required columns
    columns = [
        'timestep', 'stay_id', 'timestamp',
        # Demographics columns
        'gender', 'age', 'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 
        # Clinical measurements
        *[col for col in patient_data['measurements'].columns if col not in ['timestep', 'stay_id', 'timestamp']],
        # Fluid balance
        'fluid_total', 'fluid_step', 'uo_total', 'uo_step', 'balance',
        # Vasopressors
        'vaso_median', 'vaso_max',
        # Antibiotics
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]
    
    processed_data = []
    
    # Get actual data time range for this patient
    patient_times = sorted(list(set(patient_data['measurements']['charttime'].values)))
    
    if not patient_times:  # Skip if no data available
        return None
    
    # Get first and last timestamp with valid data
    first_time = max(patient_times[0], patient_data['start_time'] - window_before * 3600)
    last_time = min(patient_times[-1], patient_data['start_time'] + window_after * 3600)
    
    # Calculate number of complete timesteps
    total_seconds = last_time - first_time
    total_hours = total_seconds / 3600
    num_timesteps = math.ceil(total_hours / timestep)
    
    # Process each timestep
    for timestep_idx in range(num_timesteps):
        # Calculate window boundaries in epoch time
        window_start = first_time + (timestep_idx * timestep * 3600)
        window_end = window_start + (timestep * 3600)
        
        # Skip if window is outside available data range
        if window_end < first_time or window_start > last_time:
            continue
        
        # Get measurements within time window
        measurements = get_window_measurements(patient_data['measurements'], window_start, window_end)
        
        if measurements is not None:
            # Process each data type
            fluid_total, fluid_step = process_fluid_data(window_start, window_end, patient_data['fluid'])
            vaso_median, vaso_max = process_vasopressor_data(window_start, window_end, patient_data['vasopressors'])
            uo_total, uo_step = process_urine_output(window_start, window_end, patient_data['urine_output'])
            abx_info = process_antibiotics_data(window_start, window_end, patient_data['antibiotics'])
            
            # Combine all data for this timestep
            timestep_data = {
                'timestep': timestep_idx + 1,
                'stay_id': patient_data['stay_id'],
                'timestamp': window_start,  # Keep as epoch time
                # Add demographics
                **patient_data['demographics'],
                # Add measurements
                **measurements,
                # Add fluid balance
                'fluid_total': fluid_total,
                'fluid_step': fluid_step,
                'uo_total': uo_total,
                'uo_step': uo_step,
                'balance': fluid_total - uo_total,
                'vaso_median': vaso_median,
                'vaso_max': vaso_max,
                **abx_info
            }
            processed_data.append(timestep_data)
    
    # IMPORTANT: Create DataFrame and explicitly set NaNs for missing values
    df = pd.DataFrame(processed_data, columns=columns)
    return df

# Wrapper function required for pickling in multiprocessing
def _process_patient_wrapper(args_tuple):
    (stay_id, patient_traj, start_time, demographics, 
     fluid, vasopressors, urine_output, antibiotics, 
     timestep, window_before, window_after) = args_tuple
    
    patient_data = {
        'stay_id': stay_id,
        'start_time': start_time,
        'measurements': patient_traj,
        'demographics': demographics,
        'fluid': fluid,
        'vasopressors': vasopressors,
        'urine_output': urine_output,
        'antibiotics': antibiotics
    }
    
    return combine_patient_data(
        patient_data, 
        timestep=timestep,
        window_before=window_before,
        window_after=window_after
    )

def standardize_patient_trajectories(init_traj, data_dict, timestep=4, window_before=24, window_after=72, workers=1):
    print(f'Processing all patients with fixed time windows (Workers: {workers})')
    all_patient_data = []
    
    # --- MEMORY FIX: Indexing instead of creating thousands of dict copies ---
    for key in ['fluid', 'vaso', 'UO', 'abx']:
        if data_dict[key].index.name != 'stay_id':
            # drop=False ensures 'stay_id' remains a usable column for downstream functions
            data_dict[key] = data_dict[key].set_index('stay_id', drop=False).sort_index()
            
    def get_patient_events(df, stay_id):
        if stay_id in df.index:
            res = df.loc[stay_id]
            return res.to_frame().T.copy() if isinstance(res, pd.Series) else res.copy()
        return pd.DataFrame(columns=df.columns)
    # --------------------------------------------------------------------------

    demog_indexed = data_dict['demog'].set_index('stay_id', drop=False)
    onset_indexed = data_dict['onset'].set_index('stay_id', drop=False)
    
    tasks = []
    
    # Group by stay_id to process each patient
    for stay_id, patient_traj in init_traj.groupby('stay_id'):
        
        # Use onset time from onset data instead of first measurement
        start_time = onset_indexed.loc[stay_id, 'onset_time']
        if isinstance(start_time, pd.Series):
            start_time = start_time.iloc[0]
            
        demographics = demog_indexed.loc[stay_id]
        if isinstance(demographics, pd.DataFrame):
            demographics = demographics.iloc[0]
        demographics = demographics.to_dict()
        
        # Use safe lookups instead of dictionary dict(tuple(...))
        fluid = get_patient_events(data_dict['fluid'], stay_id)
        vasopressors = get_patient_events(data_dict['vaso'], stay_id)
        urine_output = get_patient_events(data_dict['UO'], stay_id)
        antibiotics = get_patient_events(data_dict['abx'], stay_id)
        
        tasks.append((
            stay_id, patient_traj, start_time, demographics, 
            fluid, vasopressors, urine_output, antibiotics, 
            timestep, window_before, window_after
        ))
        
    bar = pyprind.ProgBar(len(tasks), title='Processing trajectories')
    
    # Process this patient's data using the existing combine_patient_measurements function
    if workers <= 1:
        for task in tasks:
            res = _process_patient_wrapper(task)
            if res is not None:
                all_patient_data.append(res)
            bar.update()
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_patient_wrapper, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    all_patient_data.append(res)
                bar.update()
    
    # Combine all processed patient data
    return pd.concat(all_patient_data, ignore_index=True)


def fixgaps(x: np.ndarray) -> np.ndarray:
    """Linearly interpolates gaps (NaN values) in a time series.
    
    Interpolates over NaN values in the input array, ignoring leading and trailing NaN values.
    The interpolation is done linearly between the nearest non-NaN values.
    
    Args:
        x: Input array containing the time series data with NaN values
        
    Returns:
        Array with NaN values interpolated, except for leading/trailing NaN values
    """
    # Make a copy to avoid modifying input
    y = np.copy(x)
    
    # Find NaN and non-NaN indices
    nan_mask = np.isnan(x)
    valid_indices = np.arange(len(x))[~nan_mask]
    
    if len(valid_indices) == 0:
        return y
        
    # Ignore leading/trailing NaN values
    nan_mask[:valid_indices[0]] = False
    nan_mask[valid_indices[-1]+1:] = False
    
    # Interpolate NaN values using valid data points
    y[nan_mask] = interp1d(
        valid_indices,
        x[valid_indices]
    )(np.arange(len(x))[nan_mask])
    
    return y

def handle_missing_values(df, missing_threshold=0.8):
    """Handle missing values through interpolation and KNN imputation
    
    Args:
        df: DataFrame containing patient measurements
        missing_threshold: Threshold for dropping columns with missing values (default: 0.7)
        
    Returns:
        DataFrame with missing values handled
    """
    print('Handling missing values...')
    
    # Get columns that need imputation (exclude non-numeric/demographic columns)
    measurement_cols = [col for col in df.columns if col not in [
        'timestep', 'stay_id', 'timestamp', 'gender', 'age', 
        'charlson_comorbidity_index', 're_admission', 'los',
        'morta_hosp', 'morta_90', 'fluid_total', 'fluid_step',
        'uo_total', 'uo_step', 'balance', 'vaso_median', 'vaso_max',
        'abx_given', 'hours_since_first_abx', 'num_abx'
    ]]
    
    # Print missingness statistics before imputation for all columns
    print("\nMissingness statistics before imputation:")
    print("-" * 50)
    
    # Get all numeric columns, including those we'll exclude
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Split into measurement cols and excluded cols
    excluded_numeric_cols = [col for col in all_numeric_cols if col not in measurement_cols]
    
    # Calculate missingness for both sets
    miss_stats_meas = df[measurement_cols].isna().sum() / len(df)
    miss_stats_excl = df[excluded_numeric_cols].isna().sum() / len(df)
    
    # Sort both by missingness
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
    print()
    
    # Calculate missingness per column
    miss = df[measurement_cols].isna().sum() / len(df)
    
    # Drop columns with missing values above threshold
    cols_to_keep = miss[miss < missing_threshold].index
    df = df[df.columns[~df.columns.isin(measurement_cols)].tolist() + cols_to_keep.tolist()]
    
# Fast Native C-based Linear interpolation for columns with <5% missing
    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    for col in low_missing_cols:
        df[col] = df[col].interpolate(method='linear', limit_area='inside')
        
    # KNN imputation for remaining missing values
    cols_for_knn = [c for c in cols_to_keep if c not in low_missing_cols]
    if cols_for_knn:
        ref = df[cols_for_knn].values
        
        # Initialize scikit-learn's KNNImputer
        imputer = KNNImputer(n_neighbors=1)
        
        # Process in chunks of 10K rows
        chunk_size = 9999
        bar = pyprind.ProgBar(len(range(0, len(df), chunk_size)))
        for i in range(0, len(df), chunk_size):
            chunk_end = min(i + chunk_size, len(df))
            
            # Fit and transform on the current chunk
            ref[i:chunk_end, :] = imputer.fit_transform(ref[i:chunk_end, :])
            bar.update()
            
        df[cols_for_knn] = ref
        
    return df

def calculate_derived_variables(df):
    """Calculate derived variables like P/F ratio, Shock Index, SOFA, and SIRS scores"""
    print('Computing derived variables: P/F ratio, Shock Index, SOFA, SIRS...')
    
    df = df.copy()
    
    # Fix demographic variables
    df['gender'] = df['gender'] - 1
    df.loc[df['age'] > 150, 'age'] = 91.4
    
    # Fix mechanical ventilation & Charlson
    df['mechvent'] = df['mechvent'].fillna(0)
    df.loc[df['mechvent'] > 0, 'mechvent'] = 1
    df['charlson_comorbidity_index'] = df['charlson_comorbidity_index'].fillna(df['charlson_comorbidity_index'].median())
    
    # Fix vasopressor doses
    df['vaso_median'] = df['vaso_median'].fillna(0)
    df['vaso_max'] = df['vaso_max'].fillna(0)
    
    # Calculate P/F ratio & Shock Index
    df['pf_ratio'] = df['arterial_o2_pressure'] / (df['fio2'] / 100)
    df['shock_index'] = df['heart_rate'] / df['sbp_arterial']
    df.loc[np.isinf(df['shock_index']), 'shock_index'] = np.nan
    df['shock_index'] = df['shock_index'].fillna(df['shock_index'].mean())
    
    # --- VECTORIZED SOFA SCORE CALCULATION ---
    
    # SOFA Respiratory
    pf = df['pf_ratio'].fillna(999)
    df['sofa_resp'] = np.where(pf < 100, 4, np.where(pf < 200, 3, np.where(pf < 300, 2, np.where(pf < 400, 1, 0))))
    df.loc[df['pf_ratio'].isna(), 'sofa_resp'] = 0
    
    # SOFA Coagulation
    plt = df['platelets'].fillna(999)
    df['sofa_coag'] = np.where(plt < 20, 4, np.where(plt < 50, 3, np.where(plt < 100, 2, np.where(plt < 150, 1, 0))))
    df.loc[df['platelets'].isna(), 'sofa_coag'] = 0
    
    # SOFA Liver
    bili = df['bilirubin_total'].fillna(0)
    df['sofa_liver'] = np.where(bili >= 12.0, 4, np.where(bili >= 6.0, 3, np.where(bili >= 2.0, 2, np.where(bili >= 1.2, 1, 0))))
    df.loc[df['bilirubin_total'].isna(), 'sofa_liver'] = 0
    
    # SOFA Cardiovascular
    map_ = df['map'].fillna(999)
    vaso = df['vaso_max']
    df['sofa_cv'] = np.where(vaso > 0.1, 4, np.where((vaso > 0) & (vaso <= 0.1), 3, np.where(map_ < 65, 2, np.where(map_ < 70, 1, 0))))
    
    # SOFA CNS
    gcs = df['gcs'].fillna(999)
    df['sofa_cns'] = np.where(gcs <= 5, 4, np.where(gcs <= 9, 3, np.where(gcs <= 12, 2, np.where(gcs <= 14, 1, 0))))
    df.loc[df['gcs'].isna(), 'sofa_cns'] = 0
    
    # SOFA Renal
    cr = df['creatinine'].fillna(0)
    uo = df['uo_step'].fillna(9999)
    cr_score = np.where(cr >= 5.0, 4, np.where(cr >= 3.5, 3, np.where(cr >= 2.0, 2, np.where(cr >= 1.2, 1, 0))))
    uo_score = np.where(uo < 34, 4, np.where(uo < 84, 3, 0))
    # Original logic: Use Cr if valid, otherwise fallback to UO
    df['sofa_renal'] = np.where(~df['creatinine'].isna(), cr_score, np.where(~df['uo_step'].isna(), uo_score, 0))
    
    # Total SOFA
    df['sofa_score'] = df['sofa_resp'] + df['sofa_coag'] + df['sofa_liver'] + df['sofa_cv'] + df['sofa_cns'] + df['sofa_renal']
    
    # --- VECTORIZED SIRS SCORE CALCULATION ---
    sirs = np.zeros(len(df))
    sirs += ((df['temp_C'] >= 38) | (df['temp_C'] <= 36)).astype(int)
    sirs += (df['heart_rate'] > 90).astype(int)
    sirs += ((df['respiratory_rate'] >= 20) | (df['arterial_co2_pressure'] <= 32)).astype(int)
    sirs += ((df['wbc'] >= 12) | (df['wbc'] < 4)).astype(int)
    df['sirs_score'] = sirs
    
    return df

def apply_exclusion_criteria(df):
    """Apply exclusion criteria for the sepsis cohort
    
    Excludes patients based on:
    1. Extreme urine output (>12000)
    2. Extreme fluid intake (>10000)
    3. Early deaths from possible withdrawals (death within 24h of ICU admission)
    4. None-sepsis patients (SOFA score < 2)
    """
    print('Applying exclusion criteria')
    
    # Keep track of excluded patients for logging
    initial_patients = len(df['stay_id'].unique())
    excluded_counts = {}
    
    # Exclude patients with extreme UO
    mask = df['uo_step'] > 12000
    excluded_stays = df[mask]['stay_id'].unique()
    df = df[~df['stay_id'].isin(excluded_stays)]
    excluded_counts['extreme_uo'] = len(excluded_stays)
    
    # Exclude patients with extreme fluid intake
    mask = df['fluid_step'] > 10000
    excluded_stays = df[mask]['stay_id'].unique()
    df = df[~df['stay_id'].isin(excluded_stays)]
    excluded_counts['extreme_fluid'] = len(excluded_stays)
    
    # Exclude early deaths (within 24h of ICU admission) - Vectorized
    # Group by stay_id and get first timestamp for each patient
    patient_starts = df.groupby('stay_id')['timestamp'].min()
    patient_ends = df.groupby('stay_id')['timestamp'].max()
    morta = df.groupby('stay_id')['morta_hosp'].first()
    time_to_death = (patient_ends - patient_starts) / 3600
    
    early_deaths = morta[(morta == 1) & (time_to_death <= 24)].index.tolist()
    df = df[~df['stay_id'].isin(early_deaths)]
    excluded_counts['early_death'] = len(early_deaths)

    # Exclude non-sepsis patients (SOFA score < 2)
    max_sofa = df.groupby('stay_id')['sofa_score'].max()
    non_sepsis_stays = max_sofa[max_sofa < 2].index.tolist()
    df = df[~df['stay_id'].isin(non_sepsis_stays)]
    excluded_counts['non_sepsis'] = len(non_sepsis_stays)
    
    # Print exclusion statistics
    final_patients = len(df['stay_id'].unique())
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
    """Add sepsis flag to patient trajectories.
    
    Sepsis criteria:
    1. Sepsis-3 definition:
        - Positive culture data or antibiotic administration
        - SOFA score ≥ 2, assuming all baseline SOFA is 0
    
    Flag values:
    0 = No sepsis
    1 = Sepsis onset identified
    2 = Censored (post first sepsis)
    
    Args:
        df: DataFrame containing patient measurements
    Returns:
        DataFrame with added sepsis column
    """
    print('Adding sepsis flags to trajectories')
    
    # Initialize sepsis column
    df['sepsis'] = 0
    
    # Process each patient / Sort by timestamp to ensure chronological order
    df = df.sort_values(['stay_id', 'timestamp'])
    
    # Check for SOFA ≥ 2
    sepsis_condition = df['sofa_score'] >= 2
    sepsis_cum = sepsis_condition.groupby(df['stay_id']).cumsum()
    
    # Find first occurrence of sepsis
    is_first_sepsis = sepsis_condition & (sepsis_cum == 1)
    is_after_sepsis = (~is_first_sepsis) & (sepsis_cum >= 1)
    
    # Mark sepsis onset
    df.loc[is_first_sepsis, 'sepsis'] = 1
    
    # Mark all subsequent timestamps for this patient as censored
    df.loc[is_after_sepsis, 'sepsis'] = 2
    
    # Print statistics
    total_patients = len(df['stay_id'].unique())
    sepsis_patients = len(df[df['sepsis'] == 1]['stay_id'].unique())
    
    print("\nSepsis Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing sepsis: {sepsis_patients} ({sepsis_patients/total_patients*100:.1f}%)")
    print(f"Timesteps with sepsis onset: {(df['sepsis'] == 1).sum()}")
    print(f"Censored timesteps: {(df['sepsis'] == 2).sum()}")
    print("-" * 50)
    
    return df

def add_septic_shock_flag(df):
    """Add septic shock flag to patient trajectories.
    
    Septic shock criteria:
    1. Adequate fluid resuscitation (defined as minimum fluid intake over previous 12 hours)
    2. After adequate fluids:
        - MAP < 65 mmHg AND
        - Requires vasopressors (indicated by hypotension despite fluids) AND
        - Lactate > 2 mmol/L
    
    Flag values:
    0 = No septic shock
    1 = Septic shock identified
    2 = Censored (post first shock)
    
    Args:
        df: DataFrame containing patient measurements
    Returns:
        DataFrame with added septic_shock column
    """
    print('Adding septic shock flags to trajectories')
    
    # Initialize septic shock column
    df['septic_shock'] = 0
    
    # Define thresholds
    FLUID_WINDOW = 12  # hours
    TIMESTEP_SIZE = 4  # hours per timestep
    WINDOW_STEPS = max(1, FLUID_WINDOW // TIMESTEP_SIZE)  # number of timesteps for 12 hours
    MIN_FLUID_THRESHOLD = 2000  # mL in 12 hours
    MAP_THRESHOLD = 65
    LACTATE_THRESHOLD = 2  # mmol/L
    
    print(f"Using rolling window of {WINDOW_STEPS} timesteps ({WINDOW_STEPS * TIMESTEP_SIZE} hours)")
    print(f"Minimum fluid threshold: {MIN_FLUID_THRESHOLD}mL over {FLUID_WINDOW} hours")
    
    # Process each patient / First sort by timestamp to ensure correct rolling calculation
    df = df.sort_values(['stay_id', 'timestamp'])
    
    # Calculate rolling fluid sum for previous 12 hours
    rolling_fluid = df.groupby('stay_id')['fluid_step'].rolling(window=WINDOW_STEPS, min_periods=1).sum().reset_index(level=0, drop=True)
    
    # Check conditions for septic shock
    shock_conditions = (
        (rolling_fluid >= MIN_FLUID_THRESHOLD) &  # adequate fluids
        (df['map'] < MAP_THRESHOLD) &     # hypotension
        (df['lactic_acid'] > LACTATE_THRESHOLD)  # elevated lactate
    )
    
    shock_cum = shock_conditions.groupby(df['stay_id']).cumsum()
    
    # Find first occurrence of shock
    is_first_shock = shock_conditions & (shock_cum == 1)
    is_after_shock = (~is_first_shock) & (shock_cum >= 1)
    
    # Mark shock onset
    df.loc[is_first_shock, 'septic_shock'] = 1
    
    # Mark all subsequent timestamps for this patient as censored
    df.loc[is_after_shock, 'septic_shock'] = 2
    
    # Print statistics
    total_patients = len(df['stay_id'].unique())
    shock_patients = len(df[df['septic_shock'] == 1]['stay_id'].unique())
    
    print("\nSeptic Shock Statistics:")
    print("-" * 50)
    print(f"Total patients: {total_patients}")
    print(f"Patients developing shock: {shock_patients} ({shock_patients/total_patients*100:.1f}%)")
    print(f"Timesteps with shock onset: {(df['septic_shock'] == 1).sum()}")
    print(f"Censored timesteps: {(df['septic_shock'] == 2).sum()}")
    print("-" * 50)
    
    return df

def main():
    args = parse_args()
    
    # Load all required data
    data = load_processed_files()
    measurements, code_to_concept, hold_times = load_measurement_mappings()
    
    # === ADICIONE ESTAS 4 LINHAS AQUI ===
    print("Pre-indexing massive tables to prevent loop bottleneck...")
    data['ce'] = data['ce'].set_index('stay_id').sort_index()
    data['labU'] = data['labU'].set_index('stay_id').sort_index()
    data['MV'] = data['MV'].set_index('stay_id').sort_index()
    # ====================================

    # Load onset data 
    onset = data['onset']
    
    # Sample subjects if sample_size is specified
    if args.sample_size is not None:
        print(f'Sampling {args.sample_size} subjects for testing')
        onset = onset.sample(n=args.sample_size, random_state=42)
    
    # Process each patient
    print('Processing patient timeseries data')
    all_patient_data = []
    
    # Check if processed file already exists
    output_path = f"{args.output_dir}/patient_timeseries_v4.csv"
    
    bar = pyprind.ProgBar(len(onset))
    for _, row in onset.iterrows():
        icustayid = row['stay_id']
        onset_time = row['onset_time']
        if onset_time > 0:  # if we have a flag time
            patient_df = process_patient_measurements(
                data, measurements, code_to_concept,
                icustayid, onset_time,
                winb4=args.window_before,
                winaft=args.window_after
            )
            if patient_df is not None:
                all_patient_data.append(patient_df)
        bar.update()
    # Combine all patient data
    init_traj = pd.concat(all_patient_data, ignore_index=True)
    
    # === MEMORY FIX: Force garbage collection of massive raw files ===
    print("Freeing gigabytes of raw data from memory...")
    del all_patient_data 
    del data['ce']
    del data['labU']
    del data['MV']
    import gc
    gc.collect()
    # =================================================================
    
    # Handle outliers
    init_traj = handle_outliers(init_traj)

    # Estimate GCS from RASS
    init_traj = estimate_gcs_from_rass(init_traj)
    
    # Estimate FiO2 from O2 flow
    init_traj = estimate_fio2(init_traj)

    # Handle unit conversions
    init_traj = handle_unit_conversions(init_traj)

    # Sample and hold (forward fill)
    init_traj = sample_and_hold(init_traj, hold_times) 

    # Combine all patients
    init_traj = standardize_patient_trajectories(
        init_traj, 
        data,
        timestep=args.timestep,
        window_before=args.window_before,
        window_after=args.window_after,
        workers=args.workers
    )

    # Handle missing values
    init_traj = handle_missing_values(init_traj, args.missing_threshold)

    # Final check before returning
    print(f"FiO2 zeros after handling missing values: {(init_traj['fio2'] == 0).sum()}")

    # Calculate derived variables (SOFA, SIRS, etc.)
    init_traj = calculate_derived_variables(init_traj)    
    
    # Apply exclusion criteria at the end
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
    
    # Save processed data
    from datetime import datetime

    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = f"{args.output_dir}/patient_timeseries_{current_time}.csv"
    init_traj = init_traj.sort_values(by=['stay_id', 'timestamp']).reset_index(drop=True)
    init_traj.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
if __name__ == "__main__":
    main()
