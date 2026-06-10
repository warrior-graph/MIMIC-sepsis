import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_raw", action='store_true', help="If specified, additionally save trajectories without normalized features")
    parser.add_argument("--save_intermediate", action='store_true', default=True, help="If specified, save off intermediate tables used to construct final patient table")
    return parser.parse_args()


def load_processed_files():
    print('Loading processed files created from database using "preprocess.py"')
    files = {
        'stay': 'icustays.csv',
        'abx': 'abx.csv',
        'culture': 'culture.csv',
        'microbio': 'microbio.csv',
        'demog': 'demog.csv',
        'ce': 'chartevents.csv',
        'MV': 'mechvent.csv',
        'inputpreadm': 'preadm_fluid.csv',
        'fluid': 'fluid.csv',
        'vaso': 'vaso.csv',
        'UO': 'uo.csv'
    }

    data = {}
    for key, filename in files.items():
        data[key] = pd.read_csv(f'processed_files/{filename}', sep='|')

    # Load and combine lab data
    labs_ce = pd.read_csv('processed_files/labs_ce.csv', sep='|')
    labs_le = pd.read_csv('processed_files/labs_le.csv', sep='|')
    labs_le.rename(columns={'timestp': 'charttime'}, inplace=True)
    data['labU'] = pd.concat([labs_ce, labs_le], sort=False, ignore_index=True)

    return data


def process_microbio_data(microbio, culture):
    microbio['charttime'] = microbio['charttime'].fillna(microbio['chartdate'])
    microbio.drop(columns=['chartdate'], inplace=True)
    return pd.concat([microbio, culture], sort=False, ignore_index=True)


def process_demog_data(demog):
    demog['morta_90'] = demog['morta_90'].fillna(0)
    demog['morta_hosp'] = demog['morta_hosp'].fillna(0)
    demog['charlson_comorbidity_index'] = demog['charlson_comorbidity_index'].fillna(0)
    return demog.drop_duplicates(subset=['admittime', 'dischtime'], keep='first')


def determine_readmission(demog, cutoff=3600*24*30):
    """Vectorized readmission determination."""
    demog = demog.sort_values(['subject_id', 'admittime']).copy()
    demog['re_admission'] = 0

    # Group by subject and check if admission is within cutoff of previous discharge
    demog['prev_dischtime'] = demog.groupby('subject_id')['dischtime'].shift(1)
    mask = demog['prev_dischtime'].notna() & ((demog['admittime'] - demog['prev_dischtime']) <= cutoff)
    demog.loc[mask, 're_admission'] = 1
    demog.drop(columns=['prev_dischtime'], inplace=True)
    return demog


def fill_missing_icustay_ids(bacterio, demog, abx):
    """Optimized filling of missing ICU stay IDs."""
    print('Filling-in missing ICUSTAY IDs in bacterio')

    # Precompute demog lookup arrays
    demog_subject = demog['subject_id'].values
    demog_hadm = demog['hadm_id'].values
    demog_intime = demog['intime'].values
    demog_outtime = demog['outtime'].values
    demog_stay = demog['stay_id'].values

    # --- Fill bacterio ---
    missing_mask = bacterio['stay_id'].isna()
    missing_idx = bacterio.index[missing_mask]

    if len(missing_idx) > 0:
        bact_charttime = bacterio.loc[missing_idx, 'charttime'].values
        bact_subject = bacterio.loc[missing_idx, 'subject_id'].values

        # Build a mapping from subject_id to demog row indices
        subject_to_demog = {}
        for idx_d in range(len(demog_subject)):
            sid = demog_subject[idx_d]
            if sid not in subject_to_demog:
                subject_to_demog[sid] = []
            subject_to_demog[sid].append(idx_d)

        new_stay_ids = np.full(len(missing_idx), np.nan)

        for k in range(len(missing_idx)):
            o = bact_charttime[k]
            subjectid = bact_subject[k]

            rows = subject_to_demog.get(subjectid, [])
            if len(rows) == 0:
                continue

            found = False
            for j in rows:
                if (o >= demog_intime[j] - 48*3600) and (o <= demog_outtime[j] + 48*3600):
                    new_stay_ids[k] = demog_stay[j]
                    found = True
                    break

            if not found and len(rows) == 1:
                new_stay_ids[k] = demog_stay[rows[0]]

        # Apply updates
        update_mask = ~np.isnan(new_stay_ids)
        bacterio.loc[missing_idx[update_mask], 'stay_id'] = new_stay_ids[update_mask]

    print('Filling-in missing ICUSTAY IDs in ABx')

    # Ensure stay_id column exists in abx
    if 'stay_id' not in abx.columns:
        abx['stay_id'] = np.nan

    # Build mapping from hadm_id to demog row indices
    hadm_to_demog = {}
    for idx_d in range(len(demog_hadm)):
        hid = demog_hadm[idx_d]
        if hid not in hadm_to_demog:
            hadm_to_demog[hid] = []
        hadm_to_demog[hid].append(idx_d)

    abx_starttime = abx['starttime'].values
    abx_hadm = abx['hadm_id'].values
    abx_stay = abx['stay_id'].values.copy().astype(float)

    for k in range(len(abx)):
        o = abx_starttime[k]
        hadmid = abx_hadm[k]

        rows = hadm_to_demog.get(hadmid, [])
        if len(rows) == 0:
            continue

        found = False
        for j in rows:
            if o >= demog_intime[j] - 48*3600 and o <= demog_outtime[j] + 48*3600:
                abx_stay[k] = demog_stay[j]
                found = True
                break

        if not found and len(rows) == 1:
            abx_stay[k] = demog_stay[rows[0]]

    abx['stay_id'] = abx_stay

    return bacterio, abx


def find_infection_onset(icustayidlist, abx, bacterio):
    """Optimized infection onset finding using groupby."""
    print('Finding presumed onset of infection according to sepsis3 guidelines')

    # Pre-group data by stay_id - only keep valid stay_ids
    abx_valid = abx.dropna(subset=['stay_id'])
    bact_valid = bacterio.dropna(subset=['stay_id'])

    abx_grouped = {}
    for stay_id, group in abx_valid.groupby('stay_id'):
        abx_grouped[stay_id] = group['starttime'].values

    bact_grouped = {}
    for stay_id, group in bact_valid.groupby('stay_id'):
        bact_grouped[stay_id] = {
            'charttime': group['charttime'].values,
            'subject_id': group['subject_id'].iloc[0]
        }

    onset_rows = []

    for icustayid in icustayidlist:
        ab_arr = abx_grouped.get(icustayid)
        bact_info = bact_grouped.get(icustayid)

        if ab_arr is None or bact_info is None:
            continue

        bact_arr = bact_info['charttime']
        subj_id = bact_info['subject_id']

        if len(ab_arr) == 0 or len(bact_arr) == 0:
            continue

        # Compute distance matrix in hours
        D = np.abs(ab_arr.reshape(-1, 1) - bact_arr.reshape(1, -1)) / 3600.0

        found = False
        for i in range(D.shape[0]):
            I = np.argmin(D[i, :])
            M = D[i, I]
            ab1 = ab_arr[i]
            bact1 = bact_arr[I]

            if M <= 24 and ab1 <= bact1:
                onset_rows.append({
                    'subject_id': subj_id,
                    'stay_id': icustayid,
                    'onset_time': ab1
                })
                found = True
                break
            elif M <= 72 and ab1 >= bact1:
                onset_rows.append({
                    'subject_id': subj_id,
                    'stay_id': icustayid,
                    'onset_time': bact1
                })
                found = True
                break

    print(f'Number of preliminary, presumed septic trajectories: {len(onset_rows)}')
    onset_df = pd.DataFrame(onset_rows)
    return onset_df


def main():
    args = parse_args()
    data = load_processed_files()

    # Process microbio data
    bacterio = process_microbio_data(data['microbio'], data['culture'])

    # Process demographics
    demog = process_demog_data(data['demog'])

    # Get list of ICU stay IDs
    icustayidlist = demog.stay_id.values.tolist()

    # Calculate readmissions (vectorized)
    demog = determine_readmission(demog)

    # Process fluid data
    data['fluid']['norm_rate_of_infusion'] = data['fluid']['tev'] * data['fluid']['rate'] / data['fluid']['amount']

    # Fill missing ICU stay IDs
    bacterio, data['abx'] = fill_missing_icustay_ids(bacterio, demog, data['abx'])

    # Find infection onset
    onset = find_infection_onset(icustayidlist, data['abx'], bacterio)

    # Save processed data if requested
    if args.save_intermediate:
        onset.to_csv('processed_files/onset.csv', sep='|', index=False)
        bacterio.to_csv('processed_files/bacterio_processed.csv', sep='|', index=False)
        demog.to_csv('processed_files/demog_processed.csv', sep='|', index=False)
        data['labU'].to_csv('processed_files/labu.csv', sep='|', index=False)
        data['abx'].to_csv('processed_files/abx_processed.csv', sep='|', index=False)

    return onset, bacterio, demog, data


if __name__ == "__main__":
    main()