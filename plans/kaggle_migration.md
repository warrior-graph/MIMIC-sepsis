# Kaggle Migration Guide — MIMIC-Sepsis Benchmark

This guide explains how to run the **Stage 4 benchmark** on Kaggle using
`notebooks/kaggle_benchmark.ipynb`. Stages 1–3 (data extraction and trajectory
formatting) must have been completed **locally** or in a separate preprocessing
notebook, producing a `patient_timeseries_*.csv` file.

---

## Prerequisites

- Credentialed [PhysioNet](https://physionet.org/) access (to accept Kaggle's
  MIMIC-IV data-use agreement).
- A `patient_timeseries_*.csv` file already generated from Stage 3
  (`format_traj.py` / `format_traj_polars.py`).

---

## Step 1 — Upload the source code as a Kaggle dataset

The notebook imports modules (`benchmark`, `data_processor`, `lstm_model`, …)
directly from Python files. These must be available on the Kaggle filesystem.

1. Go to **Kaggle → Datasets → New Dataset**.
2. Upload **all `.py` files** from this project's `src/` directory.  
   (You can zip the folder or drag-and-drop the individual files.)
3. Name the dataset **`mimic-sepsis-src`** (the notebook uses this exact slug).
4. Set visibility to **Private** (recommended for MIMIC-derived code).
5. Click **Create**.

The files will be accessible at `/kaggle/input/mimic-sepsis-src/` inside the
notebook.

---

## Step 2 — Upload the processed timeseries CSV as a Kaggle dataset

1. Go to **Kaggle → Datasets → New Dataset**.
2. Upload your `patient_timeseries_*.csv` from `processed_files/`.
3. Name the dataset **`mimic-sepsis-data`**.
4. Set visibility to **Private** (the file is derived from MIMIC-IV; keep it
   private to comply with the PhysioNet data-use agreement).
5. Click **Create**.

The CSV will be accessible at `/kaggle/input/mimic-sepsis-data/` inside the
notebook.

---

## Step 3 — Create the benchmark notebook on Kaggle

1. Go to **Kaggle → Notebooks → New Notebook**.
2. In the notebook editor, click **File → Import Notebook** and upload
   `notebooks/kaggle_benchmark.ipynb` from this repository.
3. In the right-hand **Data** panel, click **Add data** and attach:
   - `mimic-sepsis-src` (your source-code dataset)
   - `mimic-sepsis-data` (your timeseries CSV dataset)
4. For GPU support (faster LSTM / Transformer training), go to
   **Settings → Accelerator → GPU T4 x2** (or P100).

---

## Step 4 — Configure and run

Open **Cell 3** of the notebook and adjust the variables:

| Variable | Description | Example |
|---|---|---|
| `DATA_PATH` | Full path to your timeseries CSV | `/kaggle/input/mimic-sepsis-data/patient_timeseries_2026-06-11.csv` |
| `TASK` | Prediction target | `septic_shock` |
| `MODEL_TYPE` | Model architecture | `transformer` |
| `HORIZON` | Steps ahead to predict (temporal tasks only) | `6` |
| `INCLUDE_TREATMENTS` | Include treatment variables as features | `False` |
| `BALANCE` | Enable window-level class balancing | `False` |

Then run all cells in order:

| Cell | Description |
|---|---|
| Cell 1 | Installs `pyprind`, `lightgbm`, `prophet`, `pytorch-forecasting` |
| Cell 2 | Sets `sys.path`, imports all modules, sets random seeds |
| Cell 3 | Configuration block (edit before running) |
| Cell 4 | Single `run_benchmark` call — trains and evaluates one model |
| Cell 5a | `run_selected_experiments` — all models for one task *(optional, uncomment)* |
| Cell 5b | `run_all_experiments` — full experiment grid *(optional, long runtime)* |
| Cell 6 | Bar charts: AUROC/AUPRC (classification) and RMSE/MAE (regression) |

---

## Output files

All outputs are written to `/kaggle/working/` (writable on Kaggle):

| File | Contents |
|---|---|
| `results/score_benchmark.csv` | Appended results from Cell 4 single runs |
| `benchmark_results.csv` | Results from `run_all_experiments` (Cell 5b) |
| `{task}_benchmark_results.csv` | Results from `run_selected_experiments` (Cell 5a) |
| `results/classif_benchmark.png` | AUROC / AUPRC bar chart (classification tasks) |
| `results/regress_benchmark.png` | RMSE / MAE bar chart (regression tasks) |

Download outputs via **Kaggle → Output** tab after the notebook run completes.

---

## Runtime estimates (Kaggle GPU T4)

| Model | Task | Approximate time |
|---|---|---|
| Linear | Any | < 1 min |
| XGBoost / LightGBM | Any | 1–5 min |
| LSTM | Classification | 3–10 min |
| Transformer | Classification | 5–15 min |
| `run_all` (all tasks × all models) | — | 2–4 hours |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'data_processor'`
The `mimic-sepsis-src` dataset is not attached, or Cell 2 has the wrong
`SRC_PATH`. Verify the dataset slug matches and that Cell 2 ran successfully.

### `FileNotFoundError` on the CSV
The `DATA_PATH` in Cell 3 is wrong. Check the exact filename via:
```python
import os
print(os.listdir("/kaggle/input/mimic-sepsis-data/"))
```

### Out-of-memory errors with LSTM or Transformer
Reduce `batch_size` in `run_benchmark` (default 32) or switch to a smaller
subset of the data using the `--sample_size` flag when regenerating the
timeseries CSV.

### `prophet` install takes too long
Prophet pulls in `pystan` which compiles C++ code on first install. This can
take 5–10 minutes. Cell 1 uses `-q` (quiet) mode; wait for it to complete.
