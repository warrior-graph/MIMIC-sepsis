# Graph Report - .  (2026-06-13)

## Corpus Check
- 82 files · ~95,407 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 595 nodes · 956 edges · 49 communities (29 shown, 20 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 71 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Polars Trajectory Formatting|Polars Trajectory Formatting]]
- [[_COMMUNITY_GPU Trajectory Formatting|GPU Trajectory Formatting]]
- [[_COMMUNITY_Benchmark Pipeline Architecture|Benchmark Pipeline Architecture]]
- [[_COMMUNITY_Time Series Data Processor|Time Series Data Processor]]
- [[_COMMUNITY_Clinical Parameters & Figures|Clinical Parameters & Figures]]
- [[_COMMUNITY_Pandas Trajectory Formatting|Pandas Trajectory Formatting]]
- [[_COMMUNITY_Gradient Boosting Models|Gradient Boosting Models]]
- [[_COMMUNITY_Caveman Agent Skills|Caveman Agent Skills]]
- [[_COMMUNITY_Benchmark Evaluation Engine|Benchmark Evaluation Engine]]
- [[_COMMUNITY_Compress Validation (agents)|Compress Validation (agents)]]
- [[_COMMUNITY_Compress Validation (roo)|Compress Validation (roo)]]
- [[_COMMUNITY_Caveman Communication Tools|Caveman Communication Tools]]
- [[_COMMUNITY_Transformer Model|Transformer Model]]
- [[_COMMUNITY_Prophet Forecasting Model|Prophet Forecasting Model]]
- [[_COMMUNITY_Temporal Fusion Transformer|Temporal Fusion Transformer]]
- [[_COMMUNITY_Compress Core Scripts (agents)|Compress Core Scripts (agents)]]
- [[_COMMUNITY_LSTM Model|LSTM Model]]
- [[_COMMUNITY_Compress Core Scripts (roo)|Compress Core Scripts (roo)]]
- [[_COMMUNITY_Results Visualization|Results Visualization]]
- [[_COMMUNITY_File Type Detection (agents)|File Type Detection (agents)]]
- [[_COMMUNITY_File Type Detection (roo)|File Type Detection (roo)]]
- [[_COMMUNITY_ICU Data Preprocessing|ICU Data Preprocessing]]
- [[_COMMUNITY_Linear Time Series Model|Linear Time Series Model]]
- [[_COMMUNITY_Compress Benchmarking (agents)|Compress Benchmarking (agents)]]
- [[_COMMUNITY_Compress Benchmarking (roo)|Compress Benchmarking (roo)]]
- [[_COMMUNITY_Metrics Calculation|Metrics Calculation]]
- [[_COMMUNITY_Compress CLI (agents)|Compress CLI (agents)]]
- [[_COMMUNITY_Compress CLI (roo)|Compress CLI (roo)]]
- [[_COMMUNITY_Caveman Help Skill|Caveman Help Skill]]
- [[_COMMUNITY_Compress Package Init (agents)|Compress Package Init (agents)]]
- [[_COMMUNITY_Compress Package Init (roo)|Compress Package Init (roo)]]
- [[_COMMUNITY_Antibiotics SQL Query|Antibiotics SQL Query]]
- [[_COMMUNITY_Benchmark GPU Logging|Benchmark GPU Logging]]
- [[_COMMUNITY_Benchmark Reproducibility|Benchmark Reproducibility]]
- [[_COMMUNITY_Chart Events SQL Query|Chart Events SQL Query]]
- [[_COMMUNITY_Culture Events SQL Query|Culture Events SQL Query]]
- [[_COMMUNITY_Demographics SQL Query|Demographics SQL Query]]
- [[_COMMUNITY_Fluid SQL Query|Fluid SQL Query]]
- [[_COMMUNITY_ICU Stays SQL Query|ICU Stays SQL Query]]
- [[_COMMUNITY_Lab Chart Events SQL Query|Lab Chart Events SQL Query]]
- [[_COMMUNITY_Lab Events SQL Query|Lab Events SQL Query]]
- [[_COMMUNITY_Mechanical Ventilation SQL Query|Mechanical Ventilation SQL Query]]
- [[_COMMUNITY_Microbiology SQL Query|Microbiology SQL Query]]
- [[_COMMUNITY_Clinical Notes SQL Query|Clinical Notes SQL Query]]
- [[_COMMUNITY_Pre-admission Fluid SQL Query|Pre-admission Fluid SQL Query]]
- [[_COMMUNITY_Urine Output SQL Query|Urine Output SQL Query]]

## God Nodes (most connected - your core abstractions)
1. `MIMIC-Sepsis Project README` - 20 edges
2. `run_benchmark()` - 19 edges
3. `TimeSeriesDataProcessor` - 19 edges
4. `main()` - 17 edges
5. `main()` - 17 edges
6. `main()` - 16 edges
7. `DataFrame` - 13 edges
8. `_ProgressBar` - 13 edges
9. `DataFrame` - 12 edges
10. `ndarray` - 12 edges

## Surprising Connections (you probably didn't know these)
- `XGBoost Model` --semantically_similar_to--> `LightGBM Model`  [INFERRED] [semantically similar]
  plans/xgboost_score_model.md → README.md
- `Cavecrew README (agents)` --semantically_similar_to--> `Cavecrew README (roo)`  [INFERRED] [semantically similar]
  .agents/skills/cavecrew/README.md → .roo/skills/cavecrew/README.md
- `Cavecrew SKILL (agents)` --semantically_similar_to--> `Cavecrew SKILL (roo)`  [INFERRED] [semantically similar]
  .agents/skills/cavecrew/SKILL.md → .roo/skills/cavecrew/SKILL.md
- `Caveman-Commit README (agents)` --semantically_similar_to--> `Caveman-Commit README (roo)`  [INFERRED] [semantically similar]
  .agents/skills/caveman-commit/README.md → .roo/skills/caveman-commit/README.md
- `Caveman-Commit SKILL (agents)` --semantically_similar_to--> `Caveman-Commit SKILL (roo)`  [INFERRED] [semantically similar]
  .agents/skills/caveman-commit/SKILL.md → .roo/skills/caveman-commit/SKILL.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Caveman Toolkit Ecosystem** — caveman_mode_concept, caveman_compress_process, cavecrew_concept, conventional_commits_concept, caveman_review_concept, caveman_stats_hook [INFERRED 0.85]
- **Cavecrew Locate-Fix-Verify Chain** — cavecrew_investigator_concept, cavecrew_builder_concept, cavecrew_reviewer_concept [EXTRACTED 1.00]
- **Agents and Roo Skill Mirror Sets** — agents_cavecrew_skill, roo_cavecrew_skill, agents_cavemancommit_skill, roo_cavemancommit_skill [INFERRED 0.85]
- **MIMIC-Sepsis 4-Stage Benchmark Pipeline Stages** — data_extraction_stage, preprocessing_stage, trajectory_stage, benchmarking_stage [EXTRACTED 1.00]
- **Score Prediction Multi-Model Framework (XGBoost/Prophet/TFT)** — xgboost_model_concept, prophet_model_concept, tft_model_concept, multistep_prediction_concept [INFERRED 0.85]
- **Caveman Skill Suite (compress/review/stats/help)** — caveman_compress_skill, caveman_review_skill, caveman_stats_skill, caveman_help_skill [EXTRACTED 1.00]
- **MIMIC-IV to Final Cohort End-to-End Pipeline Flow** — concept_mimic_iv_database, concept_sql_extraction, concept_aggregation_imputation, concept_derived_variables, concept_flag_sepsis_onset, concept_final_cohort [EXTRACTED 1.00]
- **SOFA Score Component Clinical Parameters** — concept_sofa_score, concept_creatinine, concept_bilirubin_total, concept_platelets, concept_gcs, concept_map, concept_sbp_arterial [INFERRED 0.85]
- **Treatment Timing Impact on Hospital Mortality** — concept_antibiotic_treatment, concept_vasopressor_treatment, concept_hospital_mortality, concept_early_vs_late_antibiotics, concept_early_vs_late_vasopressors [EXTRACTED 1.00]

## Communities (49 total, 20 thin omitted)

### Community 0 - "Polars Trajectory Formatting"
Cohesion: 0.07
Nodes (40): add_sepsis_flag(), add_septic_shock_flag(), apply_exclusion_criteria(), calculate_derived_variables(), combine_patient_data(), ComputeBackend, estimate_fio2(), estimate_gcs_from_rass() (+32 more)

### Community 1 - "GPU Trajectory Formatting"
Cohesion: 0.07
Nodes (37): add_sepsis_flag(), add_septic_shock_flag(), apply_exclusion_criteria(), calculate_derived_variables(), combine_patient_data(), ComputeBackend, estimate_fio2(), estimate_gcs_from_rass() (+29 more)

### Community 2 - "Benchmark Pipeline Architecture"
Cohesion: 0.12
Nodes (40): 4-Stage Benchmark Pipeline, Stage 4: Benchmarking (benchmark.py), Charlson Comorbidity Index (CCI), Class Balancing (SMOTE/undersample), Stage 1: Data Extraction (SQL → CSV), MIMIC-Sepsis Project Website (index.html), FiO2 Estimation from Oxygen Flow and Device, Fluid Total Equivalent Volume (TEV) Standardization (+32 more)

### Community 3 - "Time Series Data Processor"
Cohesion: 0.12
Nodes (20): balance_dataframe(), _ProgressBar, DataFrame, ndarray, Parameters         ----------         features : list of str         task : str, Minimal tqdm-free progress bar that works in terminals and Jupyter., Prepare features and targets based on the task type.          Parameters, For sepsis prediction:         - Use sliding windows of fixed size         - Pre (+12 more)

### Community 4 - "Clinical Parameters & Figures"
Cohesion: 0.11
Nodes (37): Data Aggregation and Imputation, Antibiotic Treatment, Bilirubin Total (Hepatic Parameter), Correlation Between Key Clinical Parameters, Clinical Rule Based Imputation, Creatinine (Renal Parameter), Demographics Raw Table, Computing Derived Variables (SOFA, SIRS) (+29 more)

### Community 5 - "Pandas Trajectory Formatting"
Cohesion: 0.08
Nodes (35): add_sepsis_flag(), add_septic_shock_flag(), apply_exclusion_criteria(), calculate_derived_variables(), combine_patient_data(), estimate_fio2(), estimate_gcs_from_rass(), fixgaps() (+27 more)

### Community 6 - "Gradient Boosting Models"
Cohesion: 0.11
Nodes (18): _flatten(), LightGBMModel, MultiScoreModel, ndarray, Gradient-boosting models for clinical score prediction.  Provides:   - XGBoostMo, Train the model.          Parameters         ----------         X : np.ndarray,, Generate predictions.          Returns probabilities (positive class) for classi, Feature importances from the trained XGBoost model (gain-based). (+10 more)

### Community 7 - "Caveman Agent Skills"
Cohesion: 0.10
Nodes (26): Cavecrew README (agents), Cavecrew SKILL (agents), Caveman README (agents), Caveman SKILL (agents), Caveman-Commit README (agents), Caveman-Commit SKILL (agents), Caveman-Compress README (agents), Caveman-Compress SECURITY (agents) (+18 more)

### Community 8 - "Benchmark Evaluation Engine"
Cohesion: 0.14
Nodes (23): evaluate_model(), evaluate_model_multistep(), get_baseline_metrics(), get_feature_columns(), load_data(), print_results(), print_results_multistep(), DataFrame (+15 more)

### Community 9 - "Compress Validation (agents)"
Cohesion: 0.20
Nodes (17): Path, count_bullets(), extract_code_blocks(), extract_headings(), extract_inline_codes(), extract_paths(), extract_urls(), Line-based fenced code block extractor.      Handles ``` and ~~~ fences with var (+9 more)

### Community 10 - "Compress Validation (roo)"
Cohesion: 0.20
Nodes (17): Path, count_bullets(), extract_code_blocks(), extract_headings(), extract_inline_codes(), extract_paths(), extract_urls(), Line-based fenced code block extractor.      Handles ``` and ~~~ fences with var (+9 more)

### Community 11 - "Caveman Communication Tools"
Cohesion: 0.14
Nodes (18): Caveman-Stats README (agents), Caveman-Stats SKILL (agents), Caveman Auto-Clarity Rule, Token Reduction via File Compression Rationale, Caveman Compress Security Policy, Caveman Compress Skill, Caveman Help README, Caveman Help Skill (+10 more)

### Community 12 - "Transformer Model"
Cohesion: 0.16
Nodes (9): PositionalEncoding, ndarray, Tensor, Train the Transformer model with AMP, early stopping, weight decay, and LR sched, Run validation and return validation loss, Generate predictions using the trained model                  Args:, Initialize the model architecture with the given dimensions, Forward pass for the transformer model                  Args:             x: Inp (+1 more)

### Community 13 - "Prophet Forecasting Model"
Cohesion: 0.17
Nodes (11): _fit_one_patient(), ProphetScoreModel, DataFrame, ndarray, Prophet-based multi-step clinical score forecasting.  Provides:   - ProphetScore, Fit one Prophet model per patient in parallel.          Parameters         -----, Forecast H steps ahead for each validation patient.          Parameters, Extract the actual H future score values for each patient.          For evaluati (+3 more)

### Community 14 - "Temporal Fusion Transformer"
Cohesion: 0.19
Nodes (9): DataFrame, ndarray, Temporal Fusion Transformer (TFT) for multi-step clinical score forecasting.  Pr, Build a pytorch-forecasting TimeSeriesDataSet.          Parameters         -----, Train the TFT model.          Parameters         ----------         train_df : p, Generate multi-step predictions.          Parameters         ----------, Return TFT variable importance (attention-based).          Returns         -----, Temporal Fusion Transformer for multi-step score prediction.      Uses pytorch-f (+1 more)

### Community 15 - "Compress Core Scripts (agents)"
Cohesion: 0.22
Nodes (14): Path, backup_dir_for(), build_compress_prompt(), build_fix_prompt(), call_claude(), compress_file(), is_sensitive_path(), Strip outer ```markdown ... ``` fence when it wraps the entire output. (+6 more)

### Community 16 - "LSTM Model"
Cohesion: 0.17
Nodes (9): DataLoader, Module, LSTMModel, ndarray, Tensor, Train the LSTM model with AMP, early stopping, weight decay, and LR scheduling., Generate predictions using the trained model, Parameters         ----------         input_dim : int             Number of inpu (+1 more)

### Community 17 - "Compress Core Scripts (roo)"
Cohesion: 0.22
Nodes (14): Path, backup_dir_for(), build_compress_prompt(), build_fix_prompt(), call_claude(), compress_file(), is_sensitive_path(), Strip outer ```markdown ... ``` fence when it wraps the entire output. (+6 more)

### Community 18 - "Results Visualization"
Cohesion: 0.31
Nodes (12): _ensure_dir(), plot_multistep_rmse(), plot_predicted_vs_actual(), plot_roc_pr(), plot_training_curves(), Plot utilities for benchmark results.  All functions save to `out_dir` and close, Scatter plot of predicted vs actual for each horizon step., Train and validation loss per epoch. (+4 more)

### Community 19 - "File Type Detection (agents)"
Cohesion: 0.24
Nodes (11): Path, detect_file_type(), _is_code_line(), _is_json_content(), _is_yaml_content(), Return True if the file is natural language and should be compressed., Check if a line looks like code., Check if content is valid JSON. (+3 more)

### Community 20 - "File Type Detection (roo)"
Cohesion: 0.24
Nodes (11): Path, detect_file_type(), _is_code_line(), _is_json_content(), _is_yaml_content(), Return True if the file is natural language and should be compressed., Check if a line looks like code., Check if content is valid JSON. (+3 more)

### Community 21 - "ICU Data Preprocessing"
Cohesion: 0.27
Nodes (11): determine_readmission(), fill_missing_icustay_ids(), find_infection_onset(), load_processed_files(), main(), parse_args(), process_demog_data(), process_microbio_data() (+3 more)

### Community 22 - "Linear Time Series Model"
Cohesion: 0.33
Nodes (4): LinearTimeSeriesModel, ndarray, Args:             task_type: 'classification' or 'regression'             random, Reshape (N, T, X) data to (N, T*X)

### Community 23 - "Compress Benchmarking (agents)"
Cohesion: 0.60
Nodes (5): Path, benchmark_pair(), count_tokens(), main(), print_table()

### Community 24 - "Compress Benchmarking (roo)"
Cohesion: 0.60
Nodes (5): Path, benchmark_pair(), count_tokens(), main(), print_table()

### Community 25 - "Metrics Calculation"
Cohesion: 0.50
Nodes (3): calculate_metrics(), ndarray, Calculate task-specific evaluation metrics

## Knowledge Gaps
- **27 isolated node(s):** `ndarray`, `ndarray`, `ndarray`, `Tensor`, `Module` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `balance_dataframe()` connect `Time Series Data Processor` to `Polars Trajectory Formatting`, `GPU Trajectory Formatting`, `Pandas Trajectory Formatting`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `TimeSeriesDataProcessor` connect `Time Series Data Processor` to `Benchmark Evaluation Engine`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `run_benchmark()` connect `Benchmark Evaluation Engine` to `Time Series Data Processor`, `Gradient Boosting Models`, `Transformer Model`, `Prophet Forecasting Model`, `Temporal Fusion Transformer`, `LSTM Model`, `Linear Time Series Model`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `run_benchmark()` (e.g. with `TimeSeriesDataProcessor` and `LinearTimeSeriesModel`) actually correct?**
  _`run_benchmark()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TimeSeriesDataProcessor` (e.g. with `DataFrame` and `ndarray`) actually correct?**
  _`TimeSeriesDataProcessor` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Caveman compress scripts.  This package provides tools to compress natural langu`, `Split YAML frontmatter from body. Returns (frontmatter, body).      Memory files`, `Resolve the out-of-tree backup directory for a given source file.      Backups m` to the rest of the system?**
  _187 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Polars Trajectory Formatting` be split into smaller, more focused modules?**
  _Cohesion score 0.06666666666666667 - nodes in this community are weakly interconnected._