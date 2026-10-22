# Diabetes Hospital Readmission — Data Science Project
try to contribute
## Project structure

- `notebooks/`: the shared analysis: `01_eda_raw.ipynb` (first look at the raw data), `02_data_cleaning.ipynb` (the cleaning step by step and the frozen split), `03_eda_target.ipynb` (the `<30` rate on the training rows).
- `LA_Models/`, `DC_Models/`: modelling notebooks of two independent work streams, compared at the end.
- In `DC_Models/` and for `notebooks/02_*` and `03_*`, each `.ipynb` has a `.py` copy in Jupytext "percent" format (code and text only, no outputs).
- `src/`: shared Python code, imported by the notebooks:
  - `data_prep.py`: loading the raw data correctly;
  - `features.py`: the deterministic cleaning (`clean`, with the decision-log proposals as defaults and the alternatives as options), the feature lists and the patient-grouped split;
  - `preprocessing.py`: the preprocessing of each model family, as scikit-learn transformers fitted inside the model pipeline;
  - `models.py`: scikit-learn wrappers for XGBoost AFT, CatBoost and Keras, and `with_seed` to refit any model with another seed;
  - `evaluation.py`: the metrics, the readable result tables and `rate_by` (a rate by level, with an interval by patient);
  - `calibration.py`: the two leakage-safe ways to calibrate the chosen model (holdout, grouped out of fold) and the capacity tiers.
- `data/`: local datasets, excluded from Git.
- `docs/`: reference documents for the project (project brief, project plan) and `decision_log.md`, the record of every methodological decision.
- `reports/`: experiment logs (`experiments-readmission*.csv`, one row per model and dataset, committed) and `df_diabetes_report.html`, the clean profiling report of the raw data (committed; other HTML reports, such as `df_train_clean_report.html` built by `02_data_cleaning.ipynb`, stay local).
- `requirements.txt`: Python dependencies; `SETUP.md`: how to install them on Windows and macOS.
- `tests/`: unit tests of the code in `src/` (`python -m unittest`, from the project folder).
- `CLAUDE.md`: project rules and verified data facts, read automatically by Claude Code.

## Setup

Python 3.13, a virtual environment in `.venv/` and `pip install -r requirements.txt`, then the two data files in `data/`. The commands differ between Windows and macOS (macOS also needs `brew install libomp`, and TensorFlow has no wheel for Intel Macs): see **[SETUP.md](SETUP.md)** for the step-by-step guide of each system.

## Running the EDA notebook

1. Open `notebooks/01_eda_raw.ipynb` in VS Code.
2. Select the kernel: **Select Kernel → Python Environments → `.venv` (Python 3.13)**. You can check it with the `print(sys.executable)` cell, which must point to `.venv`.
3. Click **Run All**.
4. To regenerate the profiling report, uncomment the `ProfileReport` cell and run it (it takes a few minutes). The report is saved to `reports/df_diabetes_report.html` and can be opened in a browser.

The notebook imports the loading functions from `src/data_prep.py`, so it must be run from the project (it finds the project root automatically).

The same way, then run `02_data_cleaning.ipynb` (about a minute; the first run also builds `reports/df_train_clean_report.html`, a few more minutes) and `03_eda_target.ipynb` (seconds).

## Running the modelling notebooks (`DC_Models/`)

Run them in this order, with the `.venv` kernel (**Restart Kernel → Run All**):

| Notebook | What it does | Time |
|---|---|---|
| `01_cleaning_variants.ipynb` | Tests the cleaning proposals of the decision log against their alternatives, with patient-grouped folds | ~3 min |
| `02_model_exploration.ipynb` | Compares the model families (linear, EBM, bagging, boosting, XGBoost AFT, Keras, ensemble) and writes `reports/experiments-readmission-dc.csv` | ~45 min |

The test set is never opened during model selection: it is locked behind `OPEN_TEST = False` in Stage 7 of `02_model_exploration.ipynb`, and it cannot be opened before `PRIMARY` declares which calibration option (rehearsed on validation in Stage 6) is the main result. To make `02` faster, lower the values of `N_ITER` at the start of Stage 5 (0 skips the hyperparameter searches).

After changing a notebook, regenerate its `.py` copy before committing:

```powershell
jupytext --to py:percent --opt notebook_metadata_filter=-all --opt cell_metadata_filter=-all DC_Models/02_model_exploration.ipynb
```

## How we obtain a clean profiling report

A first version of the report, generated with a bare `pd.read_csv()`, contained several misleading alerts that came from the loading, not from the data. We fixed them as follows:

1. **Correct missing values.** The dataset uses `"?"` as the missing-value marker, which pandas does not recognise by default, while pandas does turn the string `"None"` into NaN. In `max_glu_serum` and `A1Cresult`, however, `"None"` means "test not performed", which is an informative category and not a missing value. `load_raw()` therefore loads the data with `keep_default_na=False, na_values=["?"]`. As a result:
   - `max_glu_serum` and `A1Cresult` are no longer reported as 95% / 83% missing;
   - `weight` (96.9% missing), `payer_code`, `medical_specialty`, `race` and `diag_1/2/3` now show their real missing values instead of a `"?"` category.
2. **ID columns as categories.** `admission_type_id`, `discharge_disposition_id` and `admission_source_id` are codes, not quantities, so they are cast to `category` instead of being profiled as numbers.
3. **Lookup table decoded, not profiled.** `IDS_mapping.csv` is not a dataset: it contains 3 lookup tables stacked on top of each other, so profiling it has no analytical value. `load_mappings()` parses it into one dictionary per ID column, and `add_mapping_descriptions()` adds readable `*_desc` columns next to the original IDs. Descriptions without information (`NULL`, `Not Mapped`, `Not Available`, `Unknown/Invalid`) are merged into `Unknown`.

### Alerts that remain in the clean report

The remaining alerts are real properties of the data, or expected side effects of the fixes above. They will be handled in the data cleaning step:

| Alert | Interpretation |
|---|---|
| Constant: `examide`, `citoglipton` | Real: always "No" (shown as "False" because ydata treats them as booleans). To be dropped. |
| Missing: `weight` (96.9%), `medical_specialty` (49.1%), `payer_code` (39.6%), `race` (2.2%), `diag_3` (1.4%) | Real missing values, now visible thanks to the correct loading. |
| Imbalance: most medications, `max_glu_serum`, `A1Cresult`, `race`, ID columns | Real: many drugs are almost never prescribed, and the lab tests are mostly "None" (not performed). |
| High correlation: `*_id` ↔ `*_desc` | Expected: the description is just a decoding of the ID. Only one of the two will be used as a feature. |
| High correlation: `change` ↔ `diabetesMed` ↔ `insulin` | Structural: if `change == "Ch"` then `diabetesMed == "Yes"` in every row. |
| High correlation: `encounter_id` ↔ `patient_nbr` | Both are identifiers assigned over time; neither is used as a feature. |
| High correlation: rare drugs ↔ `weight` / `payer_code` / `race` / `max_glu_serum` | Statistical artefact: drugs such as `acetohexamide`, `glimepiride-pioglitazone`, `metformin-pioglitazone` (1 non-"No" row each), `metformin-rosiglitazone` (2) and `troglitazone` (3) make the association measures unstable. These drugs will be dropped or grouped. |
| Skewed / Zeros: `number_emergency`, `number_outpatient`, `number_inpatient`, `num_procedures` | Normal for count variables (most patients have 0 previous visits). |
| Unique: `encounter_id` | Expected: it is the row key. |

## Next steps

The goal of the next steps is to close **Phase 1** of the project plan: a clean, patient-split, frozen dataset with every decision documented. Each step ends with a verifiable output. Decisions are recorded in [`docs/decision_log.md`](docs/decision_log.md) and referenced in the code by their ID (e.g. `# see D-004`).

### Step 0 — Quick fixes

- [x] `load_mappings()`: parse `IDS_mapping.csv` with `csv.reader` instead of `str.partition(",")`. Descriptions 19, 20 and 21 are quoted and contain a comma, so they kept their quotes (e.g. `'"Expired at home. Medicaid only, hospice."'`).
- [x] `.gitignore`: add `data/processed/`.

**Output:** one housekeeping commit.

### Step 1 — Team decisions

These decisions change the number of rows, so they must be agreed before writing the cleaning code. Proposals are drafted in the decision log with status *Proposed* or *Open*.

| Decision | Proposal | Alternative | Log entry |
|---|---|---|---|
| Expired encounters (IDs 11, 19, 20) | Remove (1,652 rows) | None: the outcome is not observable | D-004 |
| Hospice encounters (IDs 13, 14) | Remove (771 rows), as in Strack et al. and the project plan | Keep them with a `hospice` flag | D-005 |
| Rare medications | Drop drugs with fewer than 100 non-"No" rows (10 drugs) | Group them into `other_meds` | D-007 |
| Train/test split | All encounters + grouped split on `patient_nbr` | First encounter per patient only (Strack et al.) | D-010 |
| `medical_specialty` | Top-N levels (~10) + "Other", fitted inside the model pipeline | Manual grouping by clinical area | D-009 |

**Evidence (2026-10-19):** `DC_Models/01_cleaning_variants.ipynb` tests every alternative against the proposals with the same patient-grouped folds and two models. Every proposal holds; only D-010 makes a real difference (training on the first encounter only costs 0.011 PR-AUC). The numbers are in each decision log entry.

**Output:** decision log entries updated from *Proposed* / *Open* to *Accepted* (pending team agreement).

### Step 2 — Deterministic cleaning in `src/`

A `clean()` function applies only row-wise, rule-based transformations that do not learn anything from the data, so it can safely run before the split. Anything that is fitted on the data (top-N categories, encoding, scaling, imputation) belongs to the model pipeline and is fitted on the training set only.

Implemented in `clean()` (`src/features.py`), with the proposals as defaults and the Step 1 alternatives as options.

- [x] Filter rows according to the decisions in Step 1.
- [x] Create the target `readmitted_30 = (readmitted == "<30")`; keep the raw `readmitted` column out of the features.
- [x] Drop `examide`, `citoglipton`, `weight` and the rare medications.
- [x] Make missing values explicit: `race`, `payer_code`, `medical_specialty` -> `"Unknown"`; `diag_1/2/3` -> `"Missing"` (in the `diag_*_group` columns; the raw codes are reference columns, never features).
- [x] Group the diagnoses into the Strack et al. ICD-9 categories (`diag_1/2/3_group`), handling `250.xx` and V/E codes before any numeric conversion, and keep the raw strings.
- [x] Add medication summary features: `n_meds_active` (drugs not "No") and `n_meds_changed` (drugs "Up" or "Down"); keep `insulin` as its own feature.
- [x] Encode `age` by its order: `age_band_order` (0 for `[0-10)` to 9 for `[90-100)`), an integer that keeps the order of the bands without turning them into ages.
- [x] End with sanity checks (`assert`): expected number of rows, no missing target, expected columns (in the notebooks), plus unit tests in `tests/test_src.py` (`python -m unittest`).
- [x] A `02_data_cleaning.ipynb` notebook showing, for each step, rows, columns and `<30` rate before and after (`clean(..., log=[])` records them).

**Output:** a tested `clean()` function and the step-by-step cleaning notebook (`notebooks/02_data_cleaning.ipynb`).

### Step 3 — Split and freeze the dataset

- [x] Create the test holdout with the strategy chosen in D-010: `split_by_patient()` (`src/features.py`), two `GroupShuffleSplit` cuts on `patient_nbr`, 60/20/20 train/validation/test, `random_state=42`.
- [x] Check that no `patient_nbr` appears in two sets (`assert` in the function and a unit test) and that the `<30` rate is similar in all three (0.114 / 0.114 / 0.112).
- [x] ~~Save `data/processed/train.parquet` and `data/processed/test.parquet` (local only).~~ Replaced by fingerprints: `data/` is not in Git, so each member would rebuild the files anyway, while `clean()` + `split_by_patient()` rebuild the same three sets in seconds. `02_data_cleaning.ipynb` prints a hash of each set's `encounter_id` to check that two computers have the same data (train `1e2fece0c48e`, validation `920e3a74bf6e`, test `fe3f135ba61d`).
- [x] Regenerate the profiling report on the clean dataset: on the **training rows** only (`reports/df_train_clean_report.html`, local), because the report shows the correlations of the target.

**Output:** the frozen modelling dataset. This closes Phase 1; from here the other work streams can start in parallel.

### Step 4 — Target-oriented EDA (training set only)

- [x] `<30` rate by `number_inpatient`, `discharge_disposition_desc`, `admission_source_desc`, `age`, `A1Cresult`, `diag_1_group`, `medical_specialty`, `n_meds_changed`, `insulin`, with 95% intervals by patient.
- [x] Distributions by `race`, `gender` and `age`, as a baseline for the fairness audit (Phase 5).
- [x] A short list of insights for the report and for feature selection.

**Output:** `notebooks/03_eda_target.ipynb`.

### Step 5 — Prepare Phase 2 (model)

- [x] `ColumnTransformer` + `Pipeline` with encoding, top-N categories and scaling (`src/preprocessing.py`). `log1p` on the count variables is not applied yet: to be tested on validation.
- [x] Baseline logistic regression with `class_weight="balanced"`, evaluated with grouped cross-validation on `patient_nbr` (`DC_Models/02_model_exploration.ipynb`).
- [x] Fix the evaluation metrics: PR-AUC, ROC-AUC, recall and precision at top-K (`src/evaluation.py`).

### Step 6 — Phase 2 (model), in progress

- [x] Compare the model families with patient-grouped tuning and a fold-by-fold check (`DC_Models/02_model_exploration.ipynb`, Stage 5): tuned XGBoost is the best single model (cross-validated PR-AUC 0.228), a soft-voting ensemble the best overall (0.230). See D-019 in the decision log.
- [ ] Calibrate the chosen model on validation, build the capacity-based risk tiers and the decision curve (Stage 6).
- [ ] SHAP explanations and the fairness baseline by `race`, `age` and `gender` (Stage 6).
- [ ] Open the test set once (Stage 7).

The research directions beyond the project plan (statistical comparison of models, ranking objectives, comorbidity indices, conformal prediction…) are listed in the decision log, D-020 to D-029.
