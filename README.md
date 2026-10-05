# Diabetes Hospital Readmission — Data Science Project

## Project structure

- `notebooks/`: Jupyter notebooks with the analysis.
- `src/`: shared Python code (e.g. `data_prep.py` for loading the data correctly).
- `data/`: local datasets, excluded from Git.
- `docs/`: reference documents for the project (project brief, project plan) and `decision_log.md`, the record of every methodological decision.
- `reports/`: generated HTML profiling reports, excluded from Git (regenerate them locally from the notebook).
- `requirements.txt`: Python dependencies.
- `CLAUDE.md`: project rules and verified data facts, read automatically by Claude Code.

## Setup

1. Create the virtual environment with **Python 3.13** (required by the course; `ydata-profiling` does not support Python 3.14) and install the dependencies:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled on this system"), run this once and try again:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

2. Copy `diabetic_data.csv` and `IDS_mapping.csv` into the `data/` folder. These files must remain available locally only (they are gitignored).

## Running the EDA notebook

1. Open `notebooks/EDA Diabetes Project.ipynb` in VS Code.
2. Select the kernel: **Select Kernel → Python Environments → `.venv` (Python 3.13)**. You can check it with the `print(sys.executable)` cell, which must point to `.venv`.
3. Click **Run All**.
4. To regenerate the profiling report, uncomment the `ProfileReport` cell and run it (it takes a few minutes). The report is saved to `reports/df_diabetes_report.html` and can be opened in a browser.

The notebook imports the loading functions from `src/data_prep.py`, so it must be run from the project (it finds the project root automatically).

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

**Output:** decision log entries updated from *Proposed* / *Open* to *Accepted*.

### Step 2 — Deterministic cleaning in `src/`

A `clean()` function applies only row-wise, rule-based transformations that do not learn anything from the data, so it can safely run before the split. Anything that is fitted on the data (top-N categories, encoding, scaling, imputation) belongs to the model pipeline and is fitted on the training set only.

- [ ] Filter rows according to the decisions in Step 1.
- [ ] Create the target `readmitted_30 = (readmitted == "<30")`; keep the raw `readmitted` column out of the features.
- [ ] Drop `examide`, `citoglipton`, `weight` and the rare medications.
- [ ] Make missing values explicit: `race`, `payer_code`, `medical_specialty` -> `"Unknown"`; `diag_1/2/3` -> `"Missing"`.
- [ ] Group the diagnoses into the Strack et al. ICD-9 categories (`diag_1/2/3_group`), handling `250.xx` and V/E codes before any numeric conversion, and keep the raw strings.
- [ ] Add medication summary features: `n_meds_active` (drugs not "No") and `n_meds_changed` (drugs "Up" or "Down"); keep `insulin` as its own feature.
- [ ] Cast `age` to an ordered category.
- [ ] End with sanity checks (`assert`): expected number of rows, no missing target, expected columns.

**Output:** a tested `clean()` function and a `02_data_cleaning.ipynb` notebook showing, for each step, rows, columns and `<30` rate before and after.

### Step 3 — Split and freeze the dataset

- [ ] Create the test holdout with the strategy chosen in D-010 (e.g. `GroupShuffleSplit`, 80/20 on `patient_nbr`, fixed `random_state`).
- [ ] Check that no `patient_nbr` appears in both sets and that the `<30` rate is similar in both.
- [ ] Save `data/processed/train.parquet` and `data/processed/test.parquet` (local only).
- [ ] Regenerate the profiling report on the clean dataset.

**Output:** the frozen modelling dataset. This closes Phase 1; from here the other work streams can start in parallel.

### Step 4 — Target-oriented EDA (training set only)

- [ ] `<30` rate by `number_inpatient`, `discharge_disposition_desc`, `admission_source_desc`, `age`, `A1Cresult`, `diag_1_group`, `medical_specialty`, `n_meds_changed`, `insulin`.
- [ ] Distributions by `race`, `gender` and `age`, as a baseline for the fairness audit (Phase 5).
- [ ] A short list of insights for the report and for feature selection.

**Output:** a `03_eda_target.ipynb` notebook.

### Step 5 — Prepare Phase 2 (model)

- [ ] `ColumnTransformer` + `Pipeline` with encoding, top-N categories and scaling (`log1p` on count variables for the logistic regression).
- [ ] Baseline logistic regression with `class_weight="balanced"`, evaluated with grouped cross-validation on `patient_nbr`.
- [ ] Fix the evaluation metrics: PR-AUC, ROC-AUC, recall and precision at top-K.
