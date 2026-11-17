# Diabetes Hospital Readmission — notes for Claude

NOVA IMS Data Science Bootcamp group project: predict 30-day readmission (`readmitted == "<30"`) of diabetic patients.
Reference documents are in `docs/` (course brief and project plan): follow them when a choice is not covered here.

## Environment

- Python 3.13 (required by the course), virtual environment in `.venv/`, dependencies in `requirements.txt`.
- `ydata-profiling` does not support Python 3.14: do not upgrade the interpreter.
- Code formatting: Black.

## Conventions

- Code, comments, markdown cells and file names are in English.
- Shared code lives in `src/`; notebooks import it instead of duplicating logic.
- `data/*.csv` are local only (gitignored): never commit them.
- `reports/df_diabetes_report.html` is the only committed HTML report: the clean profile, generated from `add_mapping_descriptions(load_raw(), load_mappings())`. Never overwrite it with a report built from a bare `pd.read_csv()` / `pl.read_csv()`. Other HTML reports stay local (gitignored).
- Modelling notebooks in `DC_Models/` are committed in both formats: the executed `.ipynb` and its Jupytext `.py:percent` copy (see README).

## Loading the data

Always use `src/data_prep.py`, never a bare `pd.read_csv()`:

- `load_raw()`: `"?"` -> NaN; `"None"` in `max_glu_serum` / `A1Cresult` is kept as a category ("test not performed", informative, not missing); the 3 `*_id` columns are cast to category (they are codes, not quantities).
- `load_mappings()`: `IDS_mapping.csv` is 3 stacked tables separated by empty lines; it is parsed into one dict per ID column.
- `add_mapping_descriptions()`: adds `*_desc` columns without overwriting the IDs; `NULL`, `Not Mapped`, `Not Available`, `Unknown/Invalid` -> `Unknown`.

## Known data facts (verified)

- 101,766 encounters, 71,518 patients. `<30` rate: 11.2% on all encounters, 8.8% on the first encounter per patient.
- `examide` and `citoglipton` are constant ("No"). `weight` is 96.9% missing.
- 2,423 encounters end in death/hospice (`discharge_disposition_id` in {11, 13, 14, 19, 20, 21}): they cannot be readmitted.
- If `change == "Ch"` then `diabetesMed == "Yes"` (structural relationship, not an error).
- `encounter_id` and `patient_nbr` are identifiers, never features.

## Methodological rules

- Patient-level leakage: train/test splits must never put encounters of the same patient on both sides (GroupKFold on `patient_nbr`, or keep only the first encounter per patient). The final choice must be recorded in the decision log.
- Everything that learns from the data (rare-level grouping, encoding, scaling, imputation, feature selection) lives inside a scikit-learn `Pipeline` (`src/preprocessing.py`), fitted on the fitting rows of each split or fold only (D-014).
- Every cross-validation and hyperparameter search is grouped on `patient_nbr` (`StratifiedGroupKFold`, `groups=` in `fit`). No random internal splits: early stopping off, Keras without `validation_split`, no `StackingClassifier` (D-017).
- Tune on the grouped CV of the training rows, compare on validation, open the test set once (D-015). Target rates and any other look at the outcome happen on training rows only.
- Run `python -m unittest` after changing `src/`.
