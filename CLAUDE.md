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
- `data/*.csv` and `reports/*.html` are local only (gitignored): never commit them.

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
