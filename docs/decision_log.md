# Decision log

This file records every methodological decision of the project: what was decided, why, which alternatives were considered and what impact the decision has on the data.

## How to use this log

- **One entry per decision**, numbered `D-001`, `D-002`, … Reference the ID in the code that implements it (e.g. `# see D-004`).
- **Status** is one of:
  - *Accepted*: agreed by the team and implemented, or ready to be implemented.
  - *Proposed*: drafted, waiting for team agreement.
  - *Open*: alternatives identified, no proposal agreed yet.
  - *Superseded by D-xxx*: replaced by a later decision.
- **Never delete or rewrite past entries.** If a decision changes, add a new entry and mark the old one as *Superseded*.
- **Always state the impact in numbers** (rows, columns, `<30` rate), so that every entry can be verified.
- **Update the log in the same commit** as the code that implements the decision.

All figures below refer to the raw `diabetic_data.csv` (101,766 encounters, 71,518 patients, `<30` rate 11.16%) loaded with `load_raw()`.

## Summary

| ID | Decision | Status |
|---|---|---|
| D-001 | Missing-value markers: `"?"` is missing, `"None"` is a category | Accepted |
| D-002 | ID columns cast to category | Accepted |
| D-003 | Decode ID columns with the lookup table | Accepted |
| D-004 | Remove expired encounters | Proposed |
| D-005 | Remove hospice encounters | Proposed |
| D-006 | Drop constant columns and `weight` | Proposed |
| D-007 | Drop rare medications | Proposed |
| D-008 | Explicit "Unknown" / "Missing" levels for missing categorical values | Proposed |
| D-009 | Group `medical_specialty` into top-N levels inside the model pipeline | Proposed |
| D-010 | Train/test split strategy | Open |
| D-011 | Binary target definition | Accepted |
| D-012 | Identifiers are never used as features | Accepted |

---

## D-001 — Missing-value markers
- **Date:** 2026-10-05 · **Status:** Accepted · **Implemented in:** `src/data_prep.py` (`load_raw()`)
- **Decision:** Load the data with `keep_default_na=False, na_values=["?"]`.
- **Rationale:** The dataset uses `"?"` as its missing-value marker, which pandas does not recognise by default. Pandas does, however, turn the string `"None"` into NaN, while in `max_glu_serum` and `A1Cresult` `"None"` means "test not performed" (course brief), which is informative and not missing.
- **Alternatives considered:** Default `pd.read_csv()` — rejected, it produced misleading profiling alerts (`max_glu_serum` 94.7% and `A1Cresult` 83.3% "missing"; `weight` reported as imbalanced instead of 96.9% missing).
- **Impact:** No rows removed. Real missing values now visible: `weight` 98,569 (96.9%), `medical_specialty` 49,949 (49.1%), `payer_code` 40,256 (39.6%), `race` 2,273 (2.2%), `diag_3` 1,423, `diag_2` 358, `diag_1` 21.

## D-002 — ID columns cast to category
- **Date:** 2026-10-05 · **Status:** Accepted · **Implemented in:** `src/data_prep.py` (`load_raw()`)
- **Decision:** Cast `admission_type_id`, `discharge_disposition_id` and `admission_source_id` to `category`.
- **Rationale:** They are codes, not quantities: their numeric order and distances have no meaning.
- **Impact:** No rows or columns removed; the columns are profiled and later encoded as categorical.

## D-003 — Decode ID columns with the lookup table
- **Date:** 2026-10-05 · **Status:** Accepted · **Implemented in:** `src/data_prep.py` (`load_mappings()`, `add_mapping_descriptions()`)
- **Decision:** Parse `IDS_mapping.csv` into one dictionary per ID column and add `*_desc` columns next to the original IDs. Merge the descriptions without information (`NULL`, `Not Mapped`, `Not Available`, `Unknown/Invalid`) into a single `Unknown` level.
- **Rationale:** The file contains 3 lookup tables stacked on top of each other, so it is not a dataset and must not be profiled or joined as one table. Keeping the IDs makes filters reproducible; the descriptions make the data readable.
- **Impact:** +3 columns. Only one of `*_id` / `*_desc` will be used as a feature, since they are perfectly associated.
- **Known issue:** descriptions 19, 20 and 21 are quoted and contain a comma, so the current parser keeps the quotes. To be fixed with `csv.reader` (README, Step 0). Filters must always use the IDs, not the descriptions.

## D-004 — Remove expired encounters
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop encounters with `discharge_disposition_id` in {11, 19, 20} (21 is defined in the lookup table but does not occur in the data).
- **Rationale:** Based on the descriptions in `IDS_mapping.csv` ("Expired", "Expired at home…", "Expired in a medical facility…"), these patients died during the encounter, so a readmission cannot be observed. All 1,652 rows are labelled `NO` by construction; keeping them would teach the model that very severe profiles carry a low risk. The prototype also has no follow-up decision to support for these patients.
- **Alternatives considered:** None viable.
- **Impact:** −1,652 rows (101,766 → 100,114; 70,439 patients); `<30` rate 11.16% → 11.34%.
- **Limitations:** The dataset has no date-of-death field: deaths after discharge cannot be identified and remain labelled `NO`. 14 encounters have an `encounter_id` greater than an expired encounter of the same patient, so either `encounter_id` is not strictly chronological or the source contains inconsistencies.

## D-005 — Remove hospice encounters
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop encounters with `discharge_disposition_id` in {13, 14} ("Hospice / home", "Hospice / medical facility").
- **Rationale:** Hospice is end-of-life care: for these patients a readmission is not a prevention target and follow-up has a different meaning (palliative care). Consistent with Strack et al. (2014) and with the project plan.
- **Alternatives considered:** Keep them with a `hospice` flag. These patients can be readmitted: 43 of 771 are `<30` and 43 are `>30`.
- **Impact (after D-004):** −771 rows (100,114 → 99,343; 69,990 patients); `<30` rate 11.34% → 11.39%.

## D-006 — Drop constant columns and `weight`
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop `examide`, `citoglipton` and `weight`.
- **Rationale:** `examide` and `citoglipton` are always "No" (no information). `weight` is 96.9% missing, and the brief asks not to cast its bands to numeric values. Already planned in the project plan.
- **Alternatives considered:** Keep a `weight_recorded` flag (recorded vs. missing) — can be reconsidered if the EDA shows a relationship with the target.
- **Impact:** −3 columns.

## D-007 — Drop rare medications
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop medication columns with fewer than 100 non-"No" rows, after computing the medication summary features (Step 2), so that their information is still counted there.
- **Rationale:** With so few non-"No" values these columns cannot be learned reliably and make the association measures unstable (source of the spurious "high correlation" profiling alerts).
- **Columns (non-"No" rows):** `acetohexamide` (1), `glimepiride-pioglitazone` (1), `metformin-pioglitazone` (1), `metformin-rosiglitazone` (2), `troglitazone` (3), `glipizide-metformin` (13), `tolbutamide` (23), `miglitol` (38), `tolazamide` (39), `chlorpropamide` (86).
- **Alternatives considered:** Group them into a single `other_meds` feature.
- **Impact:** −10 columns. The next least used drug is `acarbose` (308 rows), which is kept.

## D-008 — Explicit levels for missing categorical values
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Replace missing values with `"Unknown"` in `race`, `payer_code` and `medical_specialty`, and with `"Missing"` in `diag_1`, `diag_2` and `diag_3`. Keep the 3 `gender == "Unknown/Invalid"` encounters as they are.
- **Rationale:** Missingness itself can be informative (e.g. a missing specialty reflects how the encounter was recorded). The project plan asks for an explicit "Unknown" level; the brief asks not to infer gender.
- **Alternatives considered:** Imputation with the mode — rejected, it would invent values.
- **Impact:** No rows removed.

## D-009 — Group `medical_specialty`
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Keep the top-N most frequent levels (~10) and merge the rest into `"Other"`, as a step of the model pipeline fitted on the training set only.
- **Rationale:** 72 levels, many very rare; the top 10 cover ~86% of the non-missing values. Choosing the levels on the full dataset would leak information from the test set.
- **Alternatives considered:** Manual grouping by clinical area (e.g. surgery, internal medicine).
- **Impact:** To be measured on the training set.

## D-010 — Train/test split strategy
- **Date:** 2026-10-05 · **Status:** Open
- **Problem:** `patient_nbr` repeats (71,518 patients for 101,766 encounters). A random split by encounter would put the same patient in both train and test and inflate the metrics.
- **Options:**
  - **A — All encounters + grouped split on `patient_nbr`** (`GroupShuffleSplit` for the test holdout, `GroupKFold` for cross-validation). Keeps all the data and matches the real use case (every discharge is scored).
  - **B — First encounter per patient only** (Strack et al.). Simpler and with independent rows, but loses ~30% of the encounters; `<30` rate 8.8% on the raw data. It also assumes that `encounter_id` gives the chronological order (see the limitation in D-004).
- **Proposal:** A.
- **Impact:** To be filled once decided.

## D-011 — Binary target definition
- **Date:** 2026-10-05 · **Status:** Accepted
- **Decision:** `readmitted_30 = 1` if `readmitted == "<30"`, else `0` (`>30` and `NO` are both negative). The raw `readmitted` column is never used as a feature.
- **Rationale:** Defined by the course brief: the task is early readmission in less than 30 days.
- **Impact:** Positive class 11.16% on the raw data (11.39% after D-004 and D-005), so the classes are imbalanced: handled with class weights and evaluated with PR-AUC and top-K metrics (project plan).

## D-012 — Identifiers are never features
- **Date:** 2026-10-05 · **Status:** Accepted
- **Decision:** `encounter_id` is used only as the row key and `patient_nbr` only to group the split; neither is used as a feature.
- **Rationale:** They are identifiers assigned over time (their "high correlation" profiling alert is an artefact) and carry no clinical meaning.
- **Impact:** −2 columns from the feature set.
