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

Updates added after an entry was written are marked **Update** or **Evidence**, with their date; the original text is never rewritten.

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
| D-010 | Train/test split strategy | Proposed (option A) |
| D-011 | Binary target definition | Accepted |
| D-012 | Identifiers are never used as features | Accepted |
| D-013 | Cleaning and split as shared, tested code in `src/` | Accepted |
| D-014 | Every learned step lives inside a scikit-learn `Pipeline` | Accepted |
| D-015 | Model selection protocol | Proposed |
| D-016 | Evaluation metrics and class imbalance | Accepted |
| D-017 | No random internal splits (early stopping, Keras, stacking) | Accepted |
| D-018 | XGBoost AFT on interval-censored times | Accepted |
| D-019 | Choice of the model | Proposed |
| D-020 | Statistical comparison of models with patient-level uncertainty | Open |
| D-021 | Nested grouped cross-validation | Open |
| D-022 | Calibration, decision curve and capacity tiers | Proposed (calibration and tiers); Open (decision curve) |
| D-023 | Ranking objectives aligned with the follow-up capacity | Open |
| D-024 | Clinically grounded features and a LACE baseline | Open |
| D-025 | Using the three-level outcome | Open |
| D-026 | Patient-history features from earlier encounters | Open |
| D-027 | Conformal prediction for risk with guarantees | Open |
| D-028 | Fairness audit | Open |
| D-029 | Explainability, robustness and data limits | Open |
| D-030 | Audit against scikit-learn's "Common pitfalls and recommended practices" | Accepted |

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
- **Update (2026-10-05):** known issue resolved: `load_mappings()` now parses the file with `csv.reader` (README, Step 0), so descriptions 19, 20 and 21 no longer keep their quotes.

## D-004 — Remove expired encounters
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop encounters with `discharge_disposition_id` in {11, 19, 20} (21 is defined in the lookup table but does not occur in the data).
- **Rationale:** Based on the descriptions in `IDS_mapping.csv` ("Expired", "Expired at home…", "Expired in a medical facility…"), these patients died during the encounter, so a readmission cannot be observed. All 1,652 rows are labelled `NO` by construction; keeping them would teach the model that very severe profiles carry a low risk. The prototype also has no follow-up decision to support for these patients.
- **Alternatives considered:** None viable.
- **Impact:** −1,652 rows (101,766 → 100,114; 70,439 patients); `<30` rate 11.16% → 11.34%.
- **Limitations:** The dataset has no date-of-death field: deaths after discharge cannot be identified and remain labelled `NO`. 14 encounters have an `encounter_id` greater than an expired encounter of the same patient, so either `encounter_id` is not strictly chronological or the source contains inconsistencies.
- **Update (2026-10-19):** implemented in `clean()` (`src/features.py`, constant `EXPIRED`), codes checked against the lookup table with an `assert` in the notebooks. No alternative to test.

## D-005 — Remove hospice encounters
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop encounters with `discharge_disposition_id` in {13, 14} ("Hospice / home", "Hospice / medical facility").
- **Rationale:** Hospice is end-of-life care: for these patients a readmission is not a prevention target and follow-up has a different meaning (palliative care). Consistent with Strack et al. (2014) and with the project plan.
- **Alternatives considered:** Keep them with a `hospice` flag. These patients can be readmitted: 43 of 771 are `<30` and 43 are `>30`.
- **Impact (after D-004):** −771 rows (100,114 → 99,343; 69,990 patients); `<30` rate 11.34% → 11.39%.
- **Evidence (2026-10-19):** `DC_Models/01_cleaning_variants.ipynb`: the alternative against the proposals (variant P), same 5 patient-grouped folds on the development patients (test patients excluded), logistic regression (LR) and gradient boosting (GB), scored on the same encounters. Reference P: PR-AUC 0.2129 (LR), 0.2256 (GB). Decision rule: better/worse only with at least 4 folds of 5; a difference matters from 0.005. Alternative A1 (keep the hospice encounters with a `hospice` flag): LR −0.0011 PR-AUC (worse, 4 folds lost), GB −0.0004 (tie). Both models already score the hospice encounters lower (mean score 0.30–0.32 against 0.46; their `<30` rate is 5.6%). **Recommendation: accept the proposal** (no gain, and hospice is not a prevention target). Implemented as the default of `clean(hospice="drop")`.

## D-006 — Drop constant columns and `weight`
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop `examide`, `citoglipton` and `weight`.
- **Rationale:** `examide` and `citoglipton` are always "No" (no information). `weight` is 96.9% missing, and the brief asks not to cast its bands to numeric values. Already planned in the project plan.
- **Alternatives considered:** Keep a `weight_recorded` flag (recorded vs. missing) — can be reconsidered if the EDA shows a relationship with the target.
- **Impact:** −3 columns.
- **Evidence (2026-10-19):** `DC_Models/01_cleaning_variants.ipynb`: the alternative against the proposals (variant P), same 5 patient-grouped folds on the development patients (test patients excluded), logistic regression (LR) and gradient boosting (GB), scored on the same encounters. Reference P: PR-AUC 0.2129 (LR), 0.2256 (GB). Decision rule: better/worse only with at least 4 folds of 5; a difference matters from 0.005. Alternative A4 (`weight_recorded` flag): LR 0.0000, GB +0.0002 (ties); the `<30` rate is 11.3% with and without a recorded weight. **Recommendation: accept the proposal.** Implemented as the default of `clean(weight_flag=False)`.

## D-007 — Drop rare medications
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Drop medication columns with fewer than 100 non-"No" rows, after computing the medication summary features (Step 2), so that their information is still counted there.
- **Rationale:** With so few non-"No" values these columns cannot be learned reliably and make the association measures unstable (source of the spurious "high correlation" profiling alerts).
- **Columns (non-"No" rows):** `acetohexamide` (1), `glimepiride-pioglitazone` (1), `metformin-pioglitazone` (1), `metformin-rosiglitazone` (2), `troglitazone` (3), `glipizide-metformin` (13), `tolbutamide` (23), `miglitol` (38), `tolazamide` (39), `chlorpropamide` (86).
- **Alternatives considered:** Group them into a single `other_meds` feature.
- **Impact:** −10 columns. The next least used drug is `acarbose` (308 rows), which is kept.
- **Evidence (2026-10-19):** `DC_Models/01_cleaning_variants.ipynb`: the alternative against the proposals (variant P), same 5 patient-grouped folds on the development patients (test patients excluded), logistic regression (LR) and gradient boosting (GB), scored on the same encounters. Reference P: PR-AUC 0.2129 (LR), 0.2256 (GB). Decision rule: better/worse only with at least 4 folds of 5; a difference matters from 0.005. Alternative A2 (`other_meds` feature): LR 0.0000 (tie), GB identical predictions (the feature is never used: only about 200 encounters take a rare drug). **Recommendation: accept the proposal** (simpler). Implemented as the default of `clean(rare_meds="drop")`; the medication counts are computed before the drop.
- **Update (2026-10-20):** The list was chosen by counting non-"No" rows on **all** encounters, test patients included. The count does not use the target, and A2 shows that these drugs carry no signal, so the effect on the scores is negligible; but strictly it is a choice learned from data that include the test set (D-030). The list is now a fixed constant (`RARE_MEDICATIONS` in `src/features.py`) and is not recomputed. The strict alternative would be to count on the training rows only, inside the pipeline.

## D-008 — Explicit levels for missing categorical values
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Replace missing values with `"Unknown"` in `race`, `payer_code` and `medical_specialty`, and with `"Missing"` in `diag_1`, `diag_2` and `diag_3`. Keep the 3 `gender == "Unknown/Invalid"` encounters as they are.
- **Rationale:** Missingness itself can be informative (e.g. a missing specialty reflects how the encounter was recorded). The project plan asks for an explicit "Unknown" level; the brief asks not to infer gender.
- **Alternatives considered:** Imputation with the mode — rejected, it would invent values.
- **Impact:** No rows removed.
- **Update (2026-10-19):** implemented in `clean()`. The missing diagnoses become `"Missing"` in the `diag_*_group` features; the raw `diag_1/2/3` strings stay as reference columns (never features), so they keep their NaN.

## D-009 — Group `medical_specialty`
- **Date:** 2026-10-05 · **Status:** Proposed
- **Decision:** Keep the top-N most frequent levels (~10) and merge the rest into `"Other"`, as a step of the model pipeline fitted on the training set only.
- **Rationale:** 72 levels, many very rare; the top 10 cover ~86% of the non-missing values. Choosing the levels on the full dataset would leak information from the test set.
- **Alternatives considered:** Manual grouping by clinical area (e.g. surgery, internal medicine).
- **Impact:** To be measured on the training set.
- **Update (2026-10-19), implementation:** the rule is applied to **every** categorical feature, not only `medical_specialty`: at most the 10 most frequent levels with at least 100 training rows are kept (`OneHotEncoder`/`OrdinalEncoder` with `min_frequency=100`, `max_categories=11`, `src/preprocessing.py`), fitted inside the pipeline. The rest, and any level unseen in training, go to one infrequent level, not to `"Other"`, which is already a real value of `race`. `gender` keeps all its levels (project brief).
- **Evidence (2026-10-19):** `DC_Models/01_cleaning_variants.ipynb`: the alternative against the proposals (variant P), same 5 patient-grouped folds on the development patients (test patients excluded), logistic regression (LR) and gradient boosting (GB), scored on the same encounters. Reference P: PR-AUC 0.2129 (LR), 0.2256 (GB). Decision rule: better/worse only with at least 4 folds of 5; a difference matters from 0.005. Alternative A3 (fixed grouping by clinical area, `SPECIALTY_GROUPS` in `src/features.py`, 15 areas): LR +0.0012 (better, 5 of 5 folds), GB −0.0010 (worse, 4 folds lost); on first encounters +0.0012 / +0.0006. Mixed and below 0.005: a tie. **Recommendation: keep the proposal by default**; A3 is a defensible option for a linear final model (readable groups, and it keeps Oncology/Hematology, about 20% `<30`, that the top-10 rule merges). Caveat: the mapping was written from the level names after seeing a table of rates by specialty on all the data; target rates are looked at on training rows only from now on.

## D-010 — Train/test split strategy
- **Date:** 2026-10-05 · **Status:** Proposed (option A; was Open)
- **Problem:** `patient_nbr` repeats (71,518 patients for 101,766 encounters). A random split by encounter would put the same patient in both train and test and inflate the metrics.
- **Options:**
  - **A — All encounters + grouped split on `patient_nbr`** (`GroupShuffleSplit` for the test holdout, `GroupKFold` for cross-validation). Keeps all the data and matches the real use case (every discharge is scored).
  - **B — First encounter per patient only** (Strack et al.). Simpler and with independent rows, but loses ~30% of the encounters; `<30` rate 8.8% on the raw data. It also assumes that `encounter_id` gives the chronological order (see the limitation in D-004).
- **Proposal:** A.
- **Impact:** To be filled once decided.
- **Evidence (2026-10-19):** `DC_Models/01_cleaning_variants.ipynb`: the alternative against the proposals (variant P), same 5 patient-grouped folds on the development patients (test patients excluded), logistic regression (LR) and gradient boosting (GB), scored on the same encounters. Reference P: PR-AUC 0.2129 (LR), 0.2256 (GB). Decision rule: better/worse only with at least 4 folds of 5; a difference matters from 0.005. Alternative A5 (train on the first encounter per patient only): GB **−0.0111** PR-AUC on all encounters (0 folds won of 5) and −0.0049 even when scored on first encounters only, the population of option B; LR −0.0008. The later encounters are useful training data. **Recommendation: option A.**
- **Update (2026-10-19), implementation:** `split_by_patient()` (`src/features.py`): two `GroupShuffleSplit` cuts on `patient_nbr`, 60/20/20 train/validation/test of the patients, `random_state=42`; 59,814 / 19,727 / 19,802 encounters, `<30` rate 0.114 / 0.114 / 0.112. Both modelling streams (`LA_Models/`, `DC_Models/`) use the same split, so their final test scores are measured on the same patients.

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

## D-013 — Cleaning and split as shared, tested code in `src/`
- **Date:** 2026-10-19 · **Status:** Accepted · **Implemented in:** `src/features.py`, `tests/test_src.py`
- **Decision:** The row-by-row cleaning (`clean()`), the feature lists and the patient-grouped split (`split_by_patient()`, `first_encounter_only()`) live in `src/features.py`; every notebook imports them. The alternatives of Step 1 are options of `clean()`, whose defaults are the proposals.
- **Rationale:** Project rule (shared code lives in `src/`, notebooks import it), course structure (`src/features.py`, `tests/`) and project plan. One implementation means both work streams clean the data identically.
- **Impact:** 99,343 encounters of 69,990 patients, `<30` rate 11.39% with the defaults. 17 unit tests (`python -m unittest`) check the rules, including the leakage ones (no patient in two sets, no identifier or target among the features).

## D-014 — Every learned step lives inside a scikit-learn `Pipeline`
- **Date:** 2026-10-19 · **Status:** Accepted · **Implemented in:** `src/preprocessing.py`
- **Decision:** Rare-level grouping, encoding and scaling are scikit-learn transformers (`make_preprocessor`), combined with the model in one `Pipeline` (`make_pipeline`). Three variants: `linear` (scaled numbers, one-hot), `onehot` (raw numbers, one-hot), `ordinal` (integer codes for native categorical support); CatBoost and EBM read the categories natively.
- **Rationale:** Cloned and refitted automatically on the fitting rows of every split, cross-validation fold and hyperparameter search, so nothing learned can come from the rows being scored. One serialisable object, as the project plan asks (`pipeline.joblib`).
- **Impact:** Checked against an independent hand-written preprocessing: identical validation scores (PR-AUC 0.210 logistic regression, 0.224 gradient boosting).

## D-015 — Model selection protocol
- **Date:** 2026-10-19 · **Status:** Proposed · **Implemented in:** `DC_Models/02_model_exploration.ipynb`
- **Decision:** (1) Hyperparameters tuned by random search on `StratifiedGroupKFold(5)` of the **training** rows, with `groups=patient_nbr`. (2) Tuned candidates compared on the validation set. (3) The best 3 plus the logistic regression baseline re-scored fold by fold on train + validation; a model beats a simpler one only if it wins at least 4 of 5 folds, otherwise the simpler is preferred. (4) The chosen pipeline refitted on train + validation and scored **once** on test (locked behind `OPEN_TEST`).
- **Rationale:** Validation is never used for tuning, so its scores are not optimised directly; the paired fold-by-fold rule separates real gains from the luck of one sample of patients.
- **Alternatives considered:** Nested cross-validation (D-021), more rigorous and about five times slower.
- **Limitations:** The shortlist is chosen on validation and step (3) includes validation, so the step-(3) scores are slightly optimistic; the single test evaluation measures by how much.
- **Update (2026-10-20):** (3b) The shortlist is also refitted with 5 seeds on train and scored on validation, so that a gap is not mistaken for the luck of one seed (D-030). (4) The test set scores the chosen model with **both** calibration options of D-022; the primary option is declared (`PRIMARY`) from the rehearsal on validation **before** the test is opened, and the notebook refuses to open it otherwise. Both rows are logged; nothing is chosen after.

## D-016 — Evaluation metrics and class imbalance
- **Date:** 2026-10-19 · **Status:** Accepted · **Implemented in:** `src/evaluation.py`
- **Decision:** Main metric PR-AUC (average precision); also ROC-AUC and precision/recall on the 10% highest-risk encounters (the follow-up capacity); threshold metrics, log loss and Brier score are logged for reference. Imbalance handled with balanced class weights (or `scale_pos_weight`), never with resampling.
- **Rationale:** Project plan (PR-AUC, top-K, class weights rather than naive oversampling). With 11% positives, accuracy rewards the model that flags nobody (0.886).
- **Impact:** A random ranking scores PR-AUC ≈ 0.114 (the validation `<30` rate); results are also shown as lift over that rate.

## D-017 — No random internal splits (early stopping, Keras, stacking)
- **Date:** 2026-10-19 · **Status:** Accepted · **Implemented in:** `DC_Models/02_model_exploration.ipynb`, `src/models.py`
- **Decision:** Early stopping is off in every boosting library and network: the number of trees or epochs is a hyperparameter tuned by the grouped search. The Keras wrapper trains for a fixed number of epochs (no `validation_split`). Stacking (`StackingClassifier`) is not used; ensembles are soft-voting averages.
- **Rationale:** Each of these holds out rows **by encounter** (a random 10%, the last rows of the array, or non-grouped internal folds), which puts encounters of the same patient on both sides: patient-level leakage inside the training set, and an optimistic stopping point or meta-model.
- **Alternatives considered:** An early-stopping set cut with `GroupShuffleSplit` from the fitting rows of each fold; a stacking meta-model trained on grouped out-of-fold predictions. Both possible, not needed so far.

## D-018 — XGBoost AFT on interval-censored times
- **Date:** 2026-10-19 · **Status:** Accepted · **Implemented in:** `src/models.py` (`XGBAFTClassifier`)
- **Decision:** The accelerated-failure-time model is trained on interval-censored times built from the binary target: T in [0, 30] days if `<30`, T in [30, +∞) otherwise (`>30` and `NO` alike). It returns P(T ≤ 30) = F((log 30 − f(x)) / σ); the distribution (normal, logistic, extreme) and σ are tuned. Boosting starts at 30 days (`base_score`), with `max_delta_step=0.7`.
- **Rationale:** The dataset has no readmission dates, so with this information the AFT is a binary classifier with another link function; its time interpretation can still be useful in the report. With XGBoost's default start (0.5 days) and σ < 1 the training **diverged** (every predicted log-time ≈ −3000, PR-AUC 0.114 = random); a unit test now guards against it.
- **Impact:** Tuned AFT: validation PR-AUC 0.223, the best top-10% recall of all models (0.239).

## D-019 — Choice of the model
- **Date:** 2026-10-19 · **Status:** Proposed
- **Decision:** Tuned **XGBoost** as the production model; the soft-voting ensemble (XGBoost + AFT + Keras) reported as the challenger; **EBM** as the interpretable alternative for clinicians.
- **Rationale:** `DC_Models/02_model_exploration.ipynb`, validation PR-AUC: ensemble 0.229, XGBoost 0.226, scikit-learn gradient boosting 0.224, AFT 0.223, CatBoost / LightGBM / Keras 0.222, EBM 0.221, logistic regression 0.210, random forest 0.208, extra trees 0.202 (random: 0.114). Fold by fold on train + validation: ensemble 0.230 ± 0.003, XGBoost 0.228 ± 0.004, gradient boosting baseline 0.226 ± 0.005, logistic regression 0.213 ± 0.002. The ensemble wins 5 of 5 folds against XGBoost, but by 0.0024 on average, below the 0.005 that matters, for three models to calibrate, explain and serve. Boosting beats the logistic regression in every fold (+0.013).
- **Alternatives considered:** The ensemble as production model; EBM (0.005 behind XGBoost on validation, fully readable); the logistic regression (simplest, clearly weaker).
- **Next:** Calibration and capacity tiers on validation (D-022) before the single test evaluation. To be compared with the final model of `LA_Models/` on the same test patients.

## D-030 — Audit against scikit-learn's "Common pitfalls and recommended practices"
- **Date:** 2026-10-20 · **Status:** Accepted · **Implemented in:** `src/calibration.py`, `src/models.py` (`with_seed`), `DC_Models/02_model_exploration.ipynb`
- **Decision:** The code was checked against chapter 12 of the scikit-learn user guide (inconsistent preprocessing, data leakage, controlling randomness). What holds, and what was fixed:
  - **12.1 Inconsistent preprocessing:** holds. Every learned step is inside the pipeline (D-014), refitted with the model. **Fixed:** the calibration planned in Stage 6 did not reach the test (Stage 7 scored an uncalibrated refit), and a calibrator learned for the train-only model would not match a model refitted on train + validation: two consistent options, D-022.
  - **12.2 Data leakage:** holds for every split, fold and search (grouped by patient, D-010, D-014, D-017); the test set is still closed. Three looks at data that include the test patients, all small: (a) the rare-medication list was counted on all encounters (D-007, update); (b) `LA_Models/01_preprocessing_and_models.ipynb`, cell 10, shows the `<30` rate by ID code on all encounters, test patients included (a descriptive table of the decoded codes, used to argue that they are categories, not numbers; no rule or threshold is taken from it); (c) the clinical-area mapping of A3 was written after seeing rates by specialty on all the data (already stated in `01_cleaning_variants`). From now on target rates are looked at on training rows only.
  - **12.3 Controlling randomness:** splitters receive an integer seed (same folds for every model and search, so fold-by-fold comparisons are valid); estimators receive the integer `SEED` (reproducible runs). **Fixed:** the elastic-net logistic regression (`saga`, which shuffles the rows) had no `random_state`, so its score changed from run to run. **Added:** the scikit-learn guide warns that with one integer seed a model may look good because of that seed; the shortlist of 5.12 is now refitted with 5 seeds. Results: RESULTS_PLACEHOLDER
- **Alternatives considered:** Passing a `RandomState` instance to the estimators, as the guide suggests for robust cross-validation. Not adopted: `clone` deep-copies the instance (scikit-learn 1.9), so every fold would start from the same state anyway, and XGBoost, CatBoost and Keras take integer seeds only. The multi-seed check gives the same information explicitly.
- **Known limits:** `KerasMLPClassifier` calls `keras.utils.set_random_seed`, which also sets NumPy's global seed; harmless as long as every other estimator gets an explicit `random_state`. TensorFlow on a multi-threaded CPU is not bit-for-bit deterministic, so the Keras model and the ensemble can move in the fourth decimal between runs.

## Research directions

The entries below are **Open**: ideas to make the work more rigorous or to go beyond the project plan, ordered by expected value. Each one states how it would be done without leakage.

## D-020 — Statistical comparison of models with patient-level uncertainty
- **Date:** 2026-10-19 · **Status:** Open
- **Problem:** Differences of 0.002–0.005 PR-AUC between models are within the noise of one sample of patients, and the encounters of one patient are not independent.
- **Options:** (a) **Cluster bootstrap by patient** on the validation (and, once, the test) predictions: resample patients with replacement, recompute every metric and the **paired difference** between two models, report 95% intervals. (b) **Repeated grouped cross-validation** (e.g. 5 × 5 `StratifiedGroupKFold` with different seeds) with the **corrected resampled t-test** of Nadeau & Bengio (2003), which accounts for the overlap of the training sets. (c) For ROC-AUC, DeLong's test only with a cluster correction (Obuchowski 1997).
- **Expected impact:** Turns "the ensemble wins by 0.0024" into a statement with an interval; likely shows that most boosting libraries are statistically tied.

## D-021 — Nested grouped cross-validation
- **Date:** 2026-10-19 · **Status:** Open
- **Problem:** Selecting the model on the same data used to score it is optimistic (D-015, limitations).
- **Options:** An outer `StratifiedGroupKFold` on train + validation; inside each outer fold, the whole search and choice (D-015, steps 1–3) on the inner folds; the outer fold scores the *procedure*, not one model (Varma & Simon 2006; Cawley & Talbot 2010). Bayesian optimisation (e.g. Optuna, TPE) instead of random search, with the same grouped folds, to keep the cost acceptable.
- **Expected impact:** An unbiased estimate of the performance of the whole selection procedure, to compare with the single test score.

## D-022 — Calibration, decision curve and capacity tiers
- **Date:** 2026-10-19 · **Status:** Open (planned, Phase 2 of the project plan)
- **Options:** Calibrate the chosen pipeline on validation (`CalibratedClassifierCV(FrozenEstimator(...))`: isotonic, Platt/sigmoid; or Venn–Abers predictors, which give calibrated intervals). Assess with the calibration hierarchy of Van Calster et al. (2019): calibration-in-the-large, calibration slope and intercept, flexible reliability curves, plus the Brier score and its decomposition. **Decision curve analysis** (Vickers & Elkin 2006): net benefit against the threshold probability, compared with "follow up everyone" and "nobody". High / Medium / Low tiers set by capacity (top 10%, next 20%), with expected readmissions caught per tier and its patient-bootstrap interval.
- **Leakage rule:** Calibration and cut-offs learned on validation (or out-of-fold predictions), never on test.
- **Update (2026-10-20), status Proposed for calibration and tiers:** The calibrator and the cut-offs must also be learned **for the model that is scored**: the plan of `02` calibrated the train-only model on validation, while Stage 7 refitted an uncalibrated pipeline on train + validation (D-030). Two consistent options, implemented in `src/calibration.py` and Stage 6 of `DC_Models/02_model_exploration.ipynb`: **A, holdout**: model fitted on train, calibrator and cut-offs from its predictions on validation (`CalibratedClassifierCV(FrozenEstimator(...))`); **B, out of fold**: model refitted on train + validation, calibrator and cut-offs from its grouped out-of-fold predictions (`CalibratedClassifierCV(ensemble=False)`, folds cut by patient). Both are rehearsed with validation in the role of the test (A: 75 / 25 of the training patients; B: out of fold on train), then both are scored once on test (D-015). Isotonic calibration, tiers High = top 10%, High + Medium = top 30%. Rehearsal results: RESULTS_PLACEHOLDER

## D-023 — Ranking objectives aligned with the follow-up capacity
- **Date:** 2026-10-19 · **Status:** Open
- **Problem:** The hospital uses the **order** of the patients at a fixed capacity, while the models minimise a log loss over all patients.
- **Options:** (a) Tune on the metric of use: `scoring` = precision or recall in the top 10% instead of PR-AUC. (b) Learning-to-rank objectives (XGBoost `rank:pairwise` / `rank:ndcg`, LightGBM `lambdarank`), with one query group per cross-validation fold or per admission period. (c) Cost-sensitive losses (focal loss for the network).
- **Expected impact:** Possibly a better top-10% precision at the same PR-AUC; scores then need calibration (D-022) to be read as risks.

## D-024 — Clinically grounded features and a LACE baseline
- **Date:** 2026-10-19 · **Status:** Open
- **Options:** (a) **LACE index** (van Walraven et al. 2010: Length of stay, Acuity of admission, Comorbidity, Emergency visits), a validated readmission score, computable row by row from `time_in_hospital`, `admission_type_id`, the diagnoses and `number_emergency`: a **clinical baseline** that every model must beat, and an argument for the report. (b) **Charlson / Elixhauser comorbidity** flags from the ICD-9 codes (coding algorithms of Quan et al. 2005), richer than the 9 Strack groups, limited by the 3 recorded diagnoses. (c) **HCUP CCS** categories as a finer, standard grouping of the diagnoses. (d) The interaction found by Strack et al. (2014): HbA1c measured × primary diagnosis.
- **Leakage rule:** All row-by-row, fixed mappings: safe in `clean()` before the split (decision log entry each).

## D-025 — Using the three-level outcome
- **Date:** 2026-10-19 · **Status:** Open
- **Idea:** `>30` is a later readmission, closer to `<30` than `NO`. An **ordinal** model (cumulative link, or an ordinal boosting objective) or a **multi-task** network (P(<30) and P(any readmission)) could use it as an auxiliary signal; the score stays P(<30).
- **Leakage rule:** The raw `readmitted` column becomes an auxiliary *target* of the training rows only, never a feature (D-011).

## D-026 — Patient-history features from earlier encounters
- **Date:** 2026-10-19 · **Status:** Open (likely to be rejected)
- **Idea:** Number of the patient's earlier encounters in the dataset, earlier `<30` outcomes, time since the previous encounter.
- **Why it is risky:** `encounter_id` is not reliably chronological (D-004, limitations) and there are no dates; an earlier `<30` label is only known 30 days after that discharge, and it is often the current encounter itself. Computed on the whole file, these features read the future. The history known at discharge is already in `number_inpatient`, `number_emergency`, `number_outpatient`.
- **Only acceptable with:** features computed from strictly earlier encounters, and a pseudo-temporal validation (train on lower `encounter_id`, test on higher) as a robustness check.

## D-027 — Conformal prediction for risk with guarantees
- **Date:** 2026-10-19 · **Status:** Open
- **Idea:** Split conformal prediction on a calibration set of **patients** (grouped, so that exchangeability holds between patients) gives prediction sets with guaranteed coverage; **Mondrian** (group-conditional) conformal gives the guarantee within each `race` or `age` group. Conformal risk control could bound the share of readmissions missed by the High tier.
- **Expected impact:** A statement of uncertainty per patient ("this risk is uncertain"), useful for the clinicians' worklist and for the fairness audit.

## D-028 — Fairness audit
- **Date:** 2026-10-19 · **Status:** Open (planned, Phase 5 of the project plan)
- **Options:** By `race`, `age` band and `gender` (without the 3 `Unknown/Invalid` encounters, project brief), with patient-bootstrap intervals: PR-AUC; **calibration within groups**; **equal opportunity at the capacity cut-off** (share of each group's readmissions that reach the top 10%); share of each group in the High tier. Intersections (race × age) where the counts allow. State the trade-offs: calibration within groups and equal error rates cannot all hold when base rates differ (Kleinberg et al. 2016; Chouldechova 2017).
- **Leakage rule:** Group metrics on validation during development, on test once at the end.

## D-029 — Explainability, robustness and data limits
- **Date:** 2026-10-19 · **Status:** Open
- **Options:** (a) **SHAP** values of the chosen model (global importance, per-patient reasons for the worklist, interaction values), checked for **stability across folds**, and compared with the shape functions of EBM: features that both rank high are the robust findings. (b) **Subgroup and shift robustness:** performance by admission source, admission type and payer, and on a pseudo-temporal split by `encounter_id`. (c) **Learning curves** (score against the number of training patients): if the curve is flat, more data of the same kind will not help, and the ceiling (~0.68 ROC-AUC) comes from the features. (d) **Label limits:** readmissions to hospitals outside the network are recorded as `NO`, so the negatives contain unobserved positives (a positive-unlabelled setting); and the model is correlational (project brief): it ranks risk, it does not say that follow-up prevents readmission.
