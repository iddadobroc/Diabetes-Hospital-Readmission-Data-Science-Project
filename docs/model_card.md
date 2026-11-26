---
license: cc-by-4.0
library_name: sklearn
tags:
  - tabular-classification
  - xgboost
  - healthcare
  - readmission
  - calibration
datasets:
  - UCI Diabetes 130-US hospitals for years 1999-2008
metrics:
  - average_precision
  - roc_auc
  - brier_score
pipeline_tag: tabular-classification
---

# Diabetes readmission risk at discharge

A calibrated XGBoost pipeline that estimates, at the moment of hospital discharge, the
probability that a diabetic patient is readmitted as an inpatient within 30 days. Its
purpose is to **rank** discharges so that limited follow-up goes first to the patients
most likely to come back. NOVA IMS, Enterprise Data Science Bootcamp 2026.

Every claim below is a measured number; the decision log of the project
(`docs/decision_log.md` in the GitHub repository) records how each one was obtained.

## Intended use

- Prioritising post-discharge follow-up (calls, early visits) when the care team can
  reach only part of the discharges, with a clinician reviewing every list.
- Teaching and demonstration of an end-to-end, leakage-safe modelling workflow.

## Must not be used for

- Diagnosis, treatment, or any decision about a patient without human review.
- Denying, delaying or limiting care, or ranking patients for anything but extra help.
- Real patients' data: the model is trained on US encounters of 1999-2008.
- Causal claims: the model ranks an observed outcome; it does not show that follow-up
  prevents readmissions.

## Contents of this repository

| File | What |
|---|---|
| `model.joblib` | The pipeline: one-hot preprocessing, XGBoost (580 trees, depth 5), isotonic calibrator (scikit-learn 1.9.1, XGBoost 3.4.1 on one thread, so it rebuilds identically on any machine; load with `joblib.load`, only from this trusted repository) |
| `config.json` | Features, input levels, typical values, ID mappings, capacity cut-offs, library versions |
| `metrics.json` | Test metrics, baselines, tiers and the quality-gate values of this version |
| `sample_discharges.csv` | 2,000 test discharges in the raw UCI format, for the demo worklist |

The raw columns are turned into features by `prepare_features()` (`src/features.py`)
and `prepare_discharges()` (`src/scoring.py`) of the GitHub repository; the app on
Hugging Face uses the same code.

## Data

UCI Diabetes 130-US hospitals, 1999-2008 (Strack et al. 2014; CC BY 4.0): 101,766
inpatient encounters of 71,518 diabetic patients. The 2,423 encounters that ended in
death or hospice are excluded (no readmission can be observed, and they are not a
follow-up target), leaving 99,343. Positive class: readmitted in under 30 days (11.4%).

Patients, not encounters, are split 60 / 20 / 20 into train, validation and test, so no
patient is on two sides. The model of this repository is fitted on train + validation
(79,541 encounters); its isotonic calibrator learns from grouped out-of-fold scores.

## Results on the test patients

Measured once, on 19,802 encounters of 13,998 patients never used for fitting or
choices (2,216 readmissions, 11.2%).

| Ranking method | PR-AUC | ROC-AUC | Readmissions reached with capacity for 10% of discharges |
|---|---:|---:|---:|
| **This model** | **0.238** | **0.674** | **565 of 2,216** |
| Logistic regression | 0.218 | 0.667 | 523 |
| LACE index (clinical score) | 0.140 | 0.571 | 336 |
| Random order | 0.112 | 0.500 | 222 |

- **Calibration:** mean estimated risk 11.5% against 11.2% observed; slope 0.99; every
  risk decile within 1.8 pp of its observed rate; Brier score 0.094.
- **Tiers** (set by capacity, not by a fixed threshold): High = the 10% of discharges
  with the highest risk (observed rate 28.8%, 25% of all readmissions); Medium = the
  next 20%; High + Medium hold 52% of the readmissions.
- **Quality gate** (fixed before the test set was opened): PR-AUC at least 0.20, at
  least 0.01 above LACE, High-tier recall at least 0.20, calibration gap at most 1 pp.
  Passed on all four lines.
- **Validation** (before the test): PR-AUC 0.229 ± 0.001 over 5 seeds.

## Fairness

Share of each group's readmissions reached by the High tier on test (groups with at
least 500 encounters): Caucasian 26%, African American 21%; age under 50 34%, 50-59 27%,
60-69 22%, 70-79 25%, 80 and over 21%; female 27%, male 23% (all: 25%). The estimated
risk is within 1 pp of the observed rate in each of these groups. The intervals by
patient overlap (under 50: 23-46%, 80 and over: 17-25%), so the lower share for older
patients is a direction to watch, not a proven disparity. Groups under 500 encounters
(Asian, Hispanic, Other, Unknown race) are too small to assess; gender Unknown/Invalid
(3 encounters) is not reported.

## Limitations

- **Ranking, not certainty:** in the High tier 71% of the patients are not readmitted.
  The model tells the team where to look first, not who will come back.
- **Very high risks are uncertain:** above 60% the calibration rests on a few dozen
  test patients (43, of whom 58% were readmitted), and its last isotonic step reaches
  100% on 1 patient (not readmitted). The app shows such risks as "over 60%".
- **Labels:** readmissions to hospitals outside the network are recorded as "no", so
  some negatives are unobserved positives.
- **Era and population:** US hospitals, 1999-2008; practice and coding have changed.
- **Features:** 3 diagnosis codes per encounter, no lab values beyond HbA1c and glucose
  flags, no dates (`encounter_id` order is only approximately chronological).
- **Test set:** the same test patients had been scored once by an earlier model of the
  team (PR-AUC 0.232; decision log D-031). No choice of this model used that score, but
  the estimate is not perfectly blind.
- **Explanations** (TreeSHAP contributions shown by the app) describe associations in
  the data, not causes.
- **Serialization:** `model.joblib` is a pickle; load it only from this repository.

## Citation

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., &
Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates:
analysis of 70,000 clinical database patient records. *BioMed Research International*.
