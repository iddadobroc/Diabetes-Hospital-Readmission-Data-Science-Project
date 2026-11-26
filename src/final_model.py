"""The final model: one recipe shared by `DC_Models/03_final_model` and `train.py`.

Everything that defines the deployed model lives here, so that the notebook that
measures it and the pipeline that publishes it build exactly the same object:

- the tuned XGBoost hyperparameters (D-019, chosen in `DC_Models/02_model_exploration`);
- the calibration option declared before the test set was opened (D-022, D-032);
- the follow-up capacity that defines the risk tiers (D-022);
- the quality gate agreed before the test set was opened (D-032);
- the clinical baseline every model must beat: the LACE index (D-024).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from calibration import holdout_calibration, out_of_fold_calibration
from features import GROUP
from preprocessing import make_pipeline
from scoring import (  # noqa: F401  (re-exported for the notebook and train.py)
    assign_tiers,
    contributions,
    inner_pipeline,
    raw_scores,
)

SEED = 42

# Best of the 15-combination random search of `02_model_exploration` (5.5), on the
# grouped CV of the training rows: CV PR-AUC 0.2248, validation PR-AUC 0.2263. The exact
# values were recovered with ParameterSampler(space, n_iter=15, random_state=42).
FINAL_PARAMS = {
    "n_estimators": 580,
    "learning_rate": 0.03738105868191797,
    "max_depth": 5,
    "min_child_weight": 6.938901412739395,
    "subsample": 0.672894435115225,
    "colsample_bytree": 0.8421165132560784,
    "reg_lambda": 0.2505631890506232,
    "scale_pos_weight": 4.519545468159167,
}

# Follow-up capacity (D-022): the hospital calls the riskiest 10% of the discharges
# (High), and sends a reminder to the next 20% (Medium). Shares are cumulative.
CAPACITY = {"High": 0.10, "Medium": 0.30}

# Calibration option used for the deployed model (D-022): "A" (holdout) or "B" (out
# of fold). Set from the rehearsal on validation, BEFORE the test set is opened (D-032):
# B had the lower validation Brier score (0.0963 against 0.0968; the gap is under 0.001,
# so B by the tie rule too), calibration slope 0.99 and a High tier of 9.98%.
PRIMARY = "B"

# Quality gate (D-032), written before the test set is opened: the published model
# must meet every line on the test patients, or the pipeline stops.
QUALITY_GATE = {
    "min_pr_auc": 0.20,  # at least 0.20 (a random ranking scores ~0.11)
    "min_gain_over_lace": 0.01,  # PR-AUC at least 0.01 above the LACE baseline
    "min_recall_top10": 0.20,  # the top 10% must hold at least 20% of readmissions
    "max_calibration_gap_pp": 1.0,  # |mean risk - observed rate| at most 1 pp
}


def build_pipeline(seed=SEED):
    """The unfitted final pipeline: one-hot preprocessing + tuned XGBoost.

    XGBoost runs on ONE thread (D-034): with several threads the histogram sums are
    added in an order that depends on the number of threads and on the CPU, so a
    laptop and the GitHub runner built slightly different models (test PR-AUC 0.2390
    against 0.2379). One thread gives the same model on every machine, so the model
    published by the pipeline is the one measured in `03_final_model`.
    """
    model = XGBClassifier(
        **FINAL_PARAMS, tree_method="hist", random_state=seed, n_jobs=1
    )
    return make_pipeline("onehot", model)


def grouped_folds(seed=SEED):
    """The patient-grouped, stratified 5 folds used for every CV of the project."""
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)


def fit_calibrated(option, X, y, groups, seed=SEED, calibration_share=0.25):
    """Fit the final pipeline and its calibrator with option A or B (D-022).

    Args:
        option (str): "A": fit the model on 75% of the patients and calibrate it on the
            other 25%; "B": fit the model on all the rows and calibrate it on its
            grouped out-of-fold scores.
        X (pandas.DataFrame): The features of the fitting rows.
        y (pandas.Series): The target of the fitting rows.
        groups (pandas.Series): The patient of each row.
        seed (int): Seed of the model and of the patient cut.
        calibration_share (float): Share of the patients held out for option A.

    Returns:
        tuple: `(model, reference_risk, reference_raw)`: the calibrated model, and the
        calibrated risks and raw scores of the rows the calibrator learned from (the
        capacity cut-offs are taken on the raw scores).
    """
    pipe = build_pipeline(seed)
    if option == "B":
        return out_of_fold_calibration(
            pipe, X, y, groups, grouped_folds(seed), return_raw=True
        )
    if option == "A":
        from sklearn.model_selection import GroupShuffleSplit

        cut = GroupShuffleSplit(
            n_splits=1, test_size=calibration_share, random_state=seed
        )
        fit_idx, cal_idx = next(cut.split(X, y, groups=groups))
        return holdout_calibration(
            pipe,
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            X.iloc[cal_idx],
            y.iloc[cal_idx],
            return_raw=True,
        )
    raise ValueError(f"option must be 'A' or 'B', not {option!r}")


RANKING_METRICS = ["pr_auc", "roc_auc", "precision_top10", "recall_top10"]


def evaluate(y, p, raw):
    """The metrics of the final model: ranking on the raw score, the rest on the risk.

    Isotonic calibration turns the score into a step function, so the calibrated risks
    have ties that blur the ranking (validation PR-AUC 0.220 instead of 0.228 with the
    same order). The worklist ranks by the raw score (D-022), so PR-AUC, ROC-AUC and the
    top-10% metrics are computed on it; log loss, Brier and the threshold metrics on the
    calibrated risk, which is what is shown to the user.

    Args:
        y (array-like): True labels, 0 or 1.
        p (numpy.ndarray): Calibrated risks.
        raw (numpy.ndarray): Raw scores of the same rows.

    Returns:
        dict: The keys of `evaluation.scores`.
    """
    from evaluation import scores

    out = scores(y, p)
    ranking = scores(y, raw)
    for key in RANKING_METRICS:
        out[key] = ranking[key]
    return out


# --- LACE index (D-024): the clinical baseline -----------------------------------
# van Walraven et al. (2010). L = length of stay, A = acute (emergency) admission,
# C = Charlson comorbidity index, E = emergency visits in the previous 6 months. The
# dataset has the visits of the previous YEAR, and only 3 diagnosis codes per
# encounter, so C and E are approximations; the score is a fixed rule, row by row.

# Charlson conditions on 3-digit ICD-9 codes (Quan et al. 2005, coarsened to 3 digits),
# with the Charlson weight of each condition.
CHARLSON = {
    "myocardial_infarction": ({410, 412}, 1),
    "heart_failure": ({428}, 1),
    "peripheral_vascular": ({440, 441, 443}, 1),
    "cerebrovascular": (set(range(430, 439)), 1),
    "dementia": ({290}, 1),
    "chronic_pulmonary": (set(range(490, 497)) | set(range(500, 506)), 1),
    "rheumatic": ({710, 714, 725}, 1),
    "peptic_ulcer": (set(range(531, 535)), 1),
    "mild_liver": ({70, 571}, 1),
    "hemiplegia": ({342, 344}, 2),
    "renal": ({582, 583, 585, 586, 588}, 2),
    "malignancy": (
        set(range(140, 173)) | set(range(174, 196)) | set(range(200, 209)),
        2,
    ),
    "severe_liver": ({572}, 3),
    "metastatic": (set(range(196, 200)), 6),
    "aids": ({42, 43, 44}, 6),
}
# A milder condition does not count when its severe form is present
HIERARCHY = {
    "mild_liver": "severe_liver",
    "malignancy": "metastatic",
}


def _icd9_parts(code):
    """(3-digit integer, decimal string) of a numeric ICD-9 code; None for V/E/NaN."""
    if not isinstance(code, str) or code.startswith(("V", "E")):
        return None
    whole, _, decimals = code.partition(".")
    return int(whole), decimals


def charlson_index(row):
    """The Charlson comorbidity index from the 3 diagnosis codes of one encounter."""
    found = {}
    for col in ("diag_1", "diag_2", "diag_3"):
        parts = _icd9_parts(row[col])
        if parts is None:
            continue
        whole, decimals = parts
        if whole == 250:
            # 250.4-250.7 complicated diabetes (2), the other 250.x uncomplicated (1)
            complicated = decimals[:1] in {"4", "5", "6", "7"}
            key = "diabetes_complicated" if complicated else "diabetes"
            found[key] = 2 if complicated else 1
            continue
        for name, (codes, weight) in CHARLSON.items():
            if whole in codes:
                found[name] = weight
    if "diabetes_complicated" in found:
        found.pop("diabetes", None)
    for mild, severe in HIERARCHY.items():
        if severe in found:
            found.pop(mild, None)
    return sum(found.values())


def lace_score(df):
    """The LACE index (0 to 19) of each encounter, from the prepared features.

    Args:
        df (pandas.DataFrame): Rows with `time_in_hospital`, `admission_type_desc`,
            `admission_source_desc`, `diag_1/2/3` and `number_emergency`.

    Returns:
        pandas.Series: The LACE score of each row.
    """
    days = df["time_in_hospital"].astype(int)
    length = np.select(
        [days < 1, days == 1, days == 2, days == 3, days <= 6, days <= 13],
        [0, 1, 2, 3, 4, 5],
        default=7,
    )
    acute = (
        (df["admission_type_desc"] == "Emergency")
        | (df["admission_source_desc"] == "Emergency Room")
    ).astype(int) * 3
    charlson = df.apply(charlson_index, axis=1)
    comorbidity = np.select(
        [charlson == 0, charlson == 1, charlson == 2, charlson == 3],
        [0, 1, 2, 3],
        default=5,
    )
    emergency = np.minimum(df["number_emergency"].astype(int), 4)
    return pd.Series(length + acute + comorbidity + emergency, index=df.index)


def _lace_column(df):
    """LACE as a one-column frame, for the baseline pipeline."""
    return lace_score(df).to_frame("lace")


def make_lace_model():
    """LACE as a model: a logistic regression on the LACE score alone.

    The regression only turns the score into a probability (it learns two numbers
    from the fitting rows); the ranking is the LACE ranking, so the PR-AUC and the
    top-10% metrics are those of the clinical score.
    """
    return Pipeline(
        [
            ("lace", FunctionTransformer(_lace_column)),
            ("model", LogisticRegression()),
        ]
    )


def lace_input_columns():
    """The raw columns `make_lace_model` reads."""
    return [
        "time_in_hospital",
        "admission_type_desc",
        "admission_source_desc",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_emergency",
    ]


def groups_of(df):
    """The patient of each row, for grouped splits and calibration."""
    return df[GROUP]
