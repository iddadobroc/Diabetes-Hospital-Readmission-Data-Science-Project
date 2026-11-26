"""Scoring with the published model: risk, rank, tier and reasons of each discharge.

This module is small on purpose: the Hugging Face app imports it (with `features.py`
and `ibcs.py`) to score new discharges with the very same code as the notebooks and
`train.py`. It needs only NumPy, pandas and XGBoost.
"""

import numpy as np
import pandas as pd


def raw_scores(calibrated, X):
    """The uncalibrated score of a model calibrated with `src/calibration.py`.

    Both calibration options keep one fitted model inside `calibrated_classifiers_[0]`
    (a `FrozenEstimator` for A, the model refitted on all the rows for B). Its score
    orders the encounters exactly like the calibrated risk, without ties, so the
    capacity cut-offs are applied to it (D-022).

    Args:
        calibrated (CalibratedClassifierCV): A fitted model with `ensemble=False` or
            a frozen estimator.
        X (pandas.DataFrame): The encounters to score.

    Returns:
        numpy.ndarray: One raw score per row.
    """
    return calibrated.calibrated_classifiers_[0].estimator.predict_proba(X)[:, 1]


def inner_pipeline(calibrated):
    """The fitted preprocessing + XGBoost pipeline inside a calibrated model."""
    inner = calibrated.calibrated_classifiers_[0].estimator
    return getattr(inner, "estimator", inner)  # option A wraps it in FrozenEstimator


def assign_tiers(raw, cutoffs):
    """The tier of each encounter from its raw score and the capacity cut-offs.

    Args:
        raw (array-like): Raw scores (`raw_scores`).
        cutoffs (dict): {"High": c1, "Medium": c2} with c1 >= c2, on raw scores.

    Returns:
        numpy.ndarray: "High", "Medium" or "Low" for each encounter.
    """
    raw = np.asarray(raw)
    return np.where(
        raw >= cutoffs["High"],
        "High",
        np.where(raw >= cutoffs["Medium"], "Medium", "Low"),
    )


def contributions(pipe, X, features):
    """TreeSHAP contributions of each original feature to the raw score (log-odds).

    XGBoost computes them natively (`pred_contribs=True`, the same values as
    `shap.TreeExplainer`). The one-hot columns of a feature are summed back into it,
    so each encounter gets one value per feature in `features`: positive values push
    the risk up, negative values down. Calibration is monotone, so the direction
    holds for the calibrated risk too.

    Args:
        pipe (Pipeline): A fitted final pipeline ("prep" + XGBoost "model").
        X (pandas.DataFrame): The encounters, with the feature columns.
        features (list[str]): The original feature names.

    Returns:
        pandas.DataFrame: One row per encounter, one column per feature, plus "bias"
        (the average log-odds of the model).
    """
    import xgboost as xgb

    encoded = pipe.named_steps["prep"].transform(X)
    booster = pipe.named_steps["model"].get_booster()
    values = booster.predict(xgb.DMatrix(encoded), pred_contribs=True)
    columns = list(encoded.columns)
    # Each encoded column belongs to the longest feature name it starts with
    owner = {}
    for col in columns:
        matches = [f for f in features if col == f or col.startswith(f + "_")]
        owner[col] = max(matches, key=len)
    frame = pd.DataFrame(values[:, :-1], columns=columns, index=X.index)
    out = frame.T.groupby(pd.Series(owner)).sum().T[features]
    out["bias"] = values[:, -1]
    return out


def prepare_discharges(rows, typical, mappings):
    """Turn raw discharge rows (UCI format) into the features of the model.

    The app and the tests build discharges from a form or from an uploaded CSV. A
    column that is absent takes its typical value (computed on the fitting rows by
    `train.py`); a value that is present but missing stays missing, because missing
    values are informative levels ("Unknown", D-008). Then the same code as the
    training data runs: the ID descriptions (`add_mapping_descriptions`) and the
    derived features (`prepare_features`).

    Args:
        rows (pandas.DataFrame): One row per discharge, raw UCI columns.
        typical (dict): Column -> typical value (`config.json`, "typical").
        mappings (dict): ID column -> {code (int): description}.

    Returns:
        pandas.DataFrame: The rows with every feature column, ready for the model.
    """
    from data_prep import ID_COLUMNS, add_mapping_descriptions
    from features import prepare_features

    rows = rows.copy()
    for col, value in typical.items():
        if col not in rows.columns:
            rows[col] = value
    for col in ID_COLUMNS:
        rows[col] = rows[col].astype(int)
    for col in ("diag_1", "diag_2", "diag_3"):
        rows[col] = rows[col].map(lambda v: v if pd.isna(v) else str(v))
    return prepare_features(add_mapping_descriptions(rows, mappings))
