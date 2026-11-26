"""Calibration and capacity tiers: from a ranking score to a risk the hospital can use.

With balanced class weights the predicted probabilities are inflated (the model is
trained as if half the encounters were readmitted), so a score of 0.6 is not a 60%
risk. Calibration maps the scores to observed rates; the capacity cut-offs turn them
into the worklist (the top 10% are called, and so on).

The calibrator and the cut-offs are learned from the data, so they must come from
predictions on patients the model was not fitted on, and the model that is scored must
be the one they were learned for (decision log, D-022). Two ways to do it:

| option | the model is fitted on | the calibrator and the cut-offs learn from |
|---|---|---|
| A `holdout_calibration` | the fitting rows | its predictions on other patients |
| B `out_of_fold_calibration` | all the rows | grouped out-of-fold predictions |

A calibrates exactly the model that is scored, but fits it on fewer rows. B fits the
model on every row; its calibrator learns from the fold models, each fitted on 4/5 of
the rows, a close stand-in for the final model.

The capacity cut-offs are taken on the RAW score of the model, not on the calibrated
risk. Isotonic calibration is a step function: thousands of encounters share the same
calibrated risk, so "the top 10%" of calibrated risks can hold far more than 10% of the
encounters. The raw score has the same order (calibration is monotone) and almost no
ties, so a cut-off on it flags the intended share; the calibrated risk is what is shown.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import cross_val_predict


def holdout_calibration(
    pipe, X_fit, y_fit, X_cal, y_cal, method="isotonic", return_raw=False
):
    """Option A: fit the model on some patients, calibrate it on other patients.

    Args:
        pipe (Pipeline): The unfitted pipeline (preprocessing + model).
        X_fit, y_fit: The rows the model is fitted on.
        X_cal, y_cal: The rows of OTHER patients the calibrator is fitted on.
        method (str): "isotonic" or "sigmoid" (Platt).
        return_raw (bool): Also return the raw scores on the calibration rows.

    Returns:
        tuple: `(calibrated, reference)`: the calibrated model, and its calibrated
        predictions on the calibration rows. With `return_raw`, a third element: the
        raw scores of the same rows, from which the capacity cut-offs are taken.
    """
    model = clone(pipe).fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(X_cal, y_cal)
    reference = calibrated.predict_proba(X_cal)[:, 1]
    if return_raw:
        return calibrated, reference, model.predict_proba(X_cal)[:, 1]
    return calibrated, reference


def out_of_fold_calibration(
    pipe, X, y, groups, cv, method="isotonic", return_raw=False
):
    """Option B: fit the model on all the rows, calibrate it out of fold.

    `CalibratedClassifierCV(ensemble=False)` collects the out-of-fold scores of
    `pipe` (one fit per fold), fits the calibrator on them, then fits `pipe` once on
    all the rows. The folds are cut by patient beforehand, because the calibrator
    does not pass `groups` to the splitter.

    Args:
        pipe (Pipeline): The unfitted pipeline (preprocessing + model).
        X, y: All the rows.
        groups (array-like): The patient of each row.
        cv: A grouped splitter, such as `StratifiedGroupKFold`.
        method (str): "isotonic" or "sigmoid" (Platt).
        return_raw (bool): Also return the raw out-of-fold scores.

    Returns:
        tuple: `(calibrated, reference)`: the calibrated model, and the calibrated
        out-of-fold predictions on the rows. With `return_raw`, a third element: the
        raw out-of-fold scores, from which the capacity cut-offs are taken.
    """
    splits = list(cv.split(X, y, groups=groups))
    calibrated = CalibratedClassifierCV(pipe, method=method, cv=splits, ensemble=False)
    calibrated.fit(X, y)
    # The same out-of-fold scores the calibrator was fitted on: CalibratedClassifierCV
    # uses decision_function when the model has one, predict_proba otherwise
    response = (
        "decision_function" if hasattr(pipe, "decision_function") else "predict_proba"
    )
    raw = cross_val_predict(pipe, X, y, cv=splits, method=response)
    if raw.ndim == 2:
        raw = raw[:, 1]
    calibrator = calibrated.calibrated_classifiers_[0].calibrators[0]
    if return_raw:
        return calibrated, calibrator.predict(raw), raw
    return calibrated, calibrator.predict(raw)


def raw_scores(calibrated, X):
    """The uncalibrated score of a model returned by the two functions above.

    Both options keep one fitted model inside `calibrated_classifiers_[0]` (a
    `FrozenEstimator` for A, the model refitted on all the rows for B). Its score
    orders the encounters exactly like the calibrated risk, without ties.

    Args:
        calibrated (CalibratedClassifierCV): A fitted model with `ensemble=False` or
            a frozen estimator, as returned by `holdout_calibration` or
            `out_of_fold_calibration`.
        X (pandas.DataFrame): The encounters to score.

    Returns:
        numpy.ndarray: One raw score per row.
    """
    return calibrated.calibrated_classifiers_[0].estimator.predict_proba(X)[:, 1]


def capacity_cutoffs(reference, shares):
    """The score above which each share of the reference encounters is flagged.

    Args:
        reference (array-like): Scores on rows the model was not fitted on. Pass the
            RAW scores (`return_raw=True` of the two calibration functions): on
            isotonic risks, ties make the flagged share larger than intended.
        shares (dict): Tier name -> share of the encounters the hospital can follow
            up, such as {"High": 0.10}.

    Returns:
        dict: Tier name -> cut-off. An encounter is in the tier if its score is at
        least the cut-off.
    """
    return {
        name: float(np.quantile(reference, 1 - share)) for name, share in shares.items()
    }


def tier_report(y, p, cutoffs, score=None):
    """What each cut-off does on a set of encounters, plus a row for all of them.

    Args:
        y (array-like): True labels, 0 or 1.
        p (numpy.ndarray): Calibrated predicted risks.
        cutoffs (dict): Tier name -> cut-off, from `capacity_cutoffs`.
        score (numpy.ndarray, optional): The scores the cut-offs were taken on (the
            raw scores); if omitted, the cut-offs are compared with `p`.

    Returns:
        pandas.DataFrame: One row per tier: the cut-off, the share of encounters
        flagged (to compare with the intended share), the observed `<30` rate among
        them (precision), their mean predicted risk (equal to the observed rate if the
        risks are calibrated) and the share of all readmissions they catch (recall).
    """
    y, p = np.asarray(y), np.asarray(p)
    score = p if score is None else np.asarray(score)
    rows = {}
    for name, cutoff in ({"All encounters": -np.inf} | cutoffs).items():
        flagged = score >= cutoff
        rows[name] = {
            "cutoff": cutoff,
            "share_flagged": flagged.mean(),
            "observed_rate": y[flagged].mean(),
            "mean_risk": p[flagged].mean(),
            "recall": y[flagged].sum() / y.sum(),
        }
    return pd.DataFrame(rows).T
