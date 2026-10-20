"""Metrics shared by every modelling notebook, so all models are scored the same way.

`scores` is the function of Luis Alcaide's notebook (step 4.1), moved here: its
columns are the columns of `reports/experiments-readmission.csv`.
"""

import numpy as np
import pandas as pd
from sklearn import metrics


def scores(y, p, threshold=0.5, top=0.10):
    """Every metric for one set of true labels and predicted probabilities.

    Args:
        y (array-like): True labels, 0 or 1.
        p (numpy.ndarray): Predicted probabilities of the positive class.
        threshold (float): Probability above which a patient is flagged, for the
            threshold metrics (accuracy, precision, recall, f1).
        top (float): Share of the riskiest encounters "called" by the hospital, for
            `precision_top10` and `recall_top10`.

    Returns:
        dict: Metric name -> value. Ranking metrics (pr_auc, roc_auc, top-K) do not
        depend on `threshold`; log_loss and brier are lower-is-better.
    """
    y = np.asarray(y)
    flagged = p >= threshold
    k = int(np.ceil(top * len(p)))
    called = np.argsort(-p, kind="stable")[:k]
    return {
        "pr_auc": metrics.average_precision_score(y, p),
        "roc_auc": metrics.roc_auc_score(y, p),
        "precision_top10": y[called].mean(),
        "recall_top10": y[called].sum() / y.sum(),
        "accuracy": metrics.accuracy_score(y, flagged),
        "balanced_accuracy": metrics.balanced_accuracy_score(y, flagged),
        "precision": metrics.precision_score(y, flagged, zero_division=0),
        "recall": metrics.recall_score(y, flagged),
        "f1": metrics.f1_score(y, flagged, zero_division=0),
        "log_loss": metrics.log_loss(y, p),
        "brier": metrics.brier_score_loss(y, p),
    }


# Readable names of the metrics, for every table shown in the notebooks
LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "precision_top10": "Precision top 10%",
    "recall_top10": "Recall top 10%",
}


def leaderboard(log, base_rate, family=None):
    """The results log as a readable ranking: one row per model, best first.

    Args:
        log (pandas.DataFrame): The results log, one row per (model, dataset), with
            the columns of `scores` plus `model`, `family` and `dataset`.
        base_rate (float): Share of positives in the validation set: the PR-AUC of a
            random ranking.
        family (str, optional): Keep only the models of this family.

    Returns:
        pandas.io.formats.style.Styler: Validation scores (darker = better), the
        lift of PR-AUC over a random ranking, the train PR-AUC and the overfitting
        gap (train minus validation; a long bar means the model memorised).
    """
    val = log[log["dataset"] == "validation"].set_index("model")
    train = log[log["dataset"] == "train"].set_index("model")
    if family:
        val = val[val["family"] == family]
    board = val[["family"] + list(LABELS)].rename(columns=LABELS)
    board = board.rename(columns={"family": "Family"})
    board.insert(2, "Lift", board["PR-AUC"] / base_rate)
    board["Train PR-AUC"] = train.loc[board.index, "pr_auc"]
    board["Overfitting gap"] = board["Train PR-AUC"] - board["PR-AUC"]
    board = board.sort_values("PR-AUC", ascending=False)
    board.index.name = None
    metric_columns = list(LABELS.values())
    return (
        board.style.format(
            "{:.3f}", subset=metric_columns + ["Train PR-AUC", "Overfitting gap"]
        )
        .format("{:.2f}x", subset=["Lift"])
        .background_gradient(cmap="Blues", subset=metric_columns)
        .bar(subset=["Overfitting gap"], color="#f2a65a", vmin=0)
        .set_caption(
            f"Validation scores, best first. A random ranking has PR-AUC "
            f"{base_rate:.3f}; Lift = PR-AUC / that rate."
        )
    )


def rate_by(df, col, target, group, z=1.96):
    """The rate of `target` in each level of `col`, with an interval by patient.

    The encounters of one patient are not independent, so the usual binomial
    interval would be too narrow. The standard error is the cluster-robust one of a
    ratio: sqrt(G / (G - 1) * sum_g (y_g - rate * n_g)^2) / n, where y_g and n_g are
    the positives and the encounters of patient g, and G the number of patients.

    Args:
        df (pandas.DataFrame): One row per encounter.
        col (str): The column whose levels are compared.
        target (str): The 0/1 outcome column.
        group (str): The patient column.
        z (float): 1.96 for a 95% interval (normal approximation).

    Returns:
        pandas.DataFrame: One row per level, in order of appearance in `df`: the
        number of encounters and patients, the rate and its interval `low`-`high`
        (clipped to [0, 1]).
    """
    rows = []
    for level, part in df.groupby(col, observed=True, sort=False):
        n, rate = len(part), part[target].mean()
        per_patient = part.groupby(group)[target].agg(["sum", "size"])
        patients = len(per_patient)
        residuals = per_patient["sum"] - rate * per_patient["size"]
        correction = patients / max(patients - 1, 1)
        se = np.sqrt(correction * (residuals**2).sum()) / n
        rows.append(
            {
                col: level,
                "encounters": n,
                "patients": patients,
                "rate": rate,
                "low": max(rate - z * se, 0.0),
                "high": min(rate + z * se, 1.0),
            }
        )
    return pd.DataFrame(rows).set_index(col)


def cv_summary(cv):
    """Fold-by-fold scores summarised as "mean ± standard deviation", best first.

    Args:
        cv (pandas.DataFrame): One row per (model, fold), with a `model` column and
            the metric columns of `scores`.

    Returns:
        pandas.DataFrame: One row per model, one "mean ± std" text column per metric
        of `LABELS`, sorted by mean PR-AUC.
    """
    grouped = cv.groupby("model")[list(LABELS)]
    mean, std = grouped.mean(), grouped.std()
    table = mean.map("{:.3f}".format) + " ± " + std.map("{:.3f}".format)
    table = table.loc[mean["pr_auc"].sort_values(ascending=False).index]
    table.index.name = None
    return table.rename(columns=LABELS)
