# %% [markdown]
# # 03 — The final model: calibration, baselines, explanations, fairness, one test
#
# `02_model_exploration` compared the model families and proposed **tuned XGBoost**
# (D-019). This notebook turns that proposal into the model the hospital would use, and
# measures it once on the test patients:
#
# | Stage | What | Data |
# |---|---|---|
# | 1 | Data and the frozen split (fingerprints checked) | all |
# | 2 | The recipe reproduced, and refitted with 5 seeds | train → validation |
# | 3 | Baselines: random order, LACE index, logistic regression | train → validation |
# | 4 | Calibration rehearsal, option A against option B; `PRIMARY` | train → validation |
# | 5 | What drives the risk (TreeSHAP contributions), stability across folds | train |
# | 6 | Fairness by race, age and gender | validation |
# | 7 | **The test set, opened once**: metrics, tiers, gate, fairness | train + validation → test |
# | 8 | IBCS figures for the report, the app and the slides | test |
#
# Everything that defines the model is in `src/final_model.py`, shared with `train.py`,
# the script that the GitHub pipeline runs to publish the model on Hugging Face. The
# rules fixed **before** the test was opened are in D-032 (decision log): the
# calibration rule, the capacity tiers, the baselines and the quality gate. The test
# set had already been scored once by the LA stream on 2026-10-14 (D-031).

# %%
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

PROJECT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "data_prep.py").exists()
)
sys.path.insert(0, str(PROJECT / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import final_model  # noqa: E402
import ibcs  # noqa: E402
from calibration import capacity_cutoffs, raw_scores, tier_report  # noqa: E402
from data_prep import add_mapping_descriptions, load_mappings, load_raw  # noqa: E402
from evaluation import rate_by, scores  # noqa: E402
from features import (  # noqa: E402
    FEATURES,
    GROUP,
    SPLIT_FINGERPRINTS,
    SPLIT_SEED,
    TARGET,
    clean,
    fingerprint,
    split_by_patient,
)
from final_model import (  # noqa: E402
    CAPACITY,
    QUALITY_GATE,
    SEED,
    assign_tiers,
    build_pipeline,
    contributions,
    evaluate,
    fit_calibrated,
    grouped_folds,
    inner_pipeline,
    lace_input_columns,
    make_lace_model,
)
from preprocessing import make_pipeline  # noqa: E402

FIGURES = PROJECT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
LOG = PROJECT / "reports" / "experiments-readmission-dc.csv"
TODAY = date.today().isoformat()
pd.set_option("display.precision", 4)

# %% [markdown]
# ## Stage 1. Data and the frozen split
# The same cleaning (`clean`, decision-log defaults) and the same split as every other
# notebook. The fingerprints prove that the three sets are the very same encounters.

# %%
raw = add_mapping_descriptions(load_raw(), load_mappings())
df = clean(raw)
train, val, test = split_by_patient(df, seed=SPLIT_SEED)
parts = {"train": train, "validation": val, "test": test}
assert {k: fingerprint(v) for k, v in parts.items()} == SPLIT_FINGERPRINTS
dev = pd.concat([train, val])  # train + validation: the rows of the final fit

X_train, y_train, g_train = train[FEATURES], train[TARGET], train[GROUP]
X_val, y_val = val[FEATURES], val[TARGET]
LACE_COLS = lace_input_columns()

pd.DataFrame(
    {
        "encounters": [len(p) for p in parts.values()],
        "patients": [p[GROUP].nunique() for p in parts.values()],
        "fingerprint": [fingerprint(p) for p in parts.values()],
    },
    index=list(parts),
)

# %% [markdown]
# ## Stage 2. The recipe, reproduced and refitted with 5 seeds
# `build_pipeline()` is the tuned XGBoost of D-019, with the exact values of the
# search. One seed can flatter a model (D-030): refitting it with 5 seeds shows how
# much of the validation score is luck. A difference between models smaller than this
# spread is not a difference.

# %%
seed_rows = []
for seed in [42, 0, 1, 2, 3]:
    model = build_pipeline(seed).fit(X_train, y_train)
    s = scores(y_val, model.predict_proba(X_val)[:, 1])
    seed_rows.append({"seed": seed, "pr_auc": s["pr_auc"], "roc_auc": s["roc_auc"],
                      "recall_top10": s["recall_top10"]})
seeds = pd.DataFrame(seed_rows).set_index("seed")
display(seeds)
print(
    f"Validation PR-AUC over 5 seeds: {seeds.pr_auc.mean():.4f} ± {seeds.pr_auc.std():.4f}"
    f" (range {seeds.pr_auc.min():.4f}..{seeds.pr_auc.max():.4f})"
)

# %% [markdown]
# ## Stage 3. Baselines on validation
# A model is only useful if it beats what the hospital could do without it:
#
# - **Random order**: calling 10% of the discharges at random reaches 10% of the
#   readmissions; its PR-AUC is the `<30` rate.
# - **LACE index** (van Walraven et al. 2010), the validated clinical readmission score
#   (D-024): length of stay, acute admission, comorbidity, emergency visits.
# - **Logistic regression**, the baseline of `02_model_exploration`.

# %%
baselines = {
    "XGBoost (tuned)": build_pipeline(SEED).fit(X_train, y_train),
    "Logistic regression": make_pipeline(
        "linear", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    ).fit(X_train, y_train),
}
lace = make_lace_model().fit(train[LACE_COLS], y_train)
val_scores = {name: m.predict_proba(X_val)[:, 1] for name, m in baselines.items()}
val_scores["LACE index"] = lace.predict_proba(val[LACE_COLS])[:, 1]

rate_val = y_val.mean()
val_table = pd.DataFrame(
    {name: scores(y_val, p) for name, p in val_scores.items()}
).T[["pr_auc", "roc_auc", "precision_top10", "recall_top10"]]
val_table.loc["Random order"] = [rate_val, 0.5, rate_val, 0.10]
val_table["lift"] = val_table["pr_auc"] / rate_val
val_table.sort_values("pr_auc", ascending=False)

# %% [markdown]
# ## Stage 4. Calibration rehearsal: option A against option B
# With `scale_pos_weight` the raw scores are inflated: they rank well but are not
# risks. Two leakage-safe ways to calibrate (D-022, `src/calibration.py`):
#
# - **A, holdout**: the model is fitted on 75% of the training patients and an isotonic
#   calibrator on the other 25%.
# - **B, out of fold**: the model is fitted on all the training rows and the calibrator
#   on its grouped out-of-fold scores.
#
# Both are fitted on **train only** and scored on validation, which plays the role of
# the test. **Rule fixed in D-032 before this ran:** the lower validation Brier score
# wins; if the two differ by less than 0.001, B wins (the deployed model then uses
# every row). The capacity cut-offs come from each option's reference rows, on the
# raw score, and are checked on validation: the High tier should hold 10% of the
# encounters.


# %%
def calibration_summary(y, p):
    """Brier, log loss, calibration-in-the-large, slope and intercept (Van Calster)."""
    y = np.asarray(y)
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fit = LogisticRegression(C=np.inf).fit(logit.reshape(-1, 1), y)
    return {
        "brier": brier_score_loss(y, p),
        "mean_risk": p.mean(),
        "observed_rate": y.mean(),
        "gap_pp": 100 * (p.mean() - y.mean()),
        "slope": fit.coef_[0, 0],
        "intercept": fit.intercept_[0],
        "pr_auc_of_risk": average_precision_score(y, p),
    }


rehearsal, rehearsal_tiers = {}, {}
for option in ["A", "B"]:
    model, reference, reference_raw = fit_calibrated(option, X_train, y_train, g_train)
    p_val = model.predict_proba(X_val)[:, 1]
    cutoffs = capacity_cutoffs(reference_raw, CAPACITY)
    rehearsal[option] = calibration_summary(y_val, p_val)
    rehearsal_tiers[option] = tier_report(
        y_val, p_val, cutoffs, score=raw_scores(model, X_val)
    )
    rehearsal[option]["share_high"] = rehearsal_tiers[option].loc["High", "share_flagged"]

rehearsal = pd.DataFrame(rehearsal).T
display(rehearsal)
for option, table in rehearsal_tiers.items():
    display(table.style.format("{:.3f}").set_caption(f"Option {option}, validation"))

# %%
brier_gap = rehearsal.loc["A", "brier"] - rehearsal.loc["B", "brier"]
rehearsed_primary = "A" if brier_gap < -0.001 else "B"
print(
    f"Brier A − B = {brier_gap:+.5f} -> PRIMARY by the D-032 rule: {rehearsed_primary}"
)

# %% [markdown]
# `PRIMARY` is declared in `src/final_model.py` (and in the decision log) by hand,
# after this rehearsal and before the test is opened. Until it is declared, or if it
# disagrees with the rehearsal, Stage 7 does not run.

# %%
OPEN_TEST = final_model.PRIMARY is not None
if final_model.PRIMARY is None:
    print(
        f"PRIMARY is not declared. Declare PRIMARY = {rehearsed_primary!r} in "
        "src/final_model.py and in D-032, commit, then re-run: the test stays closed."
    )
else:
    assert final_model.PRIMARY == rehearsed_primary, "PRIMARY disagrees with the rehearsal"
    print(f"PRIMARY = {final_model.PRIMARY!r} (declared, matches the rehearsal)")

# %% [markdown]
# ## Stage 5. What drives the risk, and is it stable?
# XGBoost computes TreeSHAP contributions natively (`contributions()` in
# `src/final_model.py`): for each encounter, how much each feature moves the score
# (log-odds) up or down; the one-hot columns of a feature are summed back into it.
# The mean absolute contribution ranks the features. To check that the ranking is a
# property of the data and not of one fit, it is recomputed on the 5 grouped folds of
# the training rows (fit on 4/5 of the patients, explain the other 1/5).

# %%
fold_importance = []
for fold, (fit_idx, out_idx) in enumerate(
    grouped_folds().split(X_train, y_train, groups=g_train), start=1
):
    model = build_pipeline().fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    held_out = X_train.iloc[out_idx].sample(3000, random_state=SEED)
    contrib = contributions(model, held_out, FEATURES).drop(columns="bias")
    fold_importance.append(contrib.abs().mean().rename(f"fold {fold}"))
importance = pd.concat(fold_importance, axis=1)
importance["mean"] = importance.mean(axis=1)
importance = importance.sort_values("mean", ascending=False)
ranks = importance.drop(columns="mean").rank(ascending=False)
pairs = [
    spearmanr(ranks.iloc[:, i], ranks.iloc[:, j]).statistic
    for i in range(5)
    for j in range(i + 1, 5)
]
top12 = [set(importance[f"fold {k}"].nlargest(12).index) for k in range(1, 6)]
print(f"Spearman correlation of the rankings between folds: min {min(pairs):.3f}")
print(f"Features in the top 12 of every fold: {len(set.intersection(*top12))} of 12")
importance.head(15)

# %% [markdown]
# ## Stage 6. Fairness on validation
# For each group (race as recorded, age band, gender without the 3 Unknown/Invalid
# encounters, as the brief asks): the `<30` rate, the mean calibrated risk (does the
# model over- or under-estimate the group?), and the **share of the group's
# readmissions that reach the High tier** at 10% capacity (equal opportunity at the
# cut-off). Intervals by patient (`rate_by`). Groups with fewer than 500 encounters are
# shown but not interpreted.

# %%
AGE_GROUPS = {0: "<50", 1: "<50", 2: "<50", 3: "<50", 4: "<50",
              5: "50-59", 6: "60-69", 7: "70-79", 8: "80+", 9: "80+"}


def fairness_table(part, p, raw, cutoffs):
    """Rate, mean risk, and recall of the High tier for every group of part."""
    frame = part[[GROUP, TARGET, "race", "gender"]].copy()
    frame["age group"] = part["age_band_order"].map(AGE_GROUPS)
    frame["risk"] = p
    frame["high"] = (raw >= cutoffs["High"]).astype(int)
    rows = []
    for col in ["race", "age group", "gender"]:
        sub = frame[frame["gender"] != "Unknown/Invalid"] if col == "gender" else frame
        rates = rate_by(sub, col, TARGET, GROUP)
        reached = rate_by(sub[sub[TARGET] == 1], col, "high", GROUP)
        for level in rates.index:
            rows.append({
                "attribute": col,
                "group": level,
                "encounters": rates.loc[level, "encounters"],
                "rate": rates.loc[level, "rate"],
                "mean_risk": sub.loc[sub[col] == level, "risk"].mean(),
                "high_recall": reached.loc[level, "rate"] if level in reached.index else np.nan,
                "high_recall_low": reached.loc[level, "low"] if level in reached.index else np.nan,
                "high_recall_high": reached.loc[level, "high"] if level in reached.index else np.nan,
                "share_in_high": sub.loc[sub[col] == level, "high"].mean(),
            })
    out = pd.DataFrame(rows)
    out["gap_pp"] = 100 * (out["mean_risk"] - out["rate"])
    return out


model_v, ref_v, ref_raw_v = fit_calibrated(rehearsed_primary, X_train, y_train, g_train)
cut_v = capacity_cutoffs(ref_raw_v, CAPACITY)
fair_val = fairness_table(
    val, model_v.predict_proba(X_val)[:, 1], raw_scores(model_v, X_val), cut_v
)
fair_val.round(3)

# %% [markdown]
# ## Stage 7. The test set, opened once
# Only if `PRIMARY` is declared (Stage 4). The final model is the `PRIMARY` option
# fitted on train + validation; the baselines are refitted on the same rows. Every
# number below is computed once and written to the experiment log and to
# `reports/final_test_metrics.json`, which `train.py` must reproduce.

# %%
assert OPEN_TEST, "Stage 7 needs PRIMARY to be declared: the test set stays closed."

X_dev, y_dev, g_dev = dev[FEATURES], dev[TARGET], dev[GROUP]
X_test, y_test = test[FEATURES], test[TARGET].to_numpy()
final, ref_risk, ref_raw = fit_calibrated(final_model.PRIMARY, X_dev, y_dev, g_dev)
cutoffs = capacity_cutoffs(ref_raw, CAPACITY)
p_test = final.predict_proba(X_test)[:, 1]
raw_test = raw_scores(final, X_test)

final_name = f"XGBoost final (option {final_model.PRIMARY})"
lace_dev = make_lace_model().fit(dev[LACE_COLS], y_dev)
logit_dev = make_pipeline(
    "linear", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
).fit(X_dev, y_dev)
test_scores = {
    final_name: p_test,
    "LACE index": lace_dev.predict_proba(test[LACE_COLS])[:, 1],
    "Logistic regression": logit_dev.predict_proba(X_test)[:, 1],
}
rate_test = y_test.mean()
# Ranking metrics of the final model on its raw score, calibration on its risk (D-032)
test_metrics = {n: scores(y_test, p) for n, p in test_scores.items()}
test_metrics[final_name] = evaluate(y_test, p_test, raw_test)
test_table = pd.DataFrame(test_metrics).T
test_table.loc["Random order", ["pr_auc", "roc_auc", "precision_top10", "recall_top10"]] = [
    rate_test, 0.5, rate_test, 0.10]
display(test_table[["pr_auc", "roc_auc", "precision_top10", "recall_top10", "brier"]])

tiers_test = tier_report(y_test, p_test, cutoffs, score=raw_test)
display(tiers_test.style.format("{:.3f}").set_caption("Capacity tiers on test"))

# %% [markdown]
# **The quality gate (D-032)**, checked on the numbers above.

# %%
calib = calibration_summary(y_test, p_test)
gate_values = {
    "pr_auc": test_table.loc[final_name, "pr_auc"],
    "gain_over_lace": test_table.loc[final_name, "pr_auc"] - test_table.loc["LACE index", "pr_auc"],
    "recall_top10": tiers_test.loc["High", "recall"],
    "calibration_gap_pp": abs(calib["gap_pp"]),
}
gate = pd.DataFrame(
    {
        "value": gate_values,
        "threshold": {
            "pr_auc": QUALITY_GATE["min_pr_auc"],
            "gain_over_lace": QUALITY_GATE["min_gain_over_lace"],
            "recall_top10": QUALITY_GATE["min_recall_top10"],
            "calibration_gap_pp": QUALITY_GATE["max_calibration_gap_pp"],
        },
    }
)
gate["passed"] = [
    gate.loc["pr_auc", "value"] >= gate.loc["pr_auc", "threshold"],
    gate.loc["gain_over_lace", "value"] >= gate.loc["gain_over_lace", "threshold"],
    gate.loc["recall_top10", "value"] >= gate.loc["recall_top10", "threshold"],
    gate.loc["calibration_gap_pp", "value"] <= gate.loc["calibration_gap_pp", "threshold"],
]
display(gate)
print("Calibration on test:", {k: round(v, 4) for k, v in calib.items()})

# %% [markdown]
# **Fairness on test**, the same table as Stage 6, with the final model.

# %%
fair_test = fairness_table(test, p_test, raw_test, cutoffs)
fair_test.round(3)

# %% [markdown]
# **Saved:** the test rows of the experiment log (re-running replaces today's rows,
# never adds duplicates) and `reports/final_test_metrics.json`.

# %%
log = pd.read_csv(LOG)
note = "03_final_model, fitted on train + validation, scored once on test"
log = log[~((log["date"] == TODAY) & (log["note"] == note))]
new_rows = []
for name, metrics_ in test_metrics.items():
    family = {"LACE index": "clinical score", "Logistic regression": "linear"}.get(name, "boosting")
    new_rows.append({"date": TODAY, "model": name, "family": family, "dataset": "test"}
                    | metrics_ | {"note": note})
log = pd.concat([log, pd.DataFrame(new_rows).round(3)], ignore_index=True)
log.to_csv(LOG, index=False)

k10 = int(np.ceil(CAPACITY["High"] * len(y_test)))
order = {name: np.argsort(-p, kind="stable") for name, p in test_scores.items()}
order[final_name] = np.argsort(-raw_test, kind="stable")
reached10 = {name: int(y_test[o[:k10]].sum()) for name, o in order.items()}
summary = {
    "model": final_name,
    "primary": final_model.PRIMARY,
    "test_encounters": int(len(y_test)),
    "test_patients": int(test[GROUP].nunique()),
    "test_readmissions": int(y_test.sum()),
    "test_rate": float(rate_test),
    "metrics": {k: float(v) for k, v in test_metrics[final_name].items()},
    "calibration": {k: float(v) for k, v in calib.items()},
    "tiers": tiers_test.drop(columns="cutoff").astype(float).round(6).to_dict(orient="index"),
    "cutoffs_raw": cutoffs,
    "baselines": {
        n: {k: float(v) for k, v in test_table.loc[n, ["pr_auc", "roc_auc", "precision_top10", "recall_top10"]].items()}
        for n in ["LACE index", "Logistic regression", "Random order"]
    },
    "reached_at_10pct": reached10 | {"Random order": float(k10 * rate_test)},
    "calls_at_10pct": k10,
    "gate": {k: {"value": float(gate.loc[k, "value"]), "passed": bool(gate.loc[k, "passed"])} for k in gate.index},
}
(PROJECT / "reports" / "final_test_metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["reached_at_10pct"], indent=2))

# %% [markdown]
# ## Stage 8. Figures in the IBCS notation
# Each figure is a page: a message (a full sentence that can be checked), a three-line
# title (who / **what** in unit / when) and a footnote with the source and what was
# left out. Models are structure, so they run top to bottom; observed values (AC) are
# solid, the model's estimates (FC) hatched; no axes, no grid lines, data labels.

# %%
WHO = "Diabetic inpatient encounters, 130 US hospitals, test patients"
WHEN_TEST = "1999..2008, AC on held-out test patients"
N = ibcs.fmt_number  # numbers in the IBCS format (thin-space thousands)
SOURCE = (
    "Source: UCI Diabetes 130-US hospitals 1999..2008 (Strack et al. 2014). Test: "
    f"{N(len(y_test))} encounters of {N(test[GROUP].nunique())} patients (20% of patients, held "
    f"out from fitting and choices). Excluded: {N(2423)} encounters ending in death or hospice."
)
LABEL = {final_name: "XGBoost (final)", "LACE index": "LACE index",
         "Logistic regression": "Logistic regression", "Random order": "Random order"}

# 8.1 The headline: readmissions reached with capacity for 10% of discharges
names = [final_name, "LACE index", "Logistic regression", "Random order"]
reached = [reached10[final_name], reached10["LACE index"], reached10["Logistic regression"],
           k10 * rate_test]
delta = [r - reached10["LACE index"] for r in reached]
delta[1] = np.nan
fig, (ax, axv) = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True,
                              gridspec_kw={"width_ratios": [3, 1.4], "wspace": 0.05})
ibcs.bars_h(ax, [LABEL[n] for n in names], reached, highlight=0)
ibcs.variance_h(axv, delta, xmax=max(reached) * 0.45)
axv.set_title("ΔLACE", fontsize=9, loc="left")
ibcs.page(
    fig,
    f"Calling the {N(k10)} discharges with the highest risk (10%) would reach "
    f"{N(reached10[final_name])} of {N(y_test.sum())} readmissions,\n"
    f"{N(reached10[final_name] - reached10['LACE index'])} more than the LACE score and "
    f"{N(reached10[final_name] - k10 * rate_test)} more than calling at random",
    WHO, "Readmissions < 30 days reached", "in #, by ranking method",
    f"{WHEN_TEST}; capacity 10% of discharges", top=0.62,
)
ibcs.footnote(fig, SOURCE + "\nRandom order: expected value at the test <30 rate. "
              "LACE: length of stay, acute admission, Charlson comorbidity, emergency visits.")
ibcs.save(fig, FIGURES / "01_reached_at_capacity.png")

# 8.2 Capture curve: share of readmissions reached by share of discharges called
share = np.arange(1, 101) / 100
n = len(y_test)
fig, ax = plt.subplots(figsize=(8, 4.8))
for name, color, width in [(final_name, ibcs.AC, 2.4), ("LACE index", ibcs.PY, 1.6)]:
    o = order[name]
    curve = [y_test[o[: int(np.ceil(s * n))]].sum() / y_test.sum() for s in share]
    ax.plot(100 * share, 100 * np.array(curve), color=color, linewidth=width)
    # Labels inside the chart, where the lines are apart (legend integrated, IBCS UN 2.3)
    above = name == final_name
    ax.text(55, 100 * curve[54] + (5 if above else -6), LABEL[name], fontsize=9,
            color=ibcs.TEXT if above else ibcs.MUTED, ha="right" if above else "left")
    for s in (0.10, 0.30):
        v = 100 * curve[int(s * 100) - 1]
        ax.plot(100 * s, v, "o", color=color, markersize=4)
        ax.text(100 * s + 1.2, v + 2.5 if above else v - 6, f"{v:.0f}%", fontsize=8.5,
                color=ibcs.TEXT if above else ibcs.MUTED)
ax.plot([0, 100], [0, 100], color=ibcs.PY, linewidth=1, linestyle=(0, (4, 3)))
ax.text(80, 72, "Random order", fontsize=9, color=ibcs.MUTED, ha="left")
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.set_xticks([0, 10, 30, 50, 100], ["0%", "10%", "30%", "50%", "100%"])
ax.set_yticks([])
ibcs.clean_axes(ax)
ax.spines["bottom"].set_visible(True)
c10 = 100 * reached10[final_name] / y_test.sum()
ibcs.page(
    fig,
    f"Following up the riskiest 10% of discharges reaches {c10:.0f}% of the readmissions; "
    f"the riskiest 30% reach {100 * tiers_test.loc['Medium', 'recall']:.0f}%",
    WHO, "Readmissions < 30 days reached", "in % of all readmissions, by share of discharges called",
    WHEN_TEST, top=0.72,
)
ibcs.footnote(fig, SOURCE)
ibcs.save(fig, FIGURES / "02_capture_curve.png")

# 8.3 Calibration: observed (AC) against estimated (FC) by risk group
groups = pd.qcut(pd.Series(raw_test).rank(method="first"), 10, labels=False)
cal = pd.DataFrame({"g": groups, "y": y_test, "p": p_test}).groupby("g").agg(
    ac=("y", "mean"), fc=("p", "mean"))
cal_labels = [f"D{i + 1}" for i in range(10)]
diff = 100 * (cal["ac"] - cal["fc"])
fig, (axd, axc) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.08})
ibcs.variance_columns(axd, cal_labels, diff.to_numpy(), unit="pp", higher_is_better=False)
axd.text(-0.6, 0, "ΔFC", fontsize=8.5, va="center", ha="right")
ibcs.columns_ac_fc(axc, cal_labels, 100 * cal["ac"].to_numpy(), 100 * cal["fc"].to_numpy(),
                   decimals=0, unit="%")
axc.set_xticks(range(10), cal_labels)
ibcs.page(
    fig,
    f"Observed readmissions stayed within {diff.abs().max():.1f} pp of the model's risk in all "
    f"10 risk groups; overall {100 * rate_test:.1f}% observed against {100 * p_test.mean():.1f}% estimated",
    WHO, "Readmission rate < 30 days", "in %, AC observed and FC estimated, by risk decile (D1 lowest)",
    f"{WHEN_TEST}; FC = calibrated model risk, ΔFC = AC − FC in pp", top=0.72,
)
ibcs.footnote(fig, SOURCE)
ibcs.save(fig, FIGURES / "03_calibration.png")

# 8.4 Fairness: share of each group's readmissions reached by the High tier
show = fair_test[fair_test["encounters"] >= 500].copy()
ORDER = {"<50": 0, "50-59": 1, "60-69": 2, "70-79": 3, "80+": 4}
show["key"] = [
    (["race", "age group", "gender"].index(a), ORDER.get(g, 0), -n)
    for a, g, n in zip(show["attribute"], show["group"], show["encounters"])
]
show = show.sort_values("key")
show["label"] = show["attribute"].str.replace("age group", "age") + ": " + show["group"].astype(str)
fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(show) + 2.6))
ibcs.bars_h(ax, show["label"].tolist(), 100 * show["high_recall"].to_numpy(), unit="%",
            width=0.33, xmax=100 * show["high_recall"].max() * 1.25)
overall = 100 * tiers_test.loc["High", "recall"]
ax.axvline(overall, color=ibcs.NEUTRAL, linewidth=1, linestyle=(0, (3, 2)))
ax.text(overall, -0.9, f"all: {overall:.0f}%", color=ibcs.NEUTRAL, fontsize=8.5, ha="center")
lo, hi = 100 * show["high_recall"].min(), 100 * show["high_recall"].max()
ages = show[show["attribute"] == "age group"].set_index("group")["high_recall"]
young, old = 100 * ages.get("<50", np.nan), 100 * ages.get("80+", np.nan)
ibcs.page(
    fig,
    f"The High tier reaches between {lo:.0f}% and {hi:.0f}% of each group's readmissions "
    f"(all patients: {overall:.0f}%): under 50 {young:.0f}%, 80 and over {old:.0f}%",
    WHO, "Readmissions reached by the High tier", "in % of the group's readmissions, by group",
    f"{WHEN_TEST}; groups with at least 500 test encounters", top=0.80,
)
ibcs.footnote(fig, SOURCE + "\nGender: the 3 Unknown/Invalid encounters are not reported (brief). "
              "Race as recorded in the source; Unknown = not recorded.", bottom=0.11)
ibcs.save(fig, FIGURES / "04_fairness_high_tier.png")

# 8.5 What drives the risk
test_contrib = contributions(inner_pipeline(final), X_test, FEATURES).drop(columns="bias")
drivers = test_contrib.abs().mean().sort_values(ascending=False).head(12)
READABLE = {
    "number_inpatient": "Inpatient stays, previous year",
    "discharge_disposition_desc": "Discharge destination",
    "payer_code": "Payer",
    "medical_specialty": "Admitting specialty",
    "n_visits_last_year": "All visits, previous year",
    "num_medications": "Medications given",
    "diag_1_group": "Primary diagnosis group",
    "age_band_order": "Age band",
    "number_diagnoses": "Number of diagnoses",
    "diag_2_group": "Secondary diagnosis group",
    "num_lab_procedures": "Lab procedures",
    "time_in_hospital": "Days in hospital",
    "admission_source_desc": "Admission source",
    "n_meds_active": "Diabetes drugs prescribed",
    "num_procedures": "Procedures",
    "number_emergency": "Emergency visits, previous year",
    "diag_3_group": "Third diagnosis group",
    "number_outpatient": "Outpatient visits, previous year",
    "admission_type_desc": "Admission type",
    "A1Cresult": "HbA1c result",
    "insulin": "Insulin",
    "race": "Race",
    "metformin": "Metformin",
    "n_meds_changed": "Diabetes drugs changed",
}
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ibcs.bars_h(ax, [READABLE.get(f, f) for f in drivers.index], drivers.to_numpy(), decimals=2,
            highlight=None, width=0.66)
ibcs.page(
    fig,
    "Previous inpatient stays and the discharge destination move the risk most: "
    f"together {drivers.iloc[:2].sum() / test_contrib.abs().mean().sum():.0%} of the total effect",
    WHO, "Effect on the risk score", "in mean |TreeSHAP log-odds|, 12 largest features",
    f"{WHEN_TEST}; final XGBoost", top=0.78,
)
ibcs.footnote(fig, SOURCE + "\nAssociation, not cause: the model ranks risk, it does not show "
              "that changing a feature changes the risk.", bottom=0.12)
ibcs.save(fig, FIGURES / "05_risk_drivers.png")
drivers

# %% [markdown]
# **8.6 Model families on validation** (from `02_model_exploration`, plus the LACE
# baseline): the reason for the choice of XGBoost.

# %%
dc = pd.read_csv(LOG)
latest = dc[(dc["dataset"] == "validation") & (dc["date"] == "2026-10-19")]
fam = latest.set_index("model")["pr_auc"]
fam = fam.drop(index=[m for m in fam.index if "baseline" in m and "Logistic" not in m], errors="ignore")
fam["LACE index"] = val_table.loc["LACE index", "pr_auc"]
fam["Random order"] = rate_val
fam = fam.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(fam) + 2.4))
ibcs.bars_h(ax, list(fam.index), fam.to_numpy(), decimals=3, width=0.33,
            highlight=list(fam.index).index("XGBoost (tuned)"))
ibcs.page(
    fig,
    f"Once tuned, the boosting libraries rank readmissions alike (PR-AUC {fam.drop(['LACE index', 'Random order']).nlargest(6).min():.3f}.."
    f"{fam.max():.3f}); XGBoost is chosen as the simplest of the best to calibrate and explain",
    "Diabetic inpatient encounters, 130 US hospitals, validation patients",
    "PR-AUC", "(average precision), by model", "1999..2008, AC on validation patients; models fitted on train",
    top=0.80,
)
ibcs.footnote(fig, "Source: experiments-readmission-dc.csv (2026-10-19 run of 02_model_exploration) "
              "and 03_final_model (LACE). Random order = validation <30 rate.", bottom=0.08)
ibcs.save(fig, FIGURES / "06_model_families_validation.png")
fam.round(3)
