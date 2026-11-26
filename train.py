"""Train the published model: the recipe of D-032, rebuilt from the raw data.

The GitHub pipeline runs this script on every push (schema of the course, W10): it
cleans the data, rebuilds the frozen patient split, fits the final model on train +
validation (calibration option B), scores it on the test patients with the baselines,
and writes everything the quality gate, the Hugging Face model repository and the app
need into `models/`:

| file | what |
|---|---|
| `model.joblib` | the calibrated pipeline (preprocessing + XGBoost + isotonic calibrator) |
| `metrics.json` | test metrics, baselines, tiers, the values checked by the gate |
| `config.json` | what the app needs: features, input choices, typical values, ID mappings, cut-offs |
| `sample_discharges.csv` | 2,000 test discharges in the raw UCI format, for the worklist demo |

Every number must equal `reports/final_test_metrics.json`, written by
`DC_Models/03_final_model` when the test set was opened (checked in the tests).

    python scripts/fetch_data.py
    python train.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from calibration import capacity_cutoffs, tier_report  # noqa: E402
from data_prep import (  # noqa: E402
    ID_COLUMNS,
    add_mapping_descriptions,
    load_mappings,
    load_raw,
)
from evaluation import scores  # noqa: E402
from features import (  # noqa: E402
    CATEGORICAL,
    DEATH_OR_HOSPICE,
    FEATURES,
    GROUP,
    NUMERIC,
    SPLIT_FINGERPRINTS,
    SPLIT_SEED,
    TARGET,
    clean,
    fingerprint,
    split_by_patient,
)
from final_model import (  # noqa: E402
    CAPACITY,
    PRIMARY,
    QUALITY_GATE,
    evaluate,
    fit_calibrated,
    lace_input_columns,
    make_lace_model,
    raw_scores,
)
from preprocessing import make_pipeline  # noqa: E402

OUT = ROOT / "models"
VERSION = os.environ.get("MODEL_VERSION", "dev")
SAMPLE_SIZE = 2000


def typical_values(raw_rows):
    """The most frequent value of each raw column (median for the counts).

    The app asks for a few inputs only; every other column of the discharge takes its
    typical value, computed on the fitting rows (train + validation) only.
    """
    typical = {}
    for col in raw_rows.columns:
        values = raw_rows[col].dropna()
        if pd.api.types.is_numeric_dtype(values) and col not in ID_COLUMNS:
            typical[col] = int(values.median())
        else:
            typical[col] = str(values.astype(str).mode().iloc[0])
    return typical


def input_choices(dev):
    """The levels offered by the app for each categorical feature, most frequent first."""
    return {
        col: dev[col].astype(str).value_counts().index.tolist()[:25]
        for col in CATEGORICAL
    }


def main():
    """Rebuild the final model and write the four files of `models/`."""
    raw = add_mapping_descriptions(load_raw(), load_mappings())
    df = clean(raw)
    train, val, test = split_by_patient(df, seed=SPLIT_SEED)
    found = {"train": fingerprint(train), "validation": fingerprint(val),
             "test": fingerprint(test)}
    assert found == SPLIT_FINGERPRINTS, f"the split changed: {found}"
    dev = pd.concat([train, val])

    # The final model (D-032): option PRIMARY on train + validation
    model, _, reference_raw = fit_calibrated(PRIMARY, dev[FEATURES], dev[TARGET],
                                             dev[GROUP])
    cutoffs = capacity_cutoffs(reference_raw, CAPACITY)
    y = test[TARGET].to_numpy()
    p = model.predict_proba(test[FEATURES])[:, 1]
    raw_test = raw_scores(model, test[FEATURES])
    metrics = evaluate(y, p, raw_test)

    # Baselines on the same patients
    lace_cols = lace_input_columns()
    lace = make_lace_model().fit(dev[lace_cols], dev[TARGET])
    p_lace = lace.predict_proba(test[lace_cols])[:, 1]
    logit = make_pipeline(
        "linear", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    ).fit(dev[FEATURES], dev[TARGET])
    p_logit = logit.predict_proba(test[FEATURES])[:, 1]
    baselines = {
        "LACE index": scores(y, p_lace),
        "Logistic regression": scores(y, p_logit),
    }
    tiers = tier_report(y, p, cutoffs, score=raw_test)
    k10 = int(np.ceil(CAPACITY["High"] * len(y)))
    reached = {
        "XGBoost (final)": int(y[np.argsort(-raw_test, kind="stable")[:k10]].sum()),
        "LACE index": int(y[np.argsort(-p_lace, kind="stable")[:k10]].sum()),
        "Logistic regression": int(y[np.argsort(-p_logit, kind="stable")[:k10]].sum()),
        "Random order": float(k10 * y.mean()),
    }
    gate_values = {
        "pr_auc": metrics["pr_auc"],
        "gain_over_lace": metrics["pr_auc"] - baselines["LACE index"]["pr_auc"],
        "recall_top10": float(tiers.loc["High", "recall"]),
        "calibration_gap_pp": float(100 * abs(p.mean() - y.mean())),
    }

    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "model.joblib", compress=3)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metrics_out = {
        "version": VERSION,
        "created": created,
        "model": f"XGBoost, isotonic calibration option {PRIMARY}",
        "fitted_on": {"encounters": len(dev), "patients": int(dev[GROUP].nunique())},
        "test": {
            "encounters": len(test),
            "patients": int(test[GROUP].nunique()),
            "readmissions": int(y.sum()),
            "rate": float(y.mean()),
            "mean_risk": float(p.mean()),
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "baselines": {
            name: {k: float(v) for k, v in values.items()}
            for name, values in baselines.items()
        },
        "tiers": tiers.drop(columns="cutoff").astype(float).to_dict(orient="index"),
        "calls_at_10pct": k10,
        "reached_at_10pct": reached,
        "gate": {"values": gate_values, "thresholds": QUALITY_GATE},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # What the app needs to build and score a discharge
    mappings = load_mappings()
    dev_raw = raw[raw["encounter_id"].isin(dev["encounter_id"])]
    raw_columns = [c for c in raw.columns if not c.endswith("_desc")]
    config = {
        "version": VERSION,
        "created": created,
        "features": FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "raw_columns": raw_columns,
        "typical": typical_values(dev_raw[raw_columns].drop(columns=["readmitted"])),
        "choices": input_choices(dev),
        "numeric_ranges": {
            col: [int(dev[col].min()), int(dev[col].quantile(0.99))] for col in NUMERIC
        },
        "mappings": {col: {str(k): v for k, v in m.items()} for col, m in mappings.items()},
        "excluded_discharge_codes": DEATH_OR_HOSPICE,
        "capacity": CAPACITY,
        "cutoffs_raw": cutoffs,
        "base_rate": float(dev[TARGET].mean()),
        "libraries": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    sample = raw[raw["encounter_id"].isin(test["encounter_id"])]
    sample = sample[raw_columns].drop(columns=[GROUP]).sample(
        SAMPLE_SIZE, random_state=SPLIT_SEED
    )
    sample.to_csv(OUT / "sample_discharges.csv", index=False, na_rep="?")  # UCI format

    print(json.dumps({"version": VERSION, "metrics": metrics_out["metrics"],
                      "gate": gate_values, "reached_at_10pct": reached}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
