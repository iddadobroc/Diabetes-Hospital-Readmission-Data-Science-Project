"""Tests on the trained model, run AFTER `train.py` (course, W10: test the file that
will be published, not a model retrained inside the test).

They load `models/model.joblib` and are skipped when it is absent.
"""

import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURES  # noqa: E402
from scoring import (  # noqa: E402
    assign_tiers,
    contributions,
    inner_pipeline,
    prepare_discharges,
    raw_scores,
)

MODELS = ROOT / "models"


@unittest.skipUnless((MODELS / "model.joblib").exists(), "models/model.joblib not found")
class TestModel(unittest.TestCase):
    """The published file answers like a sensible readmission model."""

    @classmethod
    def setUpClass(cls):
        """Load the model, its config and its metrics once."""
        cls.model = joblib.load(MODELS / "model.joblib")
        cls.config = json.loads((MODELS / "config.json").read_text())
        cls.metrics = json.loads((MODELS / "metrics.json").read_text())
        cls.mappings = {
            col: {int(k): v for k, v in m.items()}
            for col, m in cls.config["mappings"].items()
        }

    def discharge(self, **values):
        """One typical discharge, with some raw values replaced."""
        rows = pd.DataFrame([values]) if values else pd.DataFrame(index=[0])
        return prepare_discharges(rows, self.config["typical"], self.mappings)

    def risk(self, **values):
        """The calibrated risk of one discharge."""
        return self.model.predict_proba(self.discharge(**values)[FEATURES])[0, 1]

    def test_typical_discharge_has_a_plausible_risk(self):
        """A typical discharge has a risk between 3% and 30% (base rate ~11%)."""
        self.assertTrue(0.03 <= self.risk() <= 0.30)

    def test_previous_stays_raise_the_risk(self):
        """Five inpatient stays in the previous year give a higher risk than none."""
        self.assertGreater(
            self.risk(number_inpatient=5), self.risk(number_inpatient=0) + 0.05
        )

    def test_sample_scores_and_tiers(self):
        """The sample discharges get risks in [0, 1] and every tier is used."""
        sample = pd.read_csv(MODELS / "sample_discharges.csv", dtype={
            "diag_1": str, "diag_2": str, "diag_3": str}, na_values=["?"],
            keep_default_na=False)
        X = prepare_discharges(sample, self.config["typical"], self.mappings)[FEATURES]
        p = self.model.predict_proba(X)[:, 1]
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        tiers = assign_tiers(raw_scores(self.model, X), self.config["cutoffs_raw"])
        self.assertEqual(set(tiers), {"High", "Medium", "Low"})

    def test_reasons_add_up_to_the_score(self):
        """The per-feature reasons sum to the raw score in log-odds."""
        X = self.discharge(number_inpatient=3)[FEATURES]
        c = contributions(inner_pipeline(self.model), X, FEATURES)
        raw = raw_scores(self.model, X)[0]
        self.assertAlmostEqual(c.sum(axis=1).iloc[0], np.log(raw / (1 - raw)), places=4)

    def test_same_numbers_as_the_notebook(self):
        """The test metrics equal those of 03_final_model when the test was opened."""
        reference = json.loads(
            (ROOT / "reports" / "final_test_metrics.json").read_text()
        )
        for key in ["pr_auc", "roc_auc", "recall_top10", "brier"]:
            self.assertAlmostEqual(
                self.metrics["metrics"][key], reference["metrics"][key], places=6
            )
        self.assertEqual(
            self.metrics["reached_at_10pct"]["XGBoost (final)"],
            reference["reached_at_10pct"][reference["model"]],
        )


if __name__ == "__main__":
    unittest.main()
