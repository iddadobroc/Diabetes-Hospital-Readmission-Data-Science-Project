"""Tests for the shared code in src. Run them from the project folder with:

    python -m unittest

Each test gives the code a small table made by hand, where the right answer is
known, and checks it. No test reads `data/`, so they run on any computer. Most of them
check a rule that protects against leakage, or a bug found while building the
notebooks (a model that cannot be cloned, a model whose training diverges).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import cv_summary, leaderboard, scores  # noqa: E402
from features import (  # noqa: E402
    FEATURES,
    GROUP,
    MEDICATIONS,
    REFERENCE,
    TARGET,
    clean,
    first_encounter_only,
    icd9_group,
    split_by_patient,
)
from models import (  # noqa: E402
    CatBoostNativeClassifier,
    KerasMLPClassifier,
    XGBAFTClassifier,
)
from preprocessing import make_preprocessor  # noqa: E402


def raw_rows(n=6, **columns):
    """A tiny table with every column `clean` needs, as `add_mapping_descriptions`
    returns it. Each keyword replaces one column (a list of n values)."""
    df = pd.DataFrame(
        {
            "encounter_id": range(1, n + 1),
            "patient_nbr": range(101, 101 + n),
            "race": ["Caucasian"] * n,
            "gender": ["Female"] * n,
            "age": ["[70-80)"] * n,
            "weight": [np.nan] * n,
            "admission_type_id": [1] * n,
            "discharge_disposition_id": [1] * n,
            "admission_source_id": [7] * n,
            "time_in_hospital": [3] * n,
            "payer_code": [np.nan] * n,
            "medical_specialty": [np.nan] * n,
            "num_lab_procedures": [40] * n,
            "num_procedures": [1] * n,
            "num_medications": [12] * n,
            "number_outpatient": [0] * n,
            "number_emergency": [1] * n,
            "number_inpatient": [2] * n,
            "diag_1": ["428"] * n,
            "diag_2": ["250.83"] * n,
            "diag_3": [np.nan] * n,
            "number_diagnoses": [7] * n,
            "max_glu_serum": ["None"] * n,
            "A1Cresult": [">8"] * n,
            "change": ["No"] * n,
            "diabetesMed": ["Yes"] * n,
            "readmitted": ["NO"] * n,
            "admission_type_desc": ["Emergency"] * n,
            "discharge_disposition_desc": ["Discharged to home"] * n,
            "admission_source_desc": ["Emergency Room"] * n,
        }
    )
    for drug in MEDICATIONS:
        df[drug] = "No"
    for name, values in columns.items():
        df[name] = values
    id_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]
    df[id_columns] = df[id_columns].astype("category")
    return df


class TestIcd9Group(unittest.TestCase):
    """`icd9_group` follows the Strack et al. groups of the project brief."""

    def test_tricky_codes(self):
        """250.xx before the ranges, V/E codes before any numeric conversion."""
        cases = {
            "250.83": "Diabetes",
            "250": "Diabetes",
            "428": "Circulatory",
            "785": "Circulatory",
            "786": "Respiratory",
            "V57": "Other",
            "E888": "Other",
            "153": "Neoplasms",
        }
        for code, group in cases.items():
            self.assertEqual(icd9_group(code), group, code)
        self.assertEqual(icd9_group(np.nan), "Missing")


class TestClean(unittest.TestCase):
    """`clean` applies the decision-log rules row by row."""

    def setUp(self):
        """Six encounters: home, expired (11), hospice (13), expired (20), <30, >30."""
        self.raw = raw_rows(
            discharge_disposition_id=[1, 11, 13, 20, 1, 1],
            readmitted=["NO", "NO", "<30", "NO", "<30", ">30"],
        )

    def test_death_and_hospice_are_removed(self):
        """D-004, D-005: only the three ordinary discharges are left."""
        out = clean(self.raw)
        self.assertEqual(list(out["encounter_id"]), [1, 5, 6])

    def test_hospice_flag_keeps_hospice_but_not_death(self):
        """Alternative of D-005: hospice kept with a flag, expired still removed."""
        out = clean(self.raw, hospice="flag")
        self.assertEqual(list(out["encounter_id"]), [1, 3, 5, 6])
        self.assertEqual(list(out["hospice"]), [0, 1, 0, 0])

    def test_target_and_no_answer_left(self):
        """D-011: only "<30" is positive, and the raw outcome is dropped."""
        out = clean(self.raw)
        self.assertEqual(list(out[TARGET]), [0, 1, 0])
        self.assertNotIn("readmitted", out.columns)

    def test_features_never_include_identifiers_or_target(self):
        """D-012: identifiers, reference columns and the target are not features."""
        forbidden = set(REFERENCE) | {GROUP, TARGET, "readmitted"}
        self.assertFalse(forbidden & set(FEATURES))
        self.assertTrue(set(FEATURES) <= set(clean(self.raw).columns))

    def test_explicit_missing_levels(self):
        """D-008: missing payer and specialty become "Unknown", diagnoses "Missing"."""
        out = clean(self.raw)
        self.assertTrue((out["payer_code"] == "Unknown").all())
        self.assertTrue((out["medical_specialty"] == "Unknown").all())
        self.assertTrue((out["diag_3_group"] == "Missing").all())

    def test_medication_counts(self):
        """The summaries count every drug, rare ones included, before dropping them."""
        raw = raw_rows(n=1, insulin=["Up"], tolbutamide=["Steady"])
        out = clean(raw)
        self.assertEqual(out.loc[0, "n_meds_active"], 2)
        self.assertEqual(out.loc[0, "n_meds_changed"], 1)
        self.assertNotIn("tolbutamide", out.columns)

    def test_alternative_options(self):
        """Alternatives of D-007, D-009, D-006 add or change one column each."""
        raw = raw_rows(
            n=2,
            tolbutamide=["Steady", "No"],
            medical_specialty=["Surgery-Neuro", "SomethingNew"],
            weight=["[75-100)", np.nan],
        )
        out = clean(raw, rare_meds="group", specialty="clinical", weight_flag=True)
        self.assertEqual(list(out["other_meds"]), ["Yes", "No"])
        self.assertEqual(list(out["medical_specialty"]), ["Surgery", "Other"])
        self.assertEqual(list(out["weight_recorded"]), [1, 0])


class TestSplits(unittest.TestCase):
    """No patient is ever on both sides of a split."""

    def setUp(self):
        """300 encounters of 100 patients, three each."""
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            {
                "encounter_id": rng.permutation(300),
                GROUP: np.repeat(np.arange(100), 3),
                TARGET: rng.integers(0, 2, 300),
            }
        )

    def test_split_by_patient(self):
        """The three sets share no patient and together hold every row."""
        train, val, test = split_by_patient(self.df, seed=42)
        sets = [set(part[GROUP]) for part in (train, val, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))

    def test_first_encounter_only(self):
        """One row per patient: the one with the smallest encounter_id."""
        first = first_encounter_only(self.df)
        self.assertEqual(len(first), 100)
        smallest = self.df.groupby(GROUP)["encounter_id"].min()
        self.assertTrue((first.set_index(GROUP)["encounter_id"] == smallest).all())


class TestPreprocessing(unittest.TestCase):
    """The preprocessing learns its levels from the rows it is fitted on only."""

    def test_rare_and_unseen_levels(self):
        """A level with < 100 fitting rows, or never seen, goes to "infrequent"."""
        fit = pd.DataFrame(
            {"x": [1.0] * 250, "colour": ["red"] * 150 + ["blue"] * 50 + ["green"] * 50}
        )
        prep = make_preprocessor(
            "onehot", numeric=["x"], categorical=["colour"], all_levels=()
        ).fit(fit)
        out = prep.transform(pd.DataFrame({"x": [1.0, 1.0], "colour": ["red", "pink"]}))
        self.assertEqual(
            list(out.columns), ["x", "colour_red", "colour_infrequent_sklearn"]
        )
        self.assertEqual(out["colour_infrequent_sklearn"].tolist(), [0.0, 1.0])

    def test_scaling_uses_fitting_rows(self):
        """Standardised with the mean and std of the fitting rows, not of new rows."""
        fit = pd.DataFrame({"x": [0.0, 2.0, 4.0], "colour": ["a", "a", "a"]})
        prep = make_preprocessor(
            "linear", numeric=["x"], categorical=["colour"], all_levels=()
        ).fit(fit)
        out = prep.transform(pd.DataFrame({"x": [2.0], "colour": ["a"]}))
        self.assertAlmostEqual(out["x"].iloc[0], 0.0)


class TestEvaluation(unittest.TestCase):
    """The metrics and the readable tables."""

    def test_perfect_ranking(self):
        """A model that ranks every positive first gets PR-AUC and ROC-AUC of 1."""
        s = scores(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]))
        self.assertEqual(s["pr_auc"], 1.0)
        self.assertEqual(s["roc_auc"], 1.0)

    def test_tables(self):
        """The leaderboard is sorted by validation PR-AUC; cv_summary is mean ± std."""
        log = pd.DataFrame(
            [
                {
                    "model": m,
                    "family": "f",
                    "dataset": d,
                    "pr_auc": v,
                    "roc_auc": 0.6,
                    "precision_top10": 0.2,
                    "recall_top10": 0.2,
                }
                for m, d, v in [
                    ("A", "train", 0.3),
                    ("A", "validation", 0.20),
                    ("B", "train", 0.3),
                    ("B", "validation", 0.25),
                ]
            ]
        )
        self.assertEqual(list(leaderboard(log, 0.1).data.index), ["B", "A"])
        cv = log.assign(fold=1)
        self.assertIn("±", cv_summary(cv).iloc[0, 0])


class TestModelWrappers(unittest.TestCase):
    """The wrappers of src/models.py behave like scikit-learn classifiers."""

    def test_can_be_cloned(self):
        """Every cross-validation clones the model: this failed for raw CatBoost."""
        for model in [
            XGBAFTClassifier(),
            CatBoostNativeClassifier(),
            KerasMLPClassifier(),
        ]:
            self.assertEqual(clone(model).get_params(), model.get_params())

    def test_aft_does_not_diverge(self):
        """With a small sigma, the AFT used to give every row the same score."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 3))
        y = (X[:, 0] + rng.normal(size=2000) > 1.2).astype(int)
        model = XGBAFTClassifier(distribution="extreme", sigma=0.6, n_estimators=50)
        p = model.fit(X, y).predict_proba(X)[:, 1]
        self.assertGreater(scores(y, p)["roc_auc"], 0.7)

    def test_catboost_reads_text_columns(self):
        """CatBoostNativeClassifier finds the categorical columns by itself."""
        X = pd.DataFrame({"x": np.arange(40.0), "colour": ["red", "blue"] * 20})
        y = np.array([0, 1] * 20)
        model = CatBoostNativeClassifier(iterations=10).fit(X, y)
        self.assertEqual(model.cat_features_, ["colour"])
        self.assertEqual(model.predict_proba(X).shape, (40, 2))


if __name__ == "__main__":
    unittest.main()
