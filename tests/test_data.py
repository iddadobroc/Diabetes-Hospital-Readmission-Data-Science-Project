"""Tests on the data, run BEFORE training (course, W9-W10: test the data first).

They read `data/diabetic_data.csv` and are skipped when it is absent, so that
`python -m unittest` still runs on a computer without the data. In the GitHub pipeline
`scripts/fetch_data.py` downloads the files first, so the tests always run there.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import add_mapping_descriptions, load_mappings, load_raw  # noqa: E402
from features import (  # noqa: E402
    DEATH_OR_HOSPICE,
    FEATURES,
    SPLIT_FINGERPRINTS,
    SPLIT_SEED,
    TARGET,
    clean,
    fingerprint,
    split_by_patient,
)

DATA = ROOT / "data" / "diabetic_data.csv"


@unittest.skipUnless(DATA.exists(), "data/diabetic_data.csv not found")
class TestData(unittest.TestCase):
    """The raw file is the one the project was built on, and cleans as expected."""

    @classmethod
    def setUpClass(cls):
        """Load and clean once for all the tests."""
        cls.raw = load_raw()
        cls.decoded = add_mapping_descriptions(cls.raw, load_mappings())
        cls.clean = clean(cls.decoded)

    def test_shape_and_identifiers(self):
        """101,766 encounters of 71,518 patients; encounter_id is a unique key."""
        self.assertEqual(len(self.raw), 101_766)
        self.assertEqual(self.raw["patient_nbr"].nunique(), 71_518)
        self.assertTrue(self.raw["encounter_id"].is_unique)

    def test_missing_markers(self):
        """'?' is missing; 'None' in the lab tests is a level (D-001)."""
        self.assertFalse((self.raw == "?").any().any())
        self.assertEqual(self.raw["weight"].isna().sum(), 98_569)
        self.assertIn("None", set(self.raw["A1Cresult"]))
        self.assertIn("None", set(self.raw["max_glu_serum"]))

    def test_values_in_range(self):
        """Counts are non-negative, stays 1-14 days, the outcome has three levels."""
        counts = ["number_outpatient", "number_emergency", "number_inpatient",
                  "num_lab_procedures", "num_procedures", "num_medications"]
        self.assertTrue((self.raw[counts] >= 0).all().all())
        self.assertTrue(self.raw["time_in_hospital"].between(1, 14).all())
        self.assertEqual(set(self.raw["readmitted"]), {"<30", ">30", "NO"})

    def test_clean_removes_death_and_hospice(self):
        """99,343 encounters left, none ending in death or hospice (D-004, D-005)."""
        self.assertEqual(len(self.clean), 99_343)
        codes = self.clean["discharge_disposition_id"].astype(int)
        self.assertFalse(codes.isin(DEATH_OR_HOSPICE).any())
        self.assertAlmostEqual(self.clean[TARGET].mean(), 0.1139, places=4)

    def test_features_complete(self):
        """Every feature exists and has no missing value after cleaning."""
        self.assertFalse(self.clean[FEATURES].isna().any().any())

    def test_split_is_the_frozen_one(self):
        """The three sets have the fingerprints of the decision log (D-010)."""
        parts = split_by_patient(self.clean, SPLIT_SEED)
        found = dict(zip(["train", "validation", "test"], map(fingerprint, parts)))
        self.assertEqual(found, SPLIT_FINGERPRINTS)


if __name__ == "__main__":
    unittest.main()
