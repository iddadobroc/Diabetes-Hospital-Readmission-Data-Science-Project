# %% [markdown]
# # Diabetes readmission: the cleaning, step by step
#
# NOVA IMS Enterprise Data Science Bootcamp, 2026-27.
#
# This notebook closes Phase 1 of the project plan (README, Steps 2 and 3). It does
# not implement the cleaning: the rules live in `clean()` (`src/features.py`), tested in
# `tests/test_src.py`, and every modelling notebook calls that same function. Here
# `clean()` is run with its step log, so that for every step we can see what happens to
# the rows, the columns and the `<30` rate, and check it against the decision log.
#
# 1. The raw data, loaded correctly (`src/data_prep.py`).
# 2. The cleaning, step by step.
# 3. What the cleaning leaves: the feature lists, no missing values.
# 4. The split by patient: the frozen modelling dataset.
# 5. The profiling report of the clean **training** rows.
#
# **About the target.** This notebook only shows the **overall** `<30` rate after each
# step, on all the encounters: it changes only when rows are removed, and it is already
# in the decision log. Rates broken down by feature are a look at the outcome, so they
# are computed on the training rows only, in `03_eda_target.ipynb`.

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 0. GET READY
# ----------------------------------------------------------------------

# %%
import hashlib
import sys
from pathlib import Path

PROJECT = next(
    p
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "src" / "data_prep.py").exists()
)
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import pandas as pd

from data_prep import add_mapping_descriptions, load_mappings, load_raw
from features import (
    CATEGORICAL,
    CONSTANT,
    DEATH_OR_HOSPICE,
    FEATURES,
    GROUP,
    NUMERIC,
    RARE_MEDICATIONS,
    REFERENCE,
    TARGET,
    clean,
    split_by_patient,
)

SEED = 42  # the seed of the project split (D-010)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 1. THE RAW DATA
# ----------------------------------------------------------------------
# **1.1** `load_raw()` reads `"?"` as missing and keeps `"None"` in `max_glu_serum` and
# `A1Cresult` as a category ("test not performed"); `add_mapping_descriptions()` adds
# the decoded `*_desc` columns of the three ID columns (D-001 to D-003). The `assert`
# checks the figures of the decision log.

# %%
mappings = load_mappings()
raw = add_mapping_descriptions(load_raw(), mappings)
assert raw.shape == (101_766, 53)
assert raw[GROUP].nunique() == 71_518
print(f"{len(raw):,} encounters of {raw[GROUP].nunique():,} patients, ", end="")
print(f"{raw.shape[1]} columns (50 of the file + 3 decoded descriptions)")

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 2. THE CLEANING, STEP BY STEP
# ----------------------------------------------------------------------
# **2.1** `clean()` with the proposals of the decision log (its defaults), recording the
# shape of the table and the `<30` rate after every step. Only steps 1a and 1b remove
# rows; the other steps add or drop columns.

# %%
log = []
df = clean(raw, log=log)

steps = pd.DataFrame(log).set_index("step")
steps.insert(1, "rows_removed", (-steps["rows"].diff()).fillna(0).astype(int))
steps.insert(3, "columns_change", steps["columns"].diff().fillna(0).astype(int))
steps.index.name = None
steps.style.format(
    {"readmitted_30_rate": "{:.2%}", "rows": "{:,}", "rows_removed": "{:,}"}
).format("{:+d}", subset=["columns_change"]).set_caption(
    "Rows, columns and overall <30 rate after each step of clean()"
)

# %% [markdown]
# **2.2 The rows removed (steps 1a and 1b).** A patient who died cannot be readmitted
# (D-004); hospice is end-of-life care, where a readmission is not something to prevent
# (D-005). Their codes come from the lookup table, by searching "Expired" and
# "Hospice". Code 21 is in the lookup table but never occurs.

# %%
codes = raw["discharge_disposition_id"].astype(int)
removed = pd.DataFrame(
    {
        "description": pd.Series(mappings["discharge_disposition_id"])[
            DEATH_OR_HOSPICE
        ],
        "encounters": [int((codes == c).sum()) for c in DEATH_OR_HOSPICE],
    }
).rename_axis("discharge_disposition_id")
assert removed["encounters"].sum() == len(raw) - len(df) == 2_423
removed

# %% [markdown]
# **2.3 The columns.** What `clean()` adds (the target and the engineered features) and
# what it drops (constant columns and `weight`, D-006; the 10 rare drugs, D-007; the raw
# outcome `readmitted`, D-011).

# %%
added = [c for c in df.columns if c not in raw.columns]
dropped = [c for c in raw.columns if c not in df.columns]
assert set(dropped) == set(CONSTANT + RARE_MEDICATIONS + ["readmitted"])
print(f"Added ({len(added)}):  ", ", ".join(added))
print(f"Dropped ({len(dropped)}):", ", ".join(dropped))

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 3. WHAT THE CLEANING LEAVES
# ----------------------------------------------------------------------
# **3.1** Every column of the clean table has one role: a feature, a reference column
# (kept for filtering and tracing, never a feature: identifiers and the raw codes that a
# feature was derived from), the patient (used only to group the splits) or the target.
# The `assert` lines check that the roles cover every column exactly once and that no
# identifier or answer is a feature (D-011, D-012).

# %%
roles = (
    {c: "numeric feature" for c in NUMERIC}
    | {c: "categorical feature" for c in CATEGORICAL}
    | {c: "reference (never a feature)" for c in REFERENCE}
    | {GROUP: "patient (groups the splits)", TARGET: "target"}
)
assert set(roles) == set(df.columns) and len(roles) == df.shape[1]
assert not set(FEATURES) & set(REFERENCE + [GROUP, TARGET])

columns = pd.DataFrame(
    {
        "role": pd.Series(roles),
        "dtype": df.dtypes.astype(str),
        "levels_or_values": df.nunique(),
        "missing": df.isna().sum(),
    }
).loc[df.columns]
columns.groupby("role").size().to_frame("columns")

# %% [markdown]
# **3.2 No missing values in the features.** Missing categories became an explicit
# level (D-008): `Unknown` for `race`, `payer_code` and `medical_specialty`, `Missing`
# for the diagnosis groups. The only missing values left are in the raw diagnosis codes,
# which are reference columns.

# %%
assert df[FEATURES].isna().sum().sum() == 0
columns[columns["missing"] > 0]

# %% [markdown]
# **3.3** The full list of columns, with their role, type and number of distinct values.

# %%
columns

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 4. THE SPLIT BY PATIENT: THE FROZEN DATASET
# ----------------------------------------------------------------------
# **4.1** `split_by_patient()` makes two `GroupShuffleSplit` cuts on `patient_nbr`: 20%
# of the patients become the test set, then 25% of the others the validation set (60 /
# 20 / 20, D-010 option A, seed 42). A patient is never in two sets (`assert` in the
# function). The `<30` rate of the test set is not shown: the test set is opened once,
# at the end of the modelling.

# %%
train, val, test = split_by_patient(df, seed=SEED)
assert (len(train), len(val), len(test)) == (59_814, 19_727, 19_802)

pd.DataFrame(
    {
        "encounters": [len(part) for part in (train, val, test)],
        "patients": [part[GROUP].nunique() for part in (train, val, test)],
        "share_of_encounters": [len(part) / len(df) for part in (train, val, test)],
        "readmitted_30_rate": [train[TARGET].mean(), val[TARGET].mean(), np.nan],
    },
    index=["train", "validation", "test (closed)"],
).style.format(
    {
        "encounters": "{:,}",
        "patients": "{:,}",
        "share_of_encounters": "{:.1%}",
        "readmitted_30_rate": "{:.2%}",
    },
    na_rep="closed",
)

# %% [markdown]
# **4.2 Why no parquet files.** The plan suggested saving the three sets in
# `data/processed/`. They are not needed: `data/` is not in Git, so a saved file would
# still be rebuilt by each member of the team, while `clean()` + `split_by_patient()`
# with seed 42 rebuild exactly the same sets in a few seconds, from tested code. A saved
# test set would also be one click away during the modelling.
#
# To check that two computers have the same frozen dataset, compare these fingerprints
# (a hash of the sorted `encounter_id` of each set): if they match, the sets are
# identical, row for row.


# %%
def fingerprint(part):
    """The first 12 hex digits of the SHA-256 of the sorted encounter ids."""
    ids = np.sort(part["encounter_id"].to_numpy()).astype("int64")
    return hashlib.sha256(ids.tobytes()).hexdigest()[:12]


pd.Series(
    {
        name: fingerprint(part)
        for name, part in [("train", train), ("validation", val), ("test", test)]
    },
    name="fingerprint",
).to_frame()

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 5. PROFILING REPORT OF THE CLEAN TRAINING ROWS
# ----------------------------------------------------------------------
# **5.1** The committed report `reports/df_diabetes_report.html` profiles the raw data.
# This one profiles what the models see: the features and the target of the **training
# rows** only. A report on all the clean rows would show the correlations of the
# target with every feature on the test patients too, a look at the outcome that the
# project rules keep on training rows.
#
# The report takes a few minutes, so it is built only when the file is missing (delete
# it to rebuild it). It stays local (`reports/*.html` is in `.gitignore`); open it in a
# browser.

# %%
REPORT = PROJECT / "reports" / "df_train_clean_report.html"

if not REPORT.exists():
    import warnings

    with warnings.catch_warnings():  # deprecation and pandas notices of the library
        warnings.simplefilter("ignore")
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            train[FEATURES + [TARGET]],
            title="Diabetes readmission: clean training rows (59,814 encounters)",
            explorative=True,
            progress_bar=False,
        )
        profile.to_file(REPORT)
print("Report:", REPORT.relative_to(PROJECT), f"({REPORT.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ----------------------------------------------------------------------
# # SUMMARY
# ----------------------------------------------------------------------
# - **Rows:** 101,766 → 99,343 encounters (−1,652 expired, −771 hospice), 69,990
#   patients. The overall `<30` rate goes from 11.16% to 11.39%, because the removed
#   encounters can never be readmitted.
# - **Columns:** 53 → 47: 8 engineered columns added (the target, the age order, three
#   diagnosis groups, two medication counts, the visits of the previous year), 14
#   dropped (2 constant drugs, `weight`, 10 rare drugs, the raw outcome).
# - **Roles:** 12 numeric and 25 categorical features, 8 reference columns, the
#   patient, the target. No missing value in any feature.
# - **Frozen dataset:** train / validation / test = 59,814 / 19,727 / 19,802
#   encounters, split by patient with seed 42, rebuilt by `src/` and checked by the
#   fingerprints above.
