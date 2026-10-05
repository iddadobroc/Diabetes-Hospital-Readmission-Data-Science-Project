"""Data loading utilities shared by the whole team.

Always load the raw data through these functions, never with a bare
pd.read_csv(): the default NaN handling of pandas is wrong for this dataset.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# The ID columns of diabetic_data.csv that IDS_mapping.csv decodes
ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

# Mapping descriptions that all mean "no information available"
UNKNOWN_DESCRIPTIONS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid", ""}


def load_raw(path=None):
    """Load diabetic_data.csv with the correct missing-value handling.

    - "?" is the real missing-value marker, so it becomes NaN.
    - "None" (e.g. in max_glu_serum and A1Cresult) means "test not performed":
      it is an informative category and must NOT become NaN.
    - The ID columns are codes, not quantities, so they are cast to category.
    """
    path = Path(path) if path else DATA_DIR / "diabetic_data.csv"
    df = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)
    df[ID_COLUMNS] = df[ID_COLUMNS].astype("category")
    return df


def load_mappings(path=None):
    """Parse IDS_mapping.csv into one {id: description} dict per ID column.

    The file is really 3 tables stacked on top of each other, separated by
    empty lines, each one with its own header row.
    """
    path = Path(path) if path else DATA_DIR / "IDS_mapping.csv"
    mappings = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, description = line.partition(",")
        key, description = key.strip(), description.strip()
        if not key:
            # Empty separator line between two tables
            current = None
        elif current is None:
            # Header row: its first field is the name of the ID column
            current = key
            mappings[current] = {}
        else:
            mappings[current][int(key)] = description
    return mappings


def add_mapping_descriptions(df, mappings, unknown_label="Unknown"):
    """Add a *_desc column for each ID column, keeping the original IDs.

    Descriptions that carry no information (NULL, Not Mapped, ...) are merged
    into a single unknown_label category.
    """
    df = df.copy()
    for col, mapping in mappings.items():
        desc = df[col].astype(int).map(mapping)
        desc = desc.where(~desc.isin(UNKNOWN_DESCRIPTIONS), unknown_label)
        df[col.replace("_id", "_desc")] = desc.fillna(unknown_label).astype("category")
    return df
