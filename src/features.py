"""Deterministic cleaning and the feature lists, shared by every modelling notebook.

The rules come from Luis Alcaide's notebook `LA_Models/01_preprocessing_and_models.ipynb`
(Stage 1) and from the decision log (`docs/decision_log.md`), moved here so that every
notebook cleans the data in exactly the same way instead of copying the code.

Everything in this file is a fixed rule applied row by row: nothing is learned from the
data (no mean, frequency or threshold). That is why `clean` can safely run on the whole
dataset BEFORE the split. Anything that learns from the data (rare levels, scaling,
encoding) belongs to the model pipeline and is fitted on the training rows only.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Discharge codes found in IDS_mapping.csv by searching "Expired" (D-004) and
# "Hospice" (D-005); 21 never occurs in the data.
EXPIRED = [11, 19, 20, 21]
HOSPICE = [13, 14]
DEATH_OR_HOSPICE = sorted(EXPIRED + HOSPICE)

# Constant columns and weight (D-006)
CONSTANT = ["examide", "citoglipton", "weight"]

MEDICATIONS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Drugs with fewer than 100 non-"No" rows (D-007)
RARE_MEDICATIONS = [
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "troglitazone",
    "glipizide-metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
    "chlorpropamide",
]
KEPT_MEDICATIONS = [
    m for m in MEDICATIONS if m not in RARE_MEDICATIONS and m not in CONSTANT
]

TARGET = "readmitted_30"  # D-011
GROUP = "patient_nbr"  # used only to group the splits (D-012)

# Identifiers and raw source values: kept for filtering and tracing, NEVER features
REFERENCE = [
    "encounter_id",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "age",
    "diag_1",
    "diag_2",
    "diag_3",
]
# Counts of previous visits: very skewed (most patients 0)
SKEWED = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "n_visits_last_year",
]
NUMERIC = [
    "age_band_order",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "n_meds_active",
    "n_meds_changed",
] + SKEWED
CATEGORICAL = [
    "race",
    "gender",
    "payer_code",
    "medical_specialty",
    "admission_type_desc",
    "discharge_disposition_desc",
    "admission_source_desc",
    "max_glu_serum",
    "A1Cresult",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "change",
    "diabetesMed",
] + KEPT_MEDICATIONS
FEATURES = NUMERIC + CATEGORICAL

# Alternative of D-009: medical_specialty grouped by clinical area, a fixed mapping
# written by hand from the level names (not from their readmission rates). Levels not
# listed here go to "Other"; missing and "PhysicianNotFound" go to "Unknown".
SPECIALTY_GROUPS = {
    "InternalMedicine": ["InternalMedicine", "Hospitalist", "Resident"],
    "GeneralPractice": ["Family/GeneralPractice", "Osteopath", "OutreachServices"],
    "Emergency": ["Emergency/Trauma"],
    "Cardiology": ["Cardiology"],
    "Nephrology": ["Nephrology"],
    "Surgery": [
        "Surgery-General",
        "Surgery-Cardiovascular/Thoracic",
        "Surgery-Cardiovascular",
        "Surgery-Thoracic",
        "Surgery-Vascular",
        "Surgery-Neuro",
        "Surgery-Plastic",
        "Surgery-PlasticwithinHeadandNeck",
        "Surgery-Colon&Rectal",
        "Surgery-Maxillofacial",
        "Surgeon",
        "SurgicalSpecialty",
        "Orthopedics",
        "Orthopedics-Reconstructive",
        "Urology",
        "Otolaryngology",
        "Ophthalmology",
        "Podiatry",
        "Anesthesiology",
        "Proctology",
        "Dentistry",
    ],
    "Oncology/Hematology": [
        "Oncology",
        "Hematology",
        "Hematology/Oncology",
        "Obsterics&Gynecology-GynecologicOnco",
    ],
    "MedicalSubspecialty": [
        "Pulmonology",
        "Gastroenterology",
        "Neurology",
        "Neurophysiology",
        "Endocrinology",
        "Endocrinology-Metabolism",
        "InfectiousDiseases",
        "Rheumatology",
        "AllergyandImmunology",
        "Dermatology",
    ],
    "Psychiatry": [
        "Psychiatry",
        "Psychiatry-Child/Adolescent",
        "Psychiatry-Addictive",
        "Psychology",
    ],
    "ObGyn": ["ObstetricsandGynecology", "Gynecology", "Obstetrics", "Perinatology"],
    "Pediatrics": [
        "Pediatrics",
        "Pediatrics-Endocrinology",
        "Pediatrics-CriticalCare",
        "Pediatrics-Pulmonology",
        "Pediatrics-Neurology",
        "Pediatrics-Hematology-Oncology",
        "Pediatrics-AllergyandImmunology",
        "Pediatrics-EmergencyMedicine",
        "Pediatrics-InfectiousDiseases",
        "Anesthesiology-Pediatric",
        "Surgery-Pediatric",
        "Cardiology-Pediatric",
    ],
    "Diagnostics": ["Radiologist", "Radiology", "Pathology"],
    "Rehabilitation": ["PhysicalMedicineandRehabilitation", "SportsMedicine", "Speech"],
    "Unknown": ["Unknown", "PhysicianNotFound"],
}
SPECIALTY_TO_GROUP = {
    level: group for group, levels in SPECIALTY_GROUPS.items() for level in levels
}

# Extra columns created only by the alternative cleaning options
EXTRA_NUMERIC = ["hospice", "weight_recorded"]
EXTRA_CATEGORICAL = ["other_meds"]


def feature_lists(df):
    """The numeric and categorical feature columns of a cleaned table.

    Args:
        df (pandas.DataFrame): The output of `clean`, with any options.

    Returns:
        tuple: `(numeric, categorical)`, the base lists plus the extra columns that
        the cleaning options added to `df`.
    """
    numeric = NUMERIC + [c for c in EXTRA_NUMERIC if c in df.columns]
    categorical = CATEGORICAL + [c for c in EXTRA_CATEGORICAL if c in df.columns]
    return numeric, categorical


def icd9_group(code):
    """Map one ICD-9 code to one of the Strack et al. (2014) diagnosis groups.

    V and E codes are handled before any numeric conversion, and 250.xx before the
    generic ranges.

    Args:
        code (str or NaN): A raw diagnosis string, such as "250.83", "428" or "V57".

    Returns:
        str: The group name; "Missing" for a missing code, "Other" for V/E codes and
        every code outside the groups of the project brief.
    """
    if pd.isna(code):
        return "Missing"
    if code.startswith(("V", "E")):
        return "Other"
    whole = int(float(code))
    if whole == 250:
        return "Diabetes"
    if 390 <= whole <= 459 or whole == 785:
        return "Circulatory"
    if 460 <= whole <= 519 or whole == 786:
        return "Respiratory"
    if 520 <= whole <= 579 or whole == 787:
        return "Digestive"
    if 580 <= whole <= 629 or whole == 788:
        return "Genitourinary"
    if 710 <= whole <= 739:
        return "Musculoskeletal"
    if 800 <= whole <= 999:
        return "Injury"
    if 140 <= whole <= 239:
        return "Neoplasms"
    return "Other"


def clean(df, hospice="drop", rare_meds="drop", specialty="raw", weight_flag=False):
    """Apply the fixed, row-by-row cleaning rules. Learns nothing from the data.

    The default options are the proposals of the decision log; the other values are
    the alternatives listed in the README (Step 1), so that both can be tested.

    Args:
        df (pandas.DataFrame): The output of
            `add_mapping_descriptions(load_raw(), load_mappings())`.
        hospice (str): "drop" removes the hospice encounters (D-005, proposal);
            "flag" keeps them and adds a 0/1 `hospice` column (alternative).
        rare_meds (str): "drop" removes the rare drugs (D-007, proposal); "group"
            also adds `other_meds`, "Yes" if any rare drug is not "No" (alternative).
        specialty (str): "raw" keeps the `medical_specialty` labels, grouped later by
            the pipeline (D-009, proposal); "clinical" maps them to the clinical areas
            of `SPECIALTY_GROUPS` (alternative).
        weight_flag (bool): True adds a 0/1 `weight_recorded` column before `weight`
            is dropped (alternative mentioned in D-006).

    Returns:
        pandas.DataFrame: One row per kept encounter, with the target `readmitted_30`,
        the features, the `REFERENCE` columns and `patient_nbr`. The raw
        `readmitted` column is dropped, because it is the answer.
    """
    df = df.copy()

    # 1. Outcome not observable: death (D-004) and, unless flagged, hospice (D-005)
    codes = df["discharge_disposition_id"].astype(int)
    if hospice == "flag":
        df["hospice"] = codes.isin(HOSPICE).astype(int)
        df = df[~codes.isin(EXPIRED)]
    else:
        df = df[~codes.isin(DEATH_OR_HOSPICE)]

    # 2. Binary target; the raw column is the answer, so it goes (D-011)
    df[TARGET] = (df["readmitted"] == "<30").astype(int)

    # 3. Age band -> its position, 0 to 9: "[70-80)" -> 7 (an order, not an age)
    df["age_band_order"] = df["age"].str[1:].str.split("-").str[0].astype(int) // 10

    # 4. Missing values as an explicit level (D-008)
    for col in ["race", "payer_code", "medical_specialty"]:
        df[col] = df[col].fillna("Unknown")
    if specialty == "clinical":
        df["medical_specialty"] = (
            df["medical_specialty"].map(SPECIALTY_TO_GROUP).fillna("Other")
        )
    if weight_flag:
        df["weight_recorded"] = df["weight"].notna().astype(int)
    if rare_meds == "group":
        any_rare = (df[RARE_MEDICATIONS] != "No").any(axis=1)
        df["other_meds"] = any_rare.map({True: "Yes", False: "No"})

    # 5. Diagnoses -> clinical groups (map on the unique codes, much faster)
    for col in ["diag_1", "diag_2", "diag_3"]:
        groups = {code: icd9_group(code) for code in df[col].unique()}
        df[f"{col}_group"] = df[col].map(groups)

    # 6. Medication summaries, computed on ALL drugs before dropping the rare ones
    df["n_meds_active"] = (df[MEDICATIONS] != "No").sum(axis=1)
    df["n_meds_changed"] = df[MEDICATIONS].isin(["Up", "Down"]).sum(axis=1)

    # 7. Total use of the hospital in the previous year
    df["n_visits_last_year"] = df[
        ["number_outpatient", "number_emergency", "number_inpatient"]
    ].sum(axis=1)

    # 8. Drop what is neither a feature nor a reference column (D-006, D-007)
    df = df.drop(columns=CONSTANT + RARE_MEDICATIONS + ["readmitted"])
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].astype(str)
    return df.reset_index(drop=True)


def first_encounter_only(df):
    """Keep only the first encounter of each patient (D-010, option B, Strack et al.).

    "First" is the smallest `encounter_id` of the patient among the rows of `df`.
    The decision log (D-004) notes that `encounter_id` may not be strictly
    chronological, so this is an approximation. It looks at other rows of the SAME
    patient only, and a patient is never split across sets, so it is safe before
    the split.

    Args:
        df (pandas.DataFrame): A table with `patient_nbr` and `encounter_id`.

    Returns:
        pandas.DataFrame: One row per patient.
    """
    first = df.groupby(GROUP)["encounter_id"].transform("min")
    return df[df["encounter_id"] == first]


def split_by_patient(df, seed, test_size=0.20, val_size=0.25):
    """Split into train, validation and test, never putting a patient in two sets.

    Same two cuts as Luis's notebook (D-010, option A): 20% of the patients become
    the test set, then 25% of the remaining patients (20% of the total) the
    validation set. With the same seed, the three sets are identical to his.

    Args:
        df (pandas.DataFrame): The output of `clean`.
        seed (int): The random seed of the two `GroupShuffleSplit`.
        test_size (float): Share of the patients in the test set.
        val_size (float): Share of the remaining patients in the validation set.

    Returns:
        tuple: `(train, val, test)`, three DataFrames.

    Raises:
        AssertionError: If a patient ends up in two sets.
    """
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, groups=df[GROUP]))
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(inner.split(dev, groups=dev[GROUP]))
    train, val = dev.iloc[train_idx], dev.iloc[val_idx]

    patients = [set(part[GROUP]) for part in (train, val, test)]
    assert not patients[0] & patients[1], "a patient is in train and validation"
    assert not patients[0] & patients[2], "a patient is in train and test"
    assert not patients[1] & patients[2], "a patient is in validation and test"
    return train, val, test
