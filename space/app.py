"""Readmission risk at discharge: the Hugging Face app of the project (Gradio).

The app loads the version of the model that the GitHub pipeline tested and published
(`model_repo.txt`, `model_version.txt`), and scores discharges with the same code as
the notebooks (`features.py`, `scoring.py`, copied next to this file by the pipeline):

1. **Patient at discharge**: the risk of one patient, as a number with its unit, the
   follow-up tier it implies, and the reasons behind it.
2. **Discharge worklist**: a list of discharges ranked by risk, cut at the follow-up
   capacity of the hospital.
3. **Evidence**: how the model was measured on 19,802 test discharges it never saw.
4. **About**: intended use, uses it must not have, data and limits.

Charts follow the IBCS standards: a message first, a three-line title, no gauges or
traffic lights, red for what raises the risk, green for what lowers it.

Run locally (after `python train.py`): `python space/app.py`, then open
http://127.0.0.1:7860.
"""

import json
import sys
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# The pipeline copies the shared modules next to the app; locally they are in src/
sys.path.insert(0, str(HERE))
if not (HERE / "scoring.py").exists():
    sys.path.insert(0, str(HERE.parent / "src"))

try:  # Hugging Face ZeroGPU runtime; absent when the app runs locally
    import spaces
except ImportError:
    spaces = None

import ibcs  # noqa: E402
from features import DEATH_OR_HOSPICE, FEATURES  # noqa: E402
from scoring import (  # noqa: E402
    assign_tiers,
    contributions,
    inner_pipeline,
    prepare_discharges,
    raw_scores,
)


# --- ZeroGPU ------------------------------------------------------------------------
# A ZeroGPU Space must declare at least one @spaces.GPU function (course, W8). The
# model is a tree ensemble that runs on CPU, so the marker below is never called: the
# predictions do not use the daily GPU quota, and the live demo cannot run out of it.
if spaces is not None:

    @spaces.GPU
    def zerogpu_marker():
        """Never called: only satisfies the ZeroGPU startup check."""
        return None


# --- The published model ----------------------------------------------------------


def read_text(name, default):
    """The content of a small text file next to the app, or `default`."""
    path = HERE / name
    return path.read_text().strip() if path.exists() else default


MODEL_REPO = read_text("model_repo.txt", "not published yet")
VERSION = read_text("model_version.txt", "local")

if MODEL_REPO != "not published yet":
    from huggingface_hub import hf_hub_download

    def fetch(name):
        """Download one file of the model repository, at the published tag."""
        return hf_hub_download(MODEL_REPO, name, revision=VERSION)

else:  # local run: the files written by train.py

    def fetch(name):
        """Read one file of models/, written by train.py."""
        return str(HERE.parent / "models" / name)

    VERSION = "local build"

MODEL = joblib.load(fetch("model.joblib"))
CONFIG = json.loads(Path(fetch("config.json")).read_text())
METRICS = json.loads(Path(fetch("metrics.json")).read_text())
SAMPLE_PATH = fetch("sample_discharges.csv")
FIGURES = HERE / "figures" if (HERE / "figures").exists() else HERE.parent / "reports" / "figures"
MAPPINGS = {
    col: {int(k): v for k, v in m.items()} for col, m in CONFIG["mappings"].items()
}
TYPICAL = CONFIG["typical"]
BASE_RATE = METRICS["test"]["rate"]
# Above 60% the isotonic calibration rests on a few dozen patients (observed rate about
# 64% on test) and its last step reaches 100% on 3 of them: risks above this are shown as
# "over 60%", never as a near-certainty (model card, limitations).
MAX_SHOWN = 0.60
CUTOFFS = CONFIG["cutoffs_raw"]

TIERS = {
    "High": (
        "the 10% of discharges with the highest risk",
        "Phone call within 48 hours and a follow-up visit within 7 days.",
    ),
    "Medium": (
        "the next 20% of discharges by risk",
        "Reminder and medication check by phone within 14 days.",
    ),
    "Low": ("the other 70% of discharges", "Standard discharge instructions."),
}

READABLE = {
    "number_inpatient": "Inpatient stays, previous year",
    "discharge_disposition_desc": "Discharge destination",
    "payer_code": "Payer",
    "medical_specialty": "Admitting specialty",
    "n_visits_last_year": "All visits, previous year",
    "num_medications": "Medications given",
    "diag_1_group": "Primary diagnosis",
    "diag_2_group": "Secondary diagnosis",
    "diag_3_group": "Third diagnosis",
    "age_band_order": "Age band",
    "number_diagnoses": "Number of diagnoses",
    "num_lab_procedures": "Lab procedures",
    "time_in_hospital": "Days in hospital",
    "admission_source_desc": "Admission source",
    "admission_type_desc": "Admission type",
    "n_meds_active": "Diabetes drugs prescribed",
    "n_meds_changed": "Diabetes drugs changed",
    "num_procedures": "Procedures",
    "number_emergency": "Emergency visits, previous year",
    "number_outpatient": "Outpatient visits, previous year",
    "A1Cresult": "HbA1c result",
    "max_glu_serum": "Glucose serum test",
    "insulin": "Insulin",
    "race": "Race",
    "gender": "Gender",
    "change": "Diabetes medication changed",
    "diabetesMed": "Diabetes medication",
}

FOOTNOTE = (
    "Source: UCI Diabetes 130-US hospitals 1999..2008 (Strack et al. 2014). Model "
    f"{VERSION}: XGBoost with isotonic calibration, fitted on "
    f"{METRICS['fitted_on']['encounters']:,} discharges. Risk = estimated probability of an "
    "inpatient readmission within 30 days. Encounters ending in death or hospice are "
    "out of scope. Educational prototype: not a medical device; every decision needs a "
    "clinician."
)


# --- Choices of the form ------------------------------------------------------------


def described(col, exclude=()):
    """(description -> code) for an ID column, informative descriptions only."""
    out = {}
    for code, text in sorted(MAPPINGS[col].items()):
        if code in exclude or text in {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid"}:
            continue
        if text.startswith("Expired") or text.startswith("Hospice"):
            continue
        out.setdefault(text, code)
    return out


DISCHARGE = described("discharge_disposition_id", exclude=DEATH_OR_HOSPICE)
SOURCE = described("admission_source_id")
# Keep the destinations and sources seen in the data, most frequent first
DISCHARGE = {d: DISCHARGE[d] for d in CONFIG["choices"]["discharge_disposition_desc"] if d in DISCHARGE}
SOURCE = {s: SOURCE[s] for s in CONFIG["choices"]["admission_source_desc"] if s in SOURCE}
AGE_BANDS = [f"[{10 * i}-{10 * (i + 1)})" for i in range(10)]
# One representative ICD-9 code per diagnosis group (Strack et al. groups)
DIAGNOSIS = {
    "Circulatory (e.g. heart failure, 428)": "428",
    "Respiratory (e.g. pneumonia, 486)": "486",
    "Digestive (e.g. pancreatitis, 577)": "577",
    "Diabetes (250.xx)": "250.8",
    "Injury (e.g. hip fracture, 820)": "820",
    "Musculoskeletal (e.g. osteoarthrosis, 715)": "715",
    "Genitourinary (e.g. urinary infection, 599)": "599",
    "Neoplasms (e.g. breast cancer, 174)": "174",
    "Other (e.g. general symptoms, 780)": "780",
}
SPECIALTIES = [s for s in CONFIG["choices"]["medical_specialty"] if s != "Unknown"][:20]
SPECIALTIES = ["Not recorded"] + SPECIALTIES


def form_row(age, discharge, source, days, inpatient, emergency, outpatient, meds,
             labs, diagnoses, diagnosis, a1c, insulin, specialty):
    """One raw discharge (UCI columns) from the answers of the form."""
    return pd.DataFrame([{
        "age": age,
        "discharge_disposition_id": DISCHARGE[discharge],
        "admission_source_id": SOURCE[source],
        "admission_type_id": 1 if SOURCE[source] == 7 else 3,  # emergency or elective
        "time_in_hospital": int(days),
        "number_inpatient": int(inpatient),
        "number_emergency": int(emergency),
        "number_outpatient": int(outpatient),
        "num_medications": int(meds),
        "num_lab_procedures": int(labs),
        "number_diagnoses": int(diagnoses),
        "diag_1": DIAGNOSIS[diagnosis],
        "A1Cresult": a1c,
        "insulin": insulin,
        "diabetesMed": "Yes" if insulin != "No" else TYPICAL["diabetesMed"],
        "change": "Ch" if insulin in ("Up", "Down") else "No",
        "medical_specialty": np.nan if specialty == "Not recorded" else specialty,
    }])


def risk_text(risk, decimals=1):
    """The risk as text, capped at MAX_SHOWN ("over 60%")."""
    if risk > MAX_SHOWN:
        return f"over {100 * MAX_SHOWN:.0f}%"
    return f"{100 * risk:.{decimals}f}%"


def value_of(feature, row):
    """The value of one feature of a prepared discharge, as short text."""
    value = row[feature]
    if feature == "age_band_order":
        return f"{10 * int(value)}-{10 * int(value) + 9} years"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return f"{int(value)}"
    text = str(value)
    for prefix in ("Discharged/transferred/referred to ", "Discharged/transferred to "):
        text = text.removeprefix(prefix)
    return text if len(text) <= 30 else text[:28] + "…"


# --- Tab 1: one patient -------------------------------------------------------------


def patient_chart(risk, reasons, row):
    """The IBCS page of one patient: risk against the average, and the reasons."""
    fig = plt.figure(figsize=(8.6, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 3.2], hspace=0.35)
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    shown = min(risk, MAX_SHOWN)
    ibcs.bars_h(ax1, ["This discharge", "Average discharge"],
                [100 * shown, 100 * BASE_RATE], decimals=1, unit="%", highlight=0,
                width=0.5, xmax=max(100 * shown, 100 * BASE_RATE) * 1.3,
                texts=[risk_text(risk), risk_text(BASE_RATE)])
    labels = [f"{READABLE.get(f, f)}: {value_of(f, row)}" for f in reasons.index]
    values = reasons.to_numpy()
    y = np.arange(len(values))
    for i, v in enumerate(values):
        ax2.barh(y[i], v, height=0.6, color=ibcs.BAD if v > 0 else ibcs.GOOD)
        ax2.text(v, y[i], (" " if v > 0 else "") + ibcs.fmt_number(v, 2, sign=True)
                 + ("" if v > 0 else " "), va="center",
                 ha="left" if v > 0 else "right", fontsize=9)
    ax2.axvline(0, color=ibcs.AC, linewidth=1.2)
    span = np.abs(values).max() * 1.5
    ax2.set_xlim(-span, span)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xticks([])
    ibcs.clean_axes(ax2)
    ax2.set_title("Why: effect of each feature on the risk score (red raises it, green lowers it)",
                  fontsize=9, loc="left", color=ibcs.MUTED)
    ratio = shown / BASE_RATE
    ibcs.page(
        fig,
        f"This discharge has a {risk_text(risk, 0)} risk of readmission within 30 days, "
        f"{'over ' if risk > MAX_SHOWN else ''}{ratio:.1f} times the average discharge",
        "One discharge, scored by the published model",
        "Risk of readmission < 30 days", "in %, and effect of each feature in log-odds",
        f"Model {VERSION}; average = test discharges 1999..2008",
    )
    ibcs.footnote(fig, "Effects are TreeSHAP contributions to the model's score: "
                       "associations in the data, not causes. The 8 largest are shown.")
    return fig


def score_patient(*answers):
    """Risk, tier, action and reasons for the discharge described by the form."""
    raw = form_row(*answers)
    prepared = prepare_discharges(raw, TYPICAL, MAPPINGS)
    X = prepared[FEATURES]
    risk = float(MODEL.predict_proba(X)[0, 1])
    tier = assign_tiers(raw_scores(MODEL, X), CUTOFFS)[0]
    effect = contributions(inner_pipeline(MODEL), X, FEATURES).drop(columns="bias").iloc[0]
    reasons = effect.reindex(effect.abs().sort_values(ascending=False).index)[:8]
    who, action = TIERS[tier]
    answer = (
        f"## {risk_text(risk)} risk of readmission within 30 days\n"
        f"Average discharge: {100 * BASE_RATE:.1f}%. **{tier} tier**: {who}.\n\n"
        f"**Suggested follow-up:** {action}\n\n"
        f"<small>The tier comes from the hospital's capacity (10% High, 20% Medium), "
        f"not from a fixed risk threshold. The follow-up actions are an example "
        f"protocol, to be set by the care team.</small>"
    )
    return answer, patient_chart(risk, reasons, prepared.iloc[0])


# --- Tab 2: the worklist ------------------------------------------------------------


def read_discharges(path):
    """Read a CSV in the UCI format ('?' = missing, 'None' = test not performed)."""
    return pd.read_csv(path, keep_default_na=False, na_values=["?"],
                       dtype={"diag_1": str, "diag_2": str, "diag_3": str})


def worklist_chart(n, k, reached, random_reached, observed):
    """IBCS page: readmissions reached by the ranking against a random choice."""
    fig, (ax, axv) = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.05})
    ibcs.bars_h(ax, ["Model ranking", "Random order"], [reached, random_reached],
                decimals=0 if observed else 1, highlight=0, width=0.5)
    ibcs.variance_h(axv, [reached - random_reached, np.nan],
                    decimals=0 if observed else 1, xmax=max(reached, 1) * 0.6)
    axv.set_title("ΔRandom", fontsize=9, loc="left")
    kind = "observed" if observed else "expected"
    ibcs.page(
        fig,
        f"Calling the {k} discharges with the highest risk ({100 * k / n:.0f}%) would reach "
        f"{reached:.0f} readmissions ({kind}), {reached - random_reached:.0f} more than "
        f"calling {k} discharges at random",
        f"{n} uploaded discharges", f"Readmissions < 30 days reached ({kind})",
        "in #, by way of choosing whom to call",
        f"Model {VERSION}; capacity {100 * k / n:.0f}% of the discharges",
    )
    ibcs.footnote(fig, "Observed: the file has the outcome (readmitted). Expected: the "
                       "sum of the calibrated risks of the called discharges. Random: the "
                       "readmission rate of the file times the number of calls.")
    return fig


def build_worklist(file, capacity):
    """Rank the discharges of the file (or of the sample) and cut at the capacity."""
    data = read_discharges(file if file else SAMPLE_PATH)
    out_of_scope = data["discharge_disposition_id"].astype(int).isin(DEATH_OR_HOSPICE)
    data = data[~out_of_scope].reset_index(drop=True)
    prepared = prepare_discharges(data, TYPICAL, MAPPINGS)
    X = prepared[FEATURES]
    risk = MODEL.predict_proba(X)[:, 1]
    raw = raw_scores(MODEL, X)
    effect = contributions(inner_pipeline(MODEL), X, FEATURES).drop(columns="bias")
    n = len(data)
    k = int(np.ceil(capacity / 100 * n))
    order = np.argsort(-raw, kind="stable")
    called = order[:k]
    observed = "readmitted" in data.columns
    if observed:
        outcome = (data["readmitted"] == "<30").to_numpy()
        reached, random_reached = outcome[called].sum(), outcome.mean() * k
    else:
        reached, random_reached = risk[called].sum(), risk.mean() * k
    table = pd.DataFrame({
        "Rank": np.arange(1, k + 1),
        "Encounter": data.loc[called, "encounter_id"].to_numpy()
        if "encounter_id" in data.columns else called + 1,
        "Risk": [risk_text(r) for r in risk[called]],
        "Tier": assign_tiers(raw[called], CUTOFFS),
        "Main reason": [READABLE.get(f, f) for f in effect.iloc[called].idxmax(axis=1)],
        "Age": prepared.loc[called, "age"].to_numpy(),
        "Days in hospital": prepared.loc[called, "time_in_hospital"].to_numpy(),
        "Inpatient stays, previous year": prepared.loc[called, "number_inpatient"].to_numpy(),
    })
    if observed:
        table["Readmitted < 30 days"] = np.where(outcome[called], "yes", "no")
    note = (
        f"**{n} discharges** scored"
        + (f" ({int(out_of_scope.sum())} ending in death or hospice left out)" if out_of_scope.any() else "")
        + f". Call list: the **{k}** with the highest risk ({capacity:.0f}% capacity)."
    )
    return note, worklist_chart(n, k, float(reached), float(random_reached), observed), table


# --- Tab 3: the evidence ------------------------------------------------------------


def evidence_markdown():
    """The test results of the published version, as a short table."""
    m, b = METRICS["metrics"], METRICS["baselines"]
    t, r = METRICS["test"], METRICS["reached_at_10pct"]
    return (
        f"### Measured once on {t['encounters']:,} test discharges of {t['patients']:,} "
        f"patients the model never saw ({t['readmissions']:,} readmissions, "
        f"{100 * t['rate']:.1f}%)\n\n"
        "| Ranking method | PR-AUC | ROC-AUC | Readmissions reached with 10% capacity |\n"
        "|---|---:|---:|---:|\n"
        f"| **XGBoost (this model)** | **{m['pr_auc']:.3f}** | **{m['roc_auc']:.3f}** | "
        f"**{r['XGBoost (final)']:,}** |\n"
        f"| Logistic regression | {b['Logistic regression']['pr_auc']:.3f} | "
        f"{b['Logistic regression']['roc_auc']:.3f} | {r['Logistic regression']:,} |\n"
        f"| LACE index (clinical score) | {b['LACE index']['pr_auc']:.3f} | "
        f"{b['LACE index']['roc_auc']:.3f} | {r['LACE index']:,} |\n"
        f"| Random order | {t['rate']:.3f} | 0.500 | {r['Random order']:.0f} |\n\n"
        f"Calibration: mean estimated risk {100 * t['mean_risk']:.1f}% against "
        f"{100 * t['rate']:.1f}% observed. Quality gate (decision log D-032): passed. "
        f"Model {VERSION}, published by the GitHub pipeline."
    )


FIGURE_FILES = [
    ("01_reached_at_capacity.png", "Model against baselines at 10% capacity"),
    ("02_capture_curve.png", "Readmissions reached by capacity"),
    ("03_calibration.png", "Calibration: observed against estimated"),
    ("04_fairness_high_tier.png", "Fairness: the High tier by group"),
    ("05_risk_drivers.png", "What drives the risk"),
    ("06_model_families_validation.png", "Model families on validation"),
]

ABOUT = f"""
## Readmission risk at discharge, a decision-support prototype

**Question.** At the moment of discharge, which diabetic patients should get extra
follow-up first, when the team can call only some of them?

**What the app does.** It estimates each discharge's risk of an inpatient readmission
within 30 days and ranks the discharges, so that limited follow-up goes first to the
patients most likely to come back. The tiers follow the capacity of the hospital
(High = 10% of discharges, Medium = next 20%), not a fixed threshold.

**Intended use.** Prioritising post-discharge follow-up, with a clinician reviewing
every list. Teaching and demonstration.

**It must not be used for:** diagnosis or treatment decisions; denying or limiting
care; decisions without human review; real patients' data (this is a prototype trained
on 1999-2008 US data); claiming that follow-up prevents readmissions (the model ranks
risk, it does not measure the effect of an intervention).

**Data.** UCI Diabetes 130-US hospitals, 1999-2008 (Strack et al. 2014): 101,766
encounters of 71,518 patients. Encounters ending in death or hospice are excluded
(no readmission can be observed). Patients are split 60 / 20 / 20 into train,
validation and test, never with one patient in two sets.

**Limits.** Readmissions to hospitals outside the network are recorded as "no"; the
data is 15 to 25 years old; only 3 diagnosis codes per encounter; the model is better
at ranking (PR-AUC {METRICS['metrics']['pr_auc']:.2f}, against {BASE_RATE:.2f} at
random) than at certainty: most High-tier patients are not readmitted. Group results
are in the Evidence tab; older patients' readmissions tend to reach the High tier less
often (21% at 80 and over, 25% overall).

**Project.** NOVA IMS, Enterprise Data Science Bootcamp 2026. Code, decision log and
model card: see the GitHub repository and the model repository `{MODEL_REPO}`.
"""

# --- The page ------------------------------------------------------------------------

with gr.Blocks(title="Readmission risk at discharge") as demo:
    gr.Markdown(
        "# Readmission risk at discharge\n"
        "Which diabetic patients should be followed up first after discharge? "
        f"Model {VERSION}. Educational prototype, not a medical device."
    )
    with gr.Tab("Patient at discharge"):
        with gr.Row():
            with gr.Column(scale=2):
                age = gr.Dropdown(AGE_BANDS, value=TYPICAL["age"], label="Age band")
                discharge = gr.Dropdown(list(DISCHARGE), value="Discharged to home",
                                        label="Discharge destination")
                source = gr.Dropdown(list(SOURCE), value="Emergency Room",
                                     label="Admission source")
                diagnosis = gr.Dropdown(list(DIAGNOSIS), value=list(DIAGNOSIS)[0],
                                        label="Primary diagnosis group")
                specialty = gr.Dropdown(SPECIALTIES, value="InternalMedicine",
                                        label="Admitting specialty")
                inpatient = gr.Slider(0, 10, value=0, step=1,
                                      label="Inpatient stays in the previous year")
                emergency = gr.Slider(0, 10, value=0, step=1,
                                      label="Emergency visits in the previous year")
                outpatient = gr.Slider(0, 10, value=0, step=1,
                                       label="Outpatient visits in the previous year")
                days = gr.Slider(1, 14, value=4, step=1, label="Days in hospital")
                meds = gr.Slider(1, 60, value=15, step=1,
                                 label="Medications given during the stay")
                labs = gr.Slider(1, 120, value=44, step=1, label="Lab procedures")
                diagnoses = gr.Slider(1, 16, value=8, step=1,
                                      label="Number of diagnoses recorded")
                a1c = gr.Radio(["None", "Norm", ">7", ">8"], value="None",
                               label="HbA1c result (None = not measured)")
                insulin = gr.Radio(["No", "Steady", "Up", "Down"], value="No",
                                   label="Insulin at discharge")
                go = gr.Button("Estimate the risk", variant="primary")
            with gr.Column(scale=3):
                answer = gr.Markdown()
                chart = gr.Plot(label="Risk and reasons")
        inputs = [age, discharge, source, days, inpatient, emergency, outpatient, meds,
                  labs, diagnoses, diagnosis, a1c, insulin, specialty]
        gr.Examples(
            examples=[
                ["[50-60)", "Discharged to home", "Physician Referral", 2, 0, 0, 0, 10,
                 30, 5, list(DIAGNOSIS)[5], "Norm", "No", "Orthopedics-Reconstructive"],
                ["[70-80)", "Discharged/transferred to home with home health service",
                 "Emergency Room", 5, 1, 0, 1, 18, 50, 9, list(DIAGNOSIS)[0], "None",
                 "Steady", "InternalMedicine"],
                ["[70-80)", "Discharged/transferred to SNF", "Emergency Room", 8, 4, 2,
                 1, 25, 60, 9, list(DIAGNOSIS)[0], ">8", "Up", "InternalMedicine"],
            ],
            inputs=inputs,
            label="Examples: low, medium and high risk",
        )
        gr.Markdown(f"<small>{FOOTNOTE}</small>")
        go.click(score_patient, inputs, [answer, chart])
        demo.load(score_patient, inputs, [answer, chart])

    with gr.Tab("Discharge worklist"):
        gr.Markdown(
            "Upload today's discharges as a CSV in the UCI format (the columns of "
            "`diabetic_data.csv`; `?` for missing values), or use 2,000 sample "
            "discharges from the test patients. Columns that are missing take a "
            "typical value; a `readmitted` column, if present, shows what the list "
            "would really have caught."
        )
        with gr.Row():
            upload = gr.File(file_types=[".csv"], label="Discharges (CSV, optional)")
            capacity = gr.Slider(5, 30, value=10, step=1,
                                 label="Follow-up capacity, % of discharges")
        run = gr.Button("Build the call list", variant="primary")
        summary = gr.Markdown()
        list_chart = gr.Plot(label="Readmissions reached")
        table = gr.Dataframe(label="Call list, highest risk first", wrap=True)
        gr.Markdown(f"<small>{FOOTNOTE}</small>")
        run.click(build_worklist, [upload, capacity], [summary, list_chart, table])

    with gr.Tab("Evidence"):
        gr.Markdown(evidence_markdown())
        for name, caption in FIGURE_FILES:
            if (FIGURES / name).exists():
                gr.Image(str(FIGURES / name), label=caption, show_label=True,
                         buttons=["download"])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
