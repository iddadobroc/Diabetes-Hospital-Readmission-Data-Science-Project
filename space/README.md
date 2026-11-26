---
title: Diabetes Readmission Risk
emoji: 🏥
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Which diabetic patients to follow up first after discharge
---

# Readmission risk at discharge

Decision-support prototype of the NOVA IMS Enterprise Data Science Bootcamp 2026. It
estimates each diabetic patient's risk of an inpatient readmission within 30 days and
ranks the discharges, so that limited follow-up goes first to the patients most likely
to come back.

- **Patient at discharge**: the risk of one patient, its follow-up tier and the reasons.
- **Discharge worklist**: a list of discharges ranked by risk and cut at the capacity.
- **Evidence**: how the model was measured on 19,802 test discharges it never saw.

The app loads the model version written in `model_version.txt` from the model
repository written in `model_repo.txt`; both files, and this folder, are uploaded by the
GitHub pipeline of the project when a version tag is pushed. Educational prototype, not
a medical device. Data: UCI Diabetes 130-US hospitals 1999-2008 (CC BY 4.0).
