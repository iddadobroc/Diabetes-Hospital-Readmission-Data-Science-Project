# Diabetes Hospital Readmission — Data Science Project

Project structure:

- `notebooks/`: Jupyter notebooks with the analysis.
- `data/`: local datasets, excluded from Git.
- `docs/`: reference documents for the project (project brief, project plan).
- `reports/`: generated HTML profiling reports, tracked in the repository.
- `requirements.txt`: Python dependencies.

To set up the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The files `data/diabetic_data.csv` and `data/IDS_mapping.csv` must remain available locally only.