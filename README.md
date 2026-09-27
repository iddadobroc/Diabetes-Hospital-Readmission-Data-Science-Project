# Analisi dei ricoveri per diabete

Struttura del progetto:

- `notebooks/`: notebook Jupyter con l'analisi.
- `data/`: dataset locali, esclusi da Git.
- `reports/`: report HTML generati localmente, esclusi da Git.
- `requirements.txt`: dipendenze Python.

Per preparare l'ambiente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

I file `data/diabetic_data.csv` e `data/IDS_mapping.csv` devono restare disponibili solo in locale.
