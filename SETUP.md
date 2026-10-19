# Setup

The project needs **Python 3.13** (required by the course; `ydata-profiling` does not support Python 3.14), a virtual environment in `.venv/` and the packages of `requirements.txt`. The steps differ between Windows and macOS, so each system has its own section. The data step and the final check are the same for everyone.

## Windows

1. Create the virtual environment and install the dependencies (PowerShell, from the project folder):

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled on this system"), run this once and try again:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

2. TensorFlow (used by the Keras model) runs on the **CPU only** on native Windows: GPU support was dropped after TensorFlow 2.11. Nothing to configure; the notebook prints a warning about it, which can be ignored.

## macOS

Checked on 2026-10-19: every package of `requirements.txt` has a ready-made wheel for Python 3.13 on **Apple Silicon** (M1–M4). On **Intel Macs**, TensorFlow 2.21 has no wheel (see step 4).

1. Install Python 3.13, either with the installer from [python.org](https://www.python.org/downloads/) or with Homebrew:

   ```bash
   brew install python@3.13
   ```

2. Install **OpenMP**, which LightGBM and XGBoost need on macOS (without it, `import lightgbm` or `import xgboost` fails with a `libomp.dylib` error):

   ```bash
   brew install libomp
   ```

3. Create the virtual environment and install the dependencies (Terminal, from the project folder):

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Intel Macs only.** `pip install -r requirements.txt` stops at `tensorflow`. Install everything else, and leave the Keras model (step 5.10 of `DC_Models/02_model_exploration.ipynb`) to a teammate on Windows or Apple Silicon:

   ```bash
   grep -v '^tensorflow' requirements.txt > /tmp/requirements-no-tf.txt
   pip install -r /tmp/requirements-no-tf.txt
   ```

## Data (all systems)

Copy `diabetic_data.csv` and `IDS_mapping.csv` into the `data/` folder. These files must remain available locally only (they are gitignored).

## Check (all systems)

1. Open the project folder in VS Code and select the kernel: **Select Kernel → Python Environments → `.venv` (Python 3.13)**.
2. Run the first cell of `DC_Models/02_model_exploration.ipynb`: it must print `True` for Python 3.13, `.venv` and the data, and `OK` for `xgboost`, `lightgbm`, `catboost`, `interpret`, `tensorflow` and `shap`. A `MISSING` means the corresponding `pip install` step did not complete.

If the environment ever breaks, delete the `.venv` folder and repeat the steps: it holds none of your work.
