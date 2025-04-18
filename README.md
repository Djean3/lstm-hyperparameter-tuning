## 📁 Project Directory Overview

### 📂 data/
- **raw/** — Original input CSV file.
- **processed/** — Contains:
  - `processed_data.npz`: Saved train/val/test splits.
  - `scaler.save`: Fitted `MinMaxScaler` for inverse transformations.

### 📂 notebooks/
- `exploration.ipynb` — Initial dataset exploration.
- `LSTM-SDM_Code_Implementation.ipynb` — Draft implementation based on the paper.
- `tuning_results.ipynb` — Placeholder to visualize or summarize tuning outcomes.

### 📂 outputs/
- **keras_tuner/lstm_tuning/** — Saved trials and metadata from Keras Tuner.
- **models/** — Directory for storing trained model artifacts.
- **graphs/** — Plots and evaluation visuals (future use).
- **mlruns/** — MLflow logs (to be added).

### 📂 src/
#### 📁 config/
- (Optional) For configuration files like `settings.py`, etc.

#### 📁 data/
- `preprocessing.py` — Converts raw CSV into model-ready format, saves outputs.
- `load_data.py` — Loads processed data and scaler into memory.

#### 📁 models/
- `base_lstm.py` — Core LSTM model definition and compile logic.
- `evaluate_hparams.py` — Evaluates models using a fixed train/test split and multiple replicates.
- `train_test_run.py` — Standalone training/testing script (useful for debugging).
- `train.py` — (Planned) A generic training entry point.

#### 📁 tuning/
- `keras_tuner_runner.py` — Runs Keras Tuner using Hyperband.
- `optuna_runner.py` — Placeholder for Optuna integration.
- `hyperopt_runner.py` — Placeholder for Hyperopt integration.
- `grid_search_runner.py` — Placeholder for manual/grid-based tuning.
- `results_logger.py` — Planned utility for saving and comparing results.

#### 📁 utils/
- `callbacks.py` — Early stopping and other Keras callback utilities.
- `metrics.py` — Custom metrics (e.g., MAPE).
- `logging.py` — MLflow logging integration (in progress).

### 📂 tests/
- (Placeholder for unit tests)

### 📄 Root Files
- `run.py` — Optional CLI runner script.
- `pyproject.toml` — Poetry environment and dependency manager.
- `README.md` — This file!
