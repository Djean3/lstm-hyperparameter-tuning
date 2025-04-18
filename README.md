📁 data/ ├── raw/ # Original raw CSV input data ├── processed/ # Preprocessed split data and scaler for reuse │ ├── processed_data.npz # Contains train/val/test arrays │ └── scaler.save # Saved MinMaxScaler object

📁 notebooks/ ├── exploration.ipynb # Initial exploration of the dataset ├── LSTM-SDM_Code_Implementation.ipynb # Draft implementation from the paper ├── tuning_results.ipynb # Placeholder for results summary

📁 outputs/ ├── keras_tuner/lstm_tuning/ # Keras Tuner Hyperband trial logs ├── models/ # Trained model artifacts (future use) ├── graphs/ # Plots and metrics (planned) ├── mlruns/ # (MLflow logs coming soon)

📁 src/ ├── config/ # (Optional future config files) │ ├── data/ │ ├── load_data.py # Loads and returns preprocessed data splits │ └── preprocessing.py # Prepares the raw CSV for modeling and saves outputs │ ├── lstm_hyperparameter_tuning/ │ └── init.py # Namespace

├── models/ │ ├── base_lstm.py # Core LSTM model builder │ ├── evaluate_hparams.py # Runs model training/evaluation with replicates │ ├── train_test_run.py # Manual run script (pre-Keras tuner) │ └── train.py # (Future training script)

├── tuning/ │ ├── keras_tuner_runner.py # Runs Keras Tuner (Hyperband) search + evaluation │ ├── optuna_runner.py # Setup for Optuna (pending) │ ├── hyperopt_runner.py # Setup for Hyperopt (pending) │ ├── grid_search_runner.py # Setup for basic grid search (pending) │ └── results_logger.py # (Planned) Results tracking and saving

├── utils/ │ ├── callbacks.py # Early stopping and other callback logic │ ├── metrics.py # Custom metrics like MAPE │ └── logging.py # Placeholder for MLflow logging (in progress)

📁 tests/ └── (Placeholder for unit tests)

📄 run.py # Optional CLI runner 📄 pyproject.toml # Poetry dependency management 📄 README.md # Project documentation