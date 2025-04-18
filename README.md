lstm-hyperparameter-tuning/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── tuning_results.ipynb
│
├── outputs/
│   ├── models/                 # Saved model checkpoints
│   ├── graphs/                 # Comparison plots, etc.
│   └── mlruns/                 # MLflow tracking dir
│
├── src/
│   ├── data/                   
│   │   ├── __init__.py
│   │   └── preprocessing.py    # Data cleaning, train/test/val split
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_lstm.py        # The core LSTM model definition
│   │   └── train.py            # Generic training function
│   │
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── keras_tuner_runner.py
│   │   ├── grid_search_runner.py
│   │   ├── hyperopt_runner.py
│   │   ├── optuna_runner.py
│   │   └── results_logger.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py          # RMSE, MAE, R², etc.
│   │   ├── logging.py
│   │   └── callbacks.py
│   │
│   └── config/
│       └── settings.py         # Hyperparams, experiment config, etc.
│
├── pyproject.toml
├── README.md
└── run.py                      # Optional central runner entry point
