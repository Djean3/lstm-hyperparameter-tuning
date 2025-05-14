# 2. utils/model_runs_logger.py - logs final evaluation epochs from best HPO params
import os
import csv

MODEL_RUN_HEADERS = [
    "trial_id", "replicate", "epoch", "final_epoch_flag",
    "train_loss", "val_loss", "test_loss",
    "train_mae", "val_mae", "test_mae",
    "train_rmse", "val_rmse", "test_rmse",
    "train_r2", "val_r2", "test_r2",
    "train_mape", "val_mape", "test_mape",
    "optimizer", "learning_rate", "dynamic_learning_rate", "batch_size", "dropout_rate", "layers",
    "param_set_id", "early_stopped", "runtime_seconds", "hyeperparameters"
]

def init_model_log(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=MODEL_RUN_HEADERS)
            writer.writeheader()

def log_model_epoch(filename, log_data):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        writer.writerow(log_data)