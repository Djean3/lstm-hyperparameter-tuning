# logging.py
import csv
import os
import hashlib
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


def get_log_filename(tuner: str, model: str, log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(log_dir, f"{tuner}_{model}_{date_str}.csv")


def generate_param_hash(params: dict) -> str:
    sorted_items = sorted(params.items())
    param_str = "_".join(f"{k}={v}" for k, v in sorted_items)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]


STANDARD_LOG_HEADERS = [
    "trial_id", "replicate", "epoch", "final_epoch_flag",
    "train_loss", "val_loss", "test_loss",
    "train_mae", "val_mae", "test_mae",
    "train_rmse", "val_rmse", "test_rmse",
    "train_r2", "val_r2", "test_r2",
    "train_mape", "val_mape", "test_mape",
    "optimizer", "learning_rate", "batch_size", "dropout_rate", "layers",
    "param_set_id", "early_stopped", "runtime_seconds" #"hyeperparameters"
]


def init_csv_logger(filename: str, headers: list[str] = None):
    headers = headers or STANDARD_LOG_HEADERS
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()


def log_epoch(filename: str, log_data: dict):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        writer.writerow(log_data)


def summarize_log(filename: str, summary_filename: str = None):
    import pandas as pd

    df = pd.read_csv(filename)

    # Only use final epoch from each replicate
    if "replicate" in df.columns and "val_loss" in df.columns:
        df = df.sort_values("val_loss", ascending=True).drop_duplicates(subset=["replicate"], keep="first")
        print(f"✅ Selected best val_loss row per replicate: {len(df)} rows")

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    # Average numeric metrics
    summary = df[numeric_cols].mean().to_dict()

    # Use mode for non-numeric values (e.g., optimizer, layers, etc.)
    for col in non_numeric_cols:
        try:
            summary[col] = df[col].mode().iloc[0]
        except:
            summary[col] = "N/A"

    # Add counts and filename
    summary["total_epochs"] = df['epoch'].nunique() if 'epoch' in df.columns else 0
    summary["total_trials"] = df['trial_id'].nunique() if 'trial_id' in df.columns else 1
    summary["filename"] = filename

    # Write to CSV
    if summary_filename is None:
        summary_filename = filename.replace(".csv", "_summary.csv")

    with open(summary_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    return summary


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred))
        }