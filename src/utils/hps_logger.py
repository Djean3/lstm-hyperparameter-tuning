import os
import csv
import tensorflow as tf

HPS_LOG_HEADERS = [
    "trial_id", "replicate", "epoch", "final_epoch_flag",
    "train_loss", "val_loss", "test_loss",
    "train_mae", "val_mae", "test_mae",
    "train_rmse", "val_rmse", "test_rmse",
    "train_r2", "val_r2", "test_r2",
    "train_mape", "val_mape", "test_mape",
    "optimizer", "learning_rate", "dynamic_learning_rate", "batch_size", "dropout_rate", "layers",
    "param_set_id", "early_stopped", "runtime_seconds", "hyeperparameters"
]

def init_hps_log(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HPS_LOG_HEADERS)
            writer.writeheader()

def log_hps_epoch(filename, log_data):
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        writer.writerow(log_data)

from tensorflow.keras.callbacks import Callback
import time
import numpy as np
from .hps_logger import log_hps_epoch  # ✅ You're already in utils/, so use relative import


class HPSLoggerCallback(Callback):
    def __init__(self, filename, trial_id, replicate, param_dict, X_test, y_test):
        super().__init__()
        self.filename = filename
        self.trial_id = trial_id
        self.replicate = replicate
        self.param_dict = param_dict
        self.X_test = X_test
        self.y_test = y_test
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        final_epoch = int(epoch == self.params['epochs'] - 1)
        runtime = time.time() - self.start_time

        results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        test_loss = results[0]
        test_mae = results[1]
        test_rmse = results[2] if len(results) > 2 else None
        test_mape = results[3] if len(results) > 3 else None
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_true = self.y_test

        # Manually calculate other metrics
        from utils.compute_metrics import compute_metrics  # adjust the path if needed
        metrics = compute_metrics(y_true, y_pred)
        rmse, r2, mape = (metrics[k] for k in ["rmse", "r2", "mape"])

        log_data = {
            "trial_id": self.trial_id,
            "replicate": self.replicate,
            "epoch": epoch,
            "final_epoch_flag": final_epoch,
            "train_loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "test_loss": test_loss,
            "train_mae": logs.get("mae"),
            "val_mae": logs.get("val_mae"),
            "test_mae": test_mae,
            "train_rmse": None,
            "val_rmse": None,
            "test_rmse": test_rmse,
            "train_r2": None,
            "val_r2": None,
            "test_r2": r2,
            "train_mape": None,
            "val_mape": None,
            "test_mape": test_mape,
            "optimizer": self.model.optimizer.__class__.__name__.lower(),
            "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
            "dynamic_learning_rate": None,
            "batch_size": self.param_dict.get("batch_size", None),
            "dropout_rate": self.param_dict.get("dropout_rate", None),
            "layers": self.param_dict.get("layers", None),
            "param_set_id": self.param_dict.get("param_set_id", "manual"),
            "early_stopped": False,
            "runtime_seconds": runtime,
            "hyeperparameters": str(self.param_dict)
}

        log_hps_epoch(self.filename, log_data)