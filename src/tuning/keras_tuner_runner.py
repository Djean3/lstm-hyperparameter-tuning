# src/tuning/keras_tuner_runner.py
import os
import warnings
import keras_tuner_runner as kt
import tensorflow as tf
from models.base_lstm import build_model
from data.load_data import load_preprocessed_data
from models.evaluate_hparams import run_test
from utils.logging import get_log_filename, init_csv_logger, log_epoch, compute_metrics, generate_param_hash, summarize_log
import numpy as np
from datetime import datetime
import absl.logging
import shutil
import os
from utils.logging import STANDARD_LOG_HEADERS

def clear_keras_logs():
    log_dir = "outputs/keras_tuner/lstm_tuning"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"✅ Removed all previous logs from: {log_dir}")
    else:
        print(f"⚠️ Path not found: {log_dir}")

# Call before running tuner
clear_keras_logs()

# Suppress TF & absl logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
absl.logging.set_verbosity(absl.logging.ERROR)

def run_keras_tuner():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=True)
    input_shape = X_train.shape[1:]

    def wrapped_model_builder(hp):
        optimizer = hp.Choice("optimizer", ["adam", "adagrad", "nadam"])
        learning_rate = hp.Choice("learning_rate", [0.1, 0.01, 0.001])
        batch_size = hp.Choice("batch_size", [4, 8, 16])

        model, _ = build_model(
            layers=[200],
            time_steps=input_shape[0],
            num_features=input_shape[1],
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            dropout_rate=0.2,
            use_early_stopping=False
        )
        model.compile(optimizer=model.optimizer, loss="mean_squared_error", metrics=["mae"])
        return model

    tuner = kt.BayesianOptimization(
        wrapped_model_builder,
        objective='val_mae',
        max_trials=20,
        directory='outputs/keras_tuner',
        project_name='lstm_tuning'
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    best_hps = tuner.get_best_hyperparameters(1)[0]

    print("\n🎯 Best Hyperparameters Found:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")

    best_params = {
        "layers": [200],
        "time_steps": input_shape[0],
        "learning_rate": best_hps.get("learning_rate"),
        "optimizer": best_hps.get("optimizer"),
        "dropout_rate": 0.2,
        "batch_size": best_hps.get("batch_size"),
        "epochs": 20,
        "replicates": 10
    }

    log_filename = get_log_filename("kerasbo", "base_lstm")
    init_csv_logger(log_filename, STANDARD_LOG_HEADERS)
    param_set_id = generate_param_hash(best_params)

    for replicate in range(best_params["replicates"]):
        model, _ = build_model(
            layers=best_params["layers"],
            time_steps=best_params["time_steps"],
            num_features=input_shape[1],
            optimizer_name=best_params["optimizer"],
            learning_rate=best_params["learning_rate"],
            dropout_rate=best_params["dropout_rate"],
            use_early_stopping=False
        )

        model.compile(optimizer=model.optimizer, loss="mean_squared_error", metrics=["mae"])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=best_params["batch_size"],
            epochs=best_params["epochs"],
            verbose=0
        )

        for epoch, (train_loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            y_pred_train = model.predict(X_train, verbose=0).flatten()
            y_pred_val = model.predict(X_val, verbose=0).flatten()
            y_pred_test = model.predict(X_test, verbose=0).flatten()

            train_metrics = compute_metrics(y_train, y_pred_train)
            val_metrics = compute_metrics(y_val, y_pred_val)
            test_metrics = compute_metrics(y_test, y_pred_test)

            log_epoch(log_filename, {
                "trial_id": 0,
                "replicate": replicate,
                "epoch": epoch,
                "final_epoch_flag": int(epoch == best_params["epochs"] - 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": 0.0,
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "train_mape": train_metrics["mape"],
                "val_mape": val_metrics["mape"],
                "test_mape": test_metrics["mape"],
                "optimizer": best_params["optimizer"],
                "learning_rate": best_params["learning_rate"],
                "batch_size": best_params["batch_size"],
                "param_set_id": param_set_id,
                "early_stopped": False,
                "runtime_seconds": 0.0
            })

    run_test(best_params)

    # Print summary of results from the CSV log
    summary = summarize_log(log_filename)
    print("\n📊 Averaged Evaluation Metrics Across Replicates:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    run_keras_tuner()
