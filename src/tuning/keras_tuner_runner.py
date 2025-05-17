# src/tuning/keras_tuner_runner.py
import os
import warnings
import keras_tuner as kt
import tensorflow as tf
from models.base_lstm import build_model
from data.load_data import load_preprocessed_data
from models.evaluate_hparams import run_test
from utils.hps_logger import init_hps_log, log_hps_epoch
from utils.model_runs_logger import init_model_log, log_model_epoch
from utils.summary_logger import summarize_model_runs
from utils.model_runs_logger import MODEL_RUN_HEADERS
from utils.hps_logger import HPS_LOG_HEADERS
from utils.hps_logger import HPSLoggerCallback
from utils.common import generate_param_hash
import numpy as np
from datetime import datetime
import absl.logging
import shutil
import time
from utils.compute_metrics import compute_metrics

from utils.compute_metrics import compute_metrics

def clear_keras_logs():
    log_dir = "outputs/keras_tuner/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"✅ Removed all previous logs from: {log_dir}")
    else:
        print(f"⚠️ Path not found: {log_dir}")

def clear_log_files():
    log_dir = "logs/"
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}: {e}")
        print(f"✅ Cleared all files from: {log_dir}")
    else:
        print(f"⚠️ Path not found: {log_dir}")

# Clear both
clear_keras_logs()
clear_log_files()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
absl.logging.set_verbosity(absl.logging.ERROR)

def run_keras_tuner():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=True)
    input_shape = X_train.shape[1:]

    def wrapped_model_builder(hp):
        optimizer = hp.Choice("optimizer", ["adam", "adagrad", "nadam"])
        learning_rate = hp.Choice("learning_rate", [0.01, 0.001, 0.0001])
        batch_size = hp.Choice("batch_size", [4, 8, 16, 32, 64, 128, 256 ])
        dropout_rate = hp.Choice("dropout_rate", [0.1, 0.2, 0.3])
        num_layers = hp.Int("num_layers", 1, 5)
        layer_size = hp.Choice("layer_size", list(range(32, 257, 16)))
        use_early_stopping = hp.Boolean("use_early_stopping", False)

        layers = [layer_size] * num_layers

        model, early_stopping_cb = build_model(
            layers=layers,
            time_steps=input_shape[0],
            num_features=input_shape[1],
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            use_early_stopping=use_early_stopping
        )

        return model

    tuner = kt.BayesianOptimization(
        wrapped_model_builder,
        objective='val_mae',
        max_trials=50,   ############### default value is 20 #####################
        directory='outputs/keras_tuner',
        project_name='lstm_tuning'
    )

    trial_id = f"trial_{int(time.time())}"
    log_file = f"logs/{trial_id}.csv"
    init_hps_log(log_file)

    hpo_callback = HPSLoggerCallback(
        filename=log_file,
        trial_id=trial_id,
        replicate=0,
        param_dict={},  # optional: can fill this from `hp` if you want
        X_test=X_test,
        y_test=y_test
    )

    

    tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,  ################DEFAULT VALUE IS 10
    callbacks=[hpo_callback]
)

    best_hps = tuner.get_best_hyperparameters(1)[0]

    best_params = {
        "layers": [best_hps.get("layer_size")] * best_hps.get("num_layers"),
        "time_steps": input_shape[0],
        "learning_rate": best_hps.get("learning_rate"),
        "optimizer": best_hps.get("optimizer"),
        "dropout_rate": best_hps.get("dropout_rate"),
        "batch_size": best_hps.get("batch_size"),
        "use_early_stopping": best_hps.get("use_early_stopping"),
        "epochs": 20,
        "replicates": 10
    }

    print("Best parameters:", best_params)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/kerasbo_base_lstm_hpo_{timestamp}.csv"
    model_log_filename = f"logs/kerasbo_base_lstm_model_runs_{timestamp}.csv"
    model_summary_filename = f"logs/kerasbo_base_lstm_model_runs_{timestamp}_summary.csv"

    init_hps_log(log_filename)
    init_model_log(model_log_filename)

    param_set_id = generate_param_hash(best_params)

    for replicate in range(best_params["replicates"]):
        model, early_stopping_cb = build_model(
            layers=best_params["layers"],
            time_steps=best_params["time_steps"],
            num_features=input_shape[1],
            optimizer_name=best_params["optimizer"],
            learning_rate=best_params["learning_rate"],
            dropout_rate=best_params["dropout_rate"],
            use_early_stopping=best_params["use_early_stopping"]
        )

        model.compile(optimizer=model.optimizer, loss="mean_squared_error", metrics=["mae"])
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0
        )
        start_time = time.time()

        callbacks = [lr_callback]
        if early_stopping_cb:
            callbacks.append(early_stopping_cb)


        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=best_params["batch_size"],
            epochs=best_params["epochs"],
            callbacks=callbacks,
            verbose=0
        )

        runtime = time.time() - start_time
        epochs_run = len(history.history["loss"])
        early_stopped = epochs_run < best_params["epochs"]

        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_val = model.predict(X_val, verbose=0).flatten()
        y_pred_test = model.predict(X_test, verbose=0).flatten()

        # Fix is here:
        from tensorflow.keras.losses import MeanSquaredError
        mse = tf.keras.losses.MeanSquaredError()
        test_loss = mse(y_test, y_pred_test).numpy()


        

        for epoch, (train_loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            train_metrics = compute_metrics(y_train, y_pred_train)
            val_metrics = compute_metrics(y_val, y_pred_val)
            test_metrics = compute_metrics(y_test, y_pred_test)

            dynamic_learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))

            row = {
                "trial_id": replicate,
                "replicate": replicate,
                "epoch": epoch,
                "final_epoch_flag": int(epoch == epochs_run - 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
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
                "dynamic_learning_rate": dynamic_learning_rate,
                "batch_size": best_params["batch_size"],
                "dropout_rate": best_params["dropout_rate"],
                "layers": best_params["layers"],
                "param_set_id": param_set_id,
                "early_stopped": early_stopped,
                "runtime_seconds": runtime,
                "hyperparameters": f"optimizer={best_params['optimizer']}, learning_rate={best_params['learning_rate']}, batch_size={best_params['batch_size']}, dropout_rate={best_params['dropout_rate']}, layers={best_params['layers']}",

            }

            log_hps_epoch(log_filename, row)
            log_model_epoch(model_log_filename, row)

    run_test(best_params)
    summary = summarize_model_runs(model_log_filename, model_summary_filename)
    print("\n📊 Averaged Evaluation Metrics Across Replicates:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    run_keras_tuner()