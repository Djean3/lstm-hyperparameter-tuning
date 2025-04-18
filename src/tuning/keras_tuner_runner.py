# src/tuning/keras_tuner_runner.py
import os
import warnings
import keras_tuner as kt
import tensorflow as tf
from models.base_lstm import build_model
from data.load_data import load_preprocessed_data
from models.evaluate_hparams import run_test

# Suppress TF & absl logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def run_keras_tuner():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=True)
    input_shape = X_train.shape[1:]

    def wrapped_model_builder(hp):
        optimizer = hp.Choice("optimizer", ["adam", "adagrad", "nadam"])
        learning_rate = hp.Choice("learning_rate", [0.1, 0.01, 0.001])
        # 🔧 Now declared here too
        hp.Choice("batch_size", [4, 8, 16])  

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

    tuner = kt.Hyperband(
        wrapped_model_builder,
        objective='val_mae',
        max_epochs=10,
        directory='outputs/keras_tuner',
        project_name='lstm_tuning'
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    best_hps = tuner.get_best_hyperparameters(1)[0]

    print("\n🎯 Best Hyperparameters Found:")
    for param in best_hps.values:
        print(f"{param}: {best_hps.get(param)}")

    # If batch_size wasn’t chosen, default it manually
    best_params = {
        "layers": [200],
        "time_steps": input_shape[0],
        "learning_rate": best_hps.get("learning_rate"),
        "optimizer": best_hps.get("optimizer"),
        "dropout_rate": 0.2,
        "batch_size": best_hps.values.get("batch_size", 32),
        "epochs": 20
    }

    run_test(best_params)

if __name__ == "__main__":
    run_keras_tuner()
