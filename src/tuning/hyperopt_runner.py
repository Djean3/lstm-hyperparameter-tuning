# src/tuning/hyperopt_runner.py
import os
import warnings
import tensorflow as tf
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from models.base_lstm import build_model
from data.load_data import load_preprocessed_data
from models.evaluate_hparams import run_test
import numpy as np
import absl.logging

# Suppress TF & absl logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
absl.logging.set_verbosity(absl.logging.ERROR)


def run_hyperopt_tuner():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=True)
    input_shape = X_train.shape[1:]

    def objective(params):
        model, _ = build_model(
            layers=[200],
            time_steps=input_shape[0],
            num_features=input_shape[1],
            optimizer_name=params['optimizer'],
            learning_rate=params['learning_rate'],
            dropout_rate=0.2,
            use_early_stopping=False
        )

        model.compile(optimizer=model.optimizer, loss='mean_squared_error', metrics=['mae'])
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=params['batch_size'],
            verbose=0
        )

        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        return {'loss': val_mae, 'status': STATUS_OK}

    search_space = {
        'optimizer': hp.choice('optimizer', ['adam', 'adagrad', 'nadam']),
        'learning_rate': hp.choice('learning_rate', [0.1, 0.01, 0.001]),
        'batch_size': hp.choice('batch_size', [4, 8, 16])
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    # Decode best choice indexes to actual values
    optimizer_list = ['adam', 'adagrad', 'nadam']
    lr_list = [0.1, 0.01, 0.001]
    batch_list = [4, 8, 16]

    best_params = {
        "layers": [200],
        "time_steps": input_shape[0],
        "learning_rate": lr_list[best['learning_rate']],
        "optimizer": optimizer_list[best['optimizer']],
        "dropout_rate": 0.2,
        "batch_size": batch_list[best['batch_size']],
        "epochs": 20,
        "replicates": 10
    }

    print("\n🌟 Best Hyperparameters Found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    run_test(best_params)


if __name__ == "__main__":
    run_hyperopt_tuner()
