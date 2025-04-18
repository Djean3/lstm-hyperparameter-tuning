# File renamed to evaluate_hparams.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.load_data import load_preprocessed_data
from models.base_lstm import build_model
from utils.metrics import mean_absolute_percentage_error

def run_test(hparams: dict, use_scaler=True, verbose=1):
    # Unpack hyperparameters
    layers = hparams.get("layers", [50])
    time_steps = hparams.get("time_steps", 5)
    learning_rate = hparams.get("learning_rate", 0.001)
    optimizer = hparams.get("optimizer", "adam")
    dropout_rate = hparams.get("dropout_rate", 0.2)
    batch_size = hparams.get("batch_size", 32)
    epochs = hparams.get("epochs", 50)

    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=use_scaler)

    num_features = X_train.shape[2]

    # Build model
    model, early_stopping = build_model(
        layers=layers,
        time_steps=time_steps,
        num_features=num_features,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        use_early_stopping=True
    )

    callbacks = [early_stopping] if early_stopping else []

    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )

    # Evaluate model
    predictions = model.predict(X_test).ravel()
    y_true = y_test.ravel()

    if use_scaler:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    print("\n📊 Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "model": model
    }


if __name__ == "__main__":
    # Example run
    hparams = {
        "layers": [200],
        "time_steps": 5,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "dropout_rate": 0.2,
        "batch_size": 32,
        "epochs": 10
    }
    run_test(hparams)
