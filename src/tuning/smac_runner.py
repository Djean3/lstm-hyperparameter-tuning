import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.load_data import load_preprocessed_data
from models.base_lstm import build_model
from utils.metrics import mean_absolute_percentage_error
from statistics import mean
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter



def run_test(hparams: dict, use_scaler=True, verbose=1):
    # Unpack hyperparameters
    layers = hparams.get("layers", [50])
    time_steps = hparams.get("time_steps", 5)
    learning_rate = hparams.get("learning_rate", 0.001)
    optimizer = hparams.get("optimizer", "adam")
    dropout_rate = hparams.get("dropout_rate", 0.2)
    batch_size = hparams.get("batch_size", 32)
    epochs = hparams.get("epochs", 50)
    replicates = hparams.get("replicates", 1)

    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=use_scaler)
    num_features = X_train.shape[2]

    # Run multiple replicates if requested
    all_metrics = []
    for i in range(replicates):
        if replicates > 1:
            print(f"\n🔁 Running replicate {i+1}/{replicates}...")

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

        all_metrics.append({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "model": model
        })

    if replicates == 1:
        print("\n📊 Evaluation Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        return all_metrics[0]
    else:
        avg_rmse = mean(m["rmse"] for m in all_metrics)
        avg_mae = mean(m["mae"] for m in all_metrics)
        avg_r2 = mean(m["r2"] for m in all_metrics)
        avg_mape = mean(m["mape"] for m in all_metrics)

        print("\n📊 Averaged Evaluation Metrics Across Replicates:")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"MAE: {avg_mae:.4f}")
        print(f"R²: {avg_r2:.4f}")
        print(f"MAPE: {avg_mape:.2f}%")

        return {
            "rmse": avg_rmse,
            "mae": avg_mae,
            "r2": avg_r2,
            "mape": avg_mape,
            "model": all_metrics[0]["model"]  # return the last model trained
        }


if __name__ == "__main__":
    from smac.scenario import Scenario
    from smac.multi_objective.parego import ParEGO
    from ConfigSpace import ConfigurationSpace, Categorical, Float, Int

    # Define configuration space
    cs = ConfigurationSpace({
        "optimizer": Categorical("optimizer", ["adam", "adagrad", "nadam"]),
        "learning_rate": Float("learning_rate", (0.0001, 0.1), log=True),
        "batch_size": Int("batch_size", (4, 16)),
    })

    def objective_function(cfg):
        hparams = {
            "layers": [200],
            "time_steps": 5,
            "learning_rate": cfg["learning_rate"],
            "optimizer": cfg["optimizer"],
            "dropout_rate": 0.2,
            "batch_size": cfg["batch_size"],
            "epochs": 10,
            "replicates": 3
        }
        result = run_test(hparams, verbose=0)
        return result["mae"]

    scenario = Scenario(cs, deterministic=True, n_trials=10)
    smac = ParEGO(scenario, objective_function)
    incumbent = smac.optimize()

    best_params = {
        "layers": [200],
        "time_steps": 5,
        "learning_rate": incumbent["learning_rate"],
        "optimizer": incumbent["optimizer"],
        "dropout_rate": 0.2,
        "batch_size": incumbent["batch_size"],
        "epochs": 20,
        "replicates": 10
    }

    print("\n🎯 Best Hyperparameters Found by SMAC:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    run_test(best_params)
