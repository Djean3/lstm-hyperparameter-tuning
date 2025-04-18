import numpy as np
import joblib

def load_preprocessed_data(include_scaler=False):
    data = np.load("data/processed/processed_data.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if include_scaler:
        scaler = joblib.load("data/processed/scaler.save")
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler

    return X_train, y_train, X_val, y_val, X_test, y_test