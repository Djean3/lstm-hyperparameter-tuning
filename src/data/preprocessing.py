import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


def haar_denoise(signal):
    import pywt
    coeffs = pywt.wavedec(signal, 'haar', level=1)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, 'haar')[:len(signal)]


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def preprocess():
    df = pd.read_csv("data/raw/input_data.csv")

    # ✅ Set date column as index and parse dates like the original notebook
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df.set_index("Date", inplace=True)

    # ✅ Haar denoising on Close price
    df['Close'] = haar_denoise(df['Close'])

    # ✅ Fill any remaining missing values
    df.fillna(method='ffill', inplace=True)

    # ✅ Scaling only the Close column (others weren't used in their notebook)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    # ✅ Create sequences for supervised learning
    window_size = 60
    X, y = create_sequences(scaled_data, window_size)

    # ✅ Split data (80% train, 10% val, 10% test)
    total_samples = len(X)
    train_end = int(total_samples * 0.8)
    val_end = int(total_samples * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # ✅ Save processed data and scaler
    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/processed_data.npz", 
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    joblib.dump(scaler, "data/processed/scaler.save")

    print("✅ Preprocessing complete. Processed data saved.")


if __name__ == "__main__":
    preprocess()
