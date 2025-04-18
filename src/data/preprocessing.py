import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import os
import joblib

RAW_DATA_PATH = "data/raw/input_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
SCALER_SAVE_PATH = "data/processed/minmax_scaler.pkl"


def haar_denoise(series, wavelet='haar', level=2):
    coeffs = pywt.wavedec(series, wavelet, mode="per")
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")


def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop 'Open' as per paper due to high correlation with Close
    df.drop(columns=['Open'], inplace=True, errors='ignore')

    # Haar wavelet denoising on Close
    df['Close'] = haar_denoise(df['Close'])

    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)  # Drop remaining NaNs if any

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    # Save processed data and scaler
    os.makedirs("data/processed", exist_ok=True)
    scaled_df.to_csv(PROCESSED_DATA_PATH, index=False)
    joblib.dump(scaler, SCALER_SAVE_PATH)

    print("✅ Preprocessing complete. Saved processed_data.csv and scaler.")


if __name__ == "__main__":
    preprocess()
