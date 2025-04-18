# =================== #
# 📦 Environment Setup
# =================== #
import os
import warnings

# Suppress TensorFlow logs and OneDNN factory warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # Disable INFO & WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'             # Disable oneDNN-related precision logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'      # Prevent TF from allocating all GPU memory

warnings.filterwarnings('ignore')                     # Suppress standard Python warnings

# Optional: suppress absl logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# =================== #
# 📦 Project Imports
# =================== #
from data.load_data import load_preprocessed_data
from models.base_lstm import build_model
import tensorflow as tf



# Load data and scaler
X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_preprocessed_data(include_scaler=True)

# Settings
layers = [200]
time_steps = X_train.shape[1]
num_features = X_train.shape[2]
learning_rate = 0.001
optimizer = "adam"
dropout_rate = 0.2
batch_size = 16
epochs = 50

# Build model
model, early_stopping = build_model(
    layers, time_steps, num_features,
    optimizer_name=optimizer,
    learning_rate=learning_rate,
    dropout_rate=dropout_rate,
    use_early_stopping=True
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping] if early_stopping else None,
    verbose=1
)

# Evaluate on test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")
