import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(layers, time_steps, num_features, optimizer_name="adam", learning_rate=0.001, dropout_rate=0.2, use_early_stopping=True):
    model = Sequential()

    for i in range(len(layers)):
        return_sequences = i < len(layers) - 1  # Only return sequences if it's not the last layer
        if i == 0:
            model.add(LSTM(layers[i], return_sequences=return_sequences, input_shape=(time_steps, num_features)))
        else:
            model.add(LSTM(layers[i], return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    # Set optimizer
    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer_name.lower() == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate)
    elif optimizer_name.lower() == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Add early stopping callback (optional to use during training)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    ) if use_early_stopping else None

    return model, early_stopping

