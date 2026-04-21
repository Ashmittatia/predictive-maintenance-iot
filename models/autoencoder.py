"""
models/autoencoder.py
LSTM Autoencoder for unsupervised anomaly detection.
Trained on NORMAL samples only. Anomaly score = reconstruction error.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).parent / "lstm_autoencoder.keras"
THRESHOLD_PATH = Path(__file__).parent / "ae_threshold.pkl"

WINDOW_SIZE  = 30
LATENT_DIM   = 16
LSTM_UNITS   = 64


def build_autoencoder(n_features: int, window: int = WINDOW_SIZE) -> keras.Model:
    """
    LSTM Autoencoder architecture:
      Encoder: LSTM(64) → LSTM(32) → Dense(latent)
      Decoder: RepeatVector → LSTM(32) → LSTM(64) → TimeDistributed Dense
    """
    inputs = keras.Input(shape=(window, n_features), name="input")

    # Encoder
    x = layers.LSTM(LSTM_UNITS, return_sequences=True, name="enc_lstm1")(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(LSTM_UNITS // 2, return_sequences=False, name="enc_lstm2")(x)
    encoded = layers.Dense(LATENT_DIM, activation="relu", name="latent")(x)

    # Decoder
    x = layers.RepeatVector(window, name="repeat")(encoded)
    x = layers.LSTM(LSTM_UNITS // 2, return_sequences=True, name="dec_lstm1")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(LSTM_UNITS, return_sequences=True, name="dec_lstm2")(x)
    decoded = layers.TimeDistributed(layers.Dense(n_features), name="output")(x)

    model = keras.Model(inputs, decoded, name="LSTMAutoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return model


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    epochs: int = 30,
    batch_size: int = 64
) -> keras.Model:
    """Train the autoencoder on normal-only sequences."""
    n_features = X_train.shape[2]
    model = build_autoencoder(n_features, window=X_train.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    val_data = (X_val, X_val) if X_val is not None else None
    history = model.fit(
        X_train, X_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[✓] Autoencoder saved → {MODEL_PATH}")
    return model, history


def compute_reconstruction_errors(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Returns per-sample MSE reconstruction error."""
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.mean(np.square(X - X_pred), axis=2), axis=1)
    return errors


def fit_threshold(model: keras.Model, X_normal: np.ndarray, percentile: float = 95.0) -> float:
    """
    Set anomaly threshold as the Nth percentile of normal reconstruction errors.
    Anything above threshold is flagged as anomaly.
    """
    errors    = compute_reconstruction_errors(model, X_normal)
    threshold = float(np.percentile(errors, percentile))
    joblib.dump(threshold, THRESHOLD_PATH)
    print(f"[✓] Threshold ({percentile}th pct) = {threshold:.6f} → saved to {THRESHOLD_PATH}")
    return threshold


def load_autoencoder() -> tuple:
    """Load model and threshold. Returns (model, threshold)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_all.py first.")
    model     = keras.models.load_model(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else 0.05
    return model, threshold


def predict_anomaly(model: keras.Model, X: np.ndarray, threshold: float) -> dict:
    """
    Given a window of sensor readings X (shape: 1, window, features),
    returns anomaly score and flag.
    """
    errors = compute_reconstruction_errors(model, X)
    score  = float(errors[0])
    return {
        "anomaly_score":    round(score, 6),
        "is_anomaly":       score > threshold,
        "threshold":        round(threshold, 6),
        "confidence":       round(min(score / (threshold + 1e-9), 3.0), 3),
    }


if __name__ == "__main__":
    # Quick smoke test
    X_fake = np.random.randn(200, WINDOW_SIZE, 10).astype(np.float32)
    model  = build_autoencoder(n_features=10)
    model.fit(X_fake, X_fake, epochs=2, batch_size=32, verbose=0)
    errs   = compute_reconstruction_errors(model, X_fake[:5])
    print("Reconstruction errors (first 5):", errs)
