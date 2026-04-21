"""
utils/preprocessing.py
Feature engineering for time-series sensor data:
  - Rolling statistics (mean, std, min, max)
  - FFT-based frequency features
  - RMS (Root Mean Square) energy
  - Z-score normalization
  - Sliding window sequences for LSTM
"""

import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

SENSOR_COLS = ["vibration", "temperature", "current", "pressure", "rpm", "humidity", "power_w"]
WINDOW_SIZE = 30   # timesteps for LSTM sequences
STEP_SIZE   = 5    # stride for sliding window

SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"


# ──────────────────────────────────────────────
# Rolling statistical features
# ──────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, cols: list = SENSOR_COLS, window: int = 20) -> pd.DataFrame:
    """Add rolling mean, std, min, max for each sensor column."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_roll_mean"] = df.groupby("device_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = df.groupby("device_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"{col}_roll_min"] = df.groupby("device_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f"{col}_roll_max"] = df.groupby("device_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    return df


def add_rms(df: pd.DataFrame, cols: list = ["vibration", "current"]) -> pd.DataFrame:
    """Root Mean Square — captures energy content."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_rms"] = df.groupby("device_id")[col].transform(
                lambda x: x.rolling(20, min_periods=1).apply(lambda w: np.sqrt(np.mean(w**2)))
            )
    return df


def add_rate_of_change(df: pd.DataFrame, cols: list = SENSOR_COLS) -> pd.DataFrame:
    """First-order difference per device (rate of change)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_roc"] = df.groupby("device_id")[col].diff().fillna(0)
    return df


def compute_fft_energy(signal: np.ndarray, n_components: int = 5) -> np.ndarray:
    """Return top-n FFT magnitude components for a 1D signal window.
    Always returns exactly n_components values (zero-padded if signal is short)."""
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:len(fft_vals) // 2]   # keep positive freqs
    # Zero-pad if fewer frequency bins than requested
    if len(fft_vals) < n_components:
        fft_vals = np.pad(fft_vals, (0, n_components - len(fft_vals)))
    top_idx = np.argsort(fft_vals)[::-1][:n_components]
    return fft_vals[top_idx]


def add_fft_features(df: pd.DataFrame, col: str = "vibration", window: int = 30) -> pd.DataFrame:
    """
    Add rolling FFT energy features for a given column.
    Groups by device_id, applies a rolling window FFT.
    """
    df = df.copy()
    feature_rows = []
    for device_id, group in df.groupby("device_id"):
        values = group[col].values
        fft_feats = np.zeros((len(values), 5))
        for i in range(len(values)):
            start = max(0, i - window + 1)
            seg   = values[start:i + 1]
            if len(seg) >= 4:
                fft_feats[i] = compute_fft_energy(seg)
        feature_rows.append(
            pd.DataFrame(
                fft_feats,
                index=group.index,
                columns=[f"{col}_fft_{k}" for k in range(5)]
            )
        )
    fft_df = pd.concat(feature_rows).sort_index()
    return pd.concat([df, fft_df], axis=1)


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].fillna(0))
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[✓] Scaler saved → {SCALER_PATH}")
    return scaler


def load_scaler() -> StandardScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run train_all.py first.")
    return joblib.load(SCALER_PATH)


def scale_features(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler = None) -> pd.DataFrame:
    df = df.copy()
    if scaler is None:
        scaler = load_scaler()
    df[feature_cols] = scaler.transform(df[feature_cols].fillna(0))
    return df


# ──────────────────────────────────────────────
# LSTM Sequence builder
# ──────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = None,
    window: int = WINDOW_SIZE,
    step: int = STEP_SIZE
) -> tuple:
    """
    Converts a flat time-series DataFrame into (X, y) arrays of shape:
      X: (n_sequences, window, n_features)
      y: (n_sequences,) — last value of target_col in window, if provided
    Sequences are built per device_id to avoid cross-device leakage.
    """
    X_list, y_list = [], []

    for _, group in df.groupby("device_id"):
        group = group.reset_index(drop=True)
        values  = group[feature_cols].fillna(0).values
        targets = group[target_col].values if target_col else None

        for i in range(0, len(values) - window, step):
            X_list.append(values[i: i + window])
            if targets is not None:
                y_list.append(targets[i + window - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    return X, y


def build_reconstruction_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    window: int = WINDOW_SIZE,
    step: int = STEP_SIZE
) -> np.ndarray:
    """
    For autoencoder training: returns X only (target = input).
    Filters to 'normal' samples (fault_label == 0) if column present.
    """
    if "fault_label" in df.columns:
        df = df[df["fault_label"] == 0].copy()
    X, _ = build_sequences(df, feature_cols, target_col=None, window=window, step=step)
    return X


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────

def run_feature_pipeline(df: pd.DataFrame, fit: bool = False) -> tuple:
    """
    End-to-end feature engineering.
    Returns (df_engineered, feature_cols, scaler)
    """
    print("Running feature pipeline...")
    df = add_rolling_features(df)
    df = add_rms(df)
    df = add_rate_of_change(df)
    df = add_fft_features(df, col="vibration")

    base_cols    = SENSOR_COLS
    roll_cols    = [c for c in df.columns if "_roll_" in c]
    rms_cols     = [c for c in df.columns if "_rms"   in c]
    roc_cols     = [c for c in df.columns if "_roc"   in c]
    fft_cols     = [c for c in df.columns if "_fft_"  in c]
    feature_cols = base_cols + roll_cols + rms_cols + roc_cols + fft_cols

    if fit:
        scaler = fit_scaler(df, feature_cols)
    else:
        scaler = load_scaler()

    df = scale_features(df, feature_cols, scaler)
    print(f"[✓] Feature pipeline complete. Total features: {len(feature_cols)}")
    return df, feature_cols, scaler


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "combined.csv")
    df_out, fcols, _ = run_feature_pipeline(df, fit=True)
    print(df_out[fcols[:10]].head())