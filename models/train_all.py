"""
models/train_all.py
Master training script. Run this once after generating the dataset.
Trains:
  1. LSTM Autoencoder (anomaly detection)
  2. Random Forest fault classifier
  3. Gradient Boosting RUL estimator
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import run_feature_pipeline, build_reconstruction_sequences, build_sequences
from models.autoencoder import train_autoencoder, fit_threshold
from models.fault_classifier import train_classifier
from models.rul_estimator import train_rul_model


DATA_PATH = ROOT / "data" / "combined.csv"


def main():
    # ── 0. Load data ──────────────────────────────────────────
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Running generator...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "data" / "generate_dataset.py")], check=True)

    print(f"\nLoading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Devices: {df['device_id'].nunique()}")
    print(f"  Labels: {df['fault_name'].value_counts().to_dict()}")

    # ── 1. Feature engineering ────────────────────────────────
    print("\n[Step 1] Feature engineering...")
    df_eng, feature_cols, scaler = run_feature_pipeline(df, fit=True)
    print(f"  Features: {len(feature_cols)}")

    # ── 2. Train LSTM Autoencoder ─────────────────────────────
    print("\n[Step 2] Training LSTM Autoencoder...")
    X_ae = build_reconstruction_sequences(df_eng, feature_cols, window=30, step=5)
    print(f"  AE sequences (normal only): {X_ae.shape}")

    if len(X_ae) < 100:
        print("  [!] Too few sequences — lowering step to 1")
        X_ae = build_reconstruction_sequences(df_eng, feature_cols, window=30, step=1)

    split = int(len(X_ae) * 0.8)
    X_ae_tr, X_ae_val = X_ae[:split], X_ae[split:]

    ae_model, _ = train_autoencoder(X_ae_tr, X_ae_val, epochs=30, batch_size=64)
    threshold   = fit_threshold(ae_model, X_ae_val, percentile=95)
    print(f"  Anomaly threshold: {threshold:.6f}")

    # ── 3. Train Fault Classifier ─────────────────────────────
    print("\n[Step 3] Training fault classifier (Random Forest)...")
    clf = train_classifier(df_eng, feature_cols, n_estimators=200)

    # ── 4. Train RUL Estimator ────────────────────────────────
    print("\n[Step 4] Training RUL estimator (Gradient Boosting)...")
    rul_model = train_rul_model(df_eng, feature_cols)

    # ── 5. Summary ────────────────────────────────────────────
    print("\n" + "="*50)
    print("Training complete! All models saved to models/")
    print("="*50)
    print("  • models/lstm_autoencoder.keras")
    print("  • models/ae_threshold.pkl")
    print("  • models/fault_classifier.pkl")
    print("  • models/fault_label_map.pkl")
    print("  • models/rul_estimator.pkl")
    print("  • models/scaler.pkl")
    print("\nNext: uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
