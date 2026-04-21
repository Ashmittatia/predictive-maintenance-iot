"""
tests/test_models.py
Unit tests for preprocessing, model interfaces, and API schemas.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# ── Preprocessing ──────────────────────────────────────────────────

class TestPreprocessing:
    def _make_df(self, n=200):
        np.random.seed(0)
        return pd.DataFrame({
            "device_id":  ["motor_01"] * n,
            "vibration":  np.random.randn(n),
            "temperature":np.random.randn(n) + 65,
            "current":    np.random.randn(n) + 10,
            "pressure":   np.random.randn(n) + 1,
            "rpm":        np.random.randn(n) + 1500,
            "humidity":   np.random.randn(n) + 50,
            "power_w":    np.random.randn(n) + 2000,
            "fault_label":np.zeros(n, dtype=int),
            "domain":     ["industrial"] * n,
        })

    def test_rolling_features(self):
        from utils.preprocessing import add_rolling_features
        df = self._make_df()
        df_out = add_rolling_features(df, window=10)
        assert "vibration_roll_mean" in df_out.columns
        assert "temperature_roll_std" in df_out.columns
        assert len(df_out) == len(df)

    def test_rms(self):
        from utils.preprocessing import add_rms
        df = self._make_df()
        df_out = add_rms(df)
        assert "vibration_rms" in df_out.columns
        assert df_out["vibration_rms"].notna().all()

    def test_rate_of_change(self):
        from utils.preprocessing import add_rate_of_change
        df = self._make_df()
        df_out = add_rate_of_change(df, cols=["vibration"])
        assert "vibration_roc" in df_out.columns

    def test_sequence_builder(self):
        from utils.preprocessing import build_sequences
        df = self._make_df(300)
        df["fault_label"] = 0
        feature_cols = ["vibration","temperature","current","pressure","rpm","humidity","power_w"]
        X, y = build_sequences(df, feature_cols, target_col="fault_label", window=30, step=5)
        assert X.ndim == 3
        assert X.shape[1] == 30
        assert X.shape[2] == len(feature_cols)
        assert y is not None

    def test_reconstruction_sequences(self):
        from utils.preprocessing import build_reconstruction_sequences
        df = self._make_df(300)
        feature_cols = ["vibration","temperature","current","pressure","rpm","humidity","power_w"]
        X = build_reconstruction_sequences(df, feature_cols, window=30, step=5)
        assert X.ndim == 3
        assert X.shape[2] == len(feature_cols)


# ── Autoencoder ────────────────────────────────────────────────────

class TestAutoencoder:
    def test_build(self):
        from models.autoencoder import build_autoencoder
        model = build_autoencoder(n_features=7, window=30)
        assert model is not None
        assert model.input_shape == (None, 30, 7)

    def test_reconstruction_shape(self):
        from models.autoencoder import build_autoencoder, compute_reconstruction_errors
        model = build_autoencoder(n_features=5, window=10)
        X = np.random.randn(20, 10, 5).astype(np.float32)
        errors = compute_reconstruction_errors(model, X)
        assert errors.shape == (20,)
        assert (errors >= 0).all()

    def test_predict_anomaly_output(self):
        from models.autoencoder import build_autoencoder, predict_anomaly
        model = build_autoencoder(n_features=5, window=10)
        X = np.random.randn(1, 10, 5).astype(np.float32)
        result = predict_anomaly(model, X, threshold=0.1)
        assert "anomaly_score"   in result
        assert "is_anomaly"      in result
        assert "threshold"       in result
        assert isinstance(result["is_anomaly"], bool)


# ── Fault Classifier ───────────────────────────────────────────────

class TestFaultClassifier:
    def _make_df(self, n=500):
        np.random.seed(0)
        return pd.DataFrame({
            "device_id":  ["motor_01"] * n,
            "vibration":  np.random.randn(n),
            "temperature":np.random.randn(n),
            "current":    np.random.randn(n),
            "pressure":   np.random.randn(n),
            "rpm":        np.random.randn(n),
            "humidity":   np.random.randn(n),
            "power_w":    np.random.randn(n),
            "fault_label":np.random.randint(0, 3, n),
            "domain":     ["industrial"] * n,
            "rul":        np.random.randint(0, 3000, n),
        })

    def test_train_and_predict(self):
        from models.fault_classifier import train_classifier, predict_fault
        df = self._make_df()
        feature_cols = ["vibration","temperature","current","pressure","rpm","humidity","power_w"]
        clf = train_classifier(df, feature_cols, n_estimators=10)
        result = predict_fault(clf, {0:"normal",1:"fault_a",2:"fault_b"}, np.zeros((1, len(feature_cols))))
        assert "fault_label"    in result
        assert "fault_name"     in result
        assert "confidence"     in result
        assert "probabilities"  in result


# ── RUL Estimator ──────────────────────────────────────────────────

class TestRULEstimator:
    def _make_df(self, n=500):
        np.random.seed(1)
        return pd.DataFrame({
            "device_id":  ["motor_01"] * n,
            "vibration":  np.random.randn(n),
            "temperature":np.random.randn(n),
            "current":    np.random.randn(n),
            "pressure":   np.random.randn(n),
            "rpm":        np.random.randn(n),
            "humidity":   np.random.randn(n),
            "power_w":    np.random.randn(n),
            "domain":     ["industrial"] * n,
            "rul":        np.random.randint(0, 3000, n),
        })

    def test_train_and_predict(self):
        from models.rul_estimator import train_rul_model, predict_rul
        df = self._make_df()
        feature_cols = ["vibration","temperature","current","pressure","rpm","humidity","power_w"]
        model = train_rul_model(df, feature_cols)
        result = predict_rul(model, np.zeros((1, len(feature_cols))))
        assert "rul_cycles"    in result
        assert "health_score"  in result
        assert "urgency"       in result
        assert result["rul_cycles"] >= 0
        assert 0 <= result["health_score"] <= 100


# ── Database ───────────────────────────────────────────────────────

class TestDatabase:
    def test_init_and_insert(self, tmp_path):
        import utils.database as db
        # Patch DB path
        original = db.DB_PATH
        db.DB_PATH = tmp_path / "test.db"

        db.init_db()
        db.insert_reading("test_dev", "motor", "industrial", {
            "vibration": 0.5, "temperature": 65, "current": 10,
            "pressure": 1.0, "rpm": 1500, "humidity": 50, "power_w": 2000
        })
        readings = db.get_recent_readings("test_dev", limit=10)
        assert len(readings) >= 1
        assert readings[0]["device_id"] == "test_dev"

        db.DB_PATH = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
