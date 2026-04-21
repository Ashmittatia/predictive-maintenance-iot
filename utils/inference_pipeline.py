"""
utils/inference_pipeline.py
Batch inference utility — runs all three models on a DataFrame of readings.
Used by the dashboard for offline/batch analysis and by the API for bulk requests.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from typing import Optional
import joblib

from utils.config import (
    SENSOR_COLS, WINDOW_SIZE, STEP_SIZE,
    AE_MODEL_PATH, AE_THRESHOLD_PATH,
    CLF_MODEL_PATH, CLF_LABELS_PATH,
    RUL_MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH,
)


class PredMaintenancePipeline:
    """
    End-to-end inference pipeline.
    Loads all three models once; call .predict_dataframe() or .predict_single() at runtime.
    """

    def __init__(self, auto_load: bool = True):
        self.ae_model    = None
        self.ae_threshold= 0.05
        self.clf         = None
        self.clf_labels  = {}
        self.rul_model   = None
        self.scaler      = None
        self.feature_cols= SENSOR_COLS[:]
        self._loaded     = False

        if auto_load:
            self.load()

    def load(self) -> bool:
        """Load all models. Returns True on success, False if models not trained yet."""
        try:
            from models.autoencoder      import load_autoencoder
            from models.fault_classifier import load_classifier
            from models.rul_estimator    import load_rul_model
            from utils.preprocessing     import load_scaler

            self.ae_model, self.ae_threshold = load_autoencoder()
            self.clf, self.clf_labels        = load_classifier()
            self.rul_model                   = load_rul_model()
            self.scaler                      = load_scaler()

            if FEATURE_COLS_PATH.exists():
                self.feature_cols = joblib.load(FEATURE_COLS_PATH)

            self._loaded = True
            print(f"[Pipeline] All models loaded. Features: {len(self.feature_cols)}")
            return True

        except FileNotFoundError as e:
            print(f"[Pipeline] Models not found — {e}")
            print("[Pipeline] Run: python models/train_all.py")
            return False

        except Exception as e:
            print(f"[Pipeline] Load error — {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded and all(m is not None for m in [self.ae_model, self.clf, self.rul_model])

    def _get_feature_vector(self, row: pd.Series) -> np.ndarray:
        """Extract feature vector from a single DataFrame row."""
        vec = []
        for col in self.feature_cols:
            vec.append(float(row.get(col, 0.0)))
        return np.array(vec, dtype=np.float32).reshape(1, -1)

    def _get_window(self, df: pd.DataFrame, end_idx: int) -> np.ndarray:
        """Extract a WINDOW_SIZE × n_features window ending at end_idx."""
        start = max(0, end_idx - WINDOW_SIZE + 1)
        sub   = df.iloc[start:end_idx + 1]
        data  = sub[SENSOR_COLS].fillna(0).values.astype(np.float32)
        # Pad if shorter than window
        if len(data) < WINDOW_SIZE:
            pad  = np.zeros((WINDOW_SIZE - len(data), data.shape[1]), dtype=np.float32)
            data = np.vstack([pad, data])
        return data[np.newaxis, :, :]   # (1, WINDOW_SIZE, n_features)

    def predict_single(self, sensors: dict, device_id: str = "device") -> dict:
        """
        Run full inference on a single sensor reading dict.
        Falls back to heuristic scores if models not loaded.
        """
        if not self.is_ready:
            return self._heuristic_predict(sensors, device_id)

        from models.autoencoder      import predict_anomaly
        from models.fault_classifier import predict_fault
        from models.rul_estimator    import predict_rul

        # Build window from single reading (replicate)
        arr   = np.array([[sensors.get(c, 0.0) for c in SENSOR_COLS]], dtype=np.float32)
        X_win = np.tile(arr, (WINDOW_SIZE, 1))[np.newaxis, :, :]

        vec   = np.array([[sensors.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)

        ae_r  = predict_anomaly(self.ae_model, X_win, self.ae_threshold)
        clf_r = predict_fault(self.clf, self.clf_labels, vec)
        rul_r = predict_rul(self.rul_model, vec)

        return {
            "device_id":       device_id,
            "anomaly_score":   ae_r["anomaly_score"],
            "is_anomaly":      ae_r["is_anomaly"],
            "threshold":       ae_r["threshold"],
            "fault_label":     clf_r["fault_label"],
            "fault_name":      clf_r["fault_name"],
            "fault_confidence":clf_r["confidence"],
            "probabilities":   clf_r["probabilities"],
            "rul_cycles":      rul_r["rul_cycles"],
            "health_score":    rul_r["health_score"],
            "urgency":         rul_r["urgency"],
        }

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        device_col: str = "device_id",
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run full inference on every row of a DataFrame.
        Returns the input DataFrame with prediction columns appended.
        """
        results = []
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            if progress and i % 500 == 0:
                print(f"  [{i}/{n}]", end="\r")

            sensors = {c: row.get(c, 0.0) for c in SENSOR_COLS}
            device  = str(row.get(device_col, "unknown"))
            pred    = self.predict_single(sensors, device)
            results.append(pred)

        if progress:
            print(f"  [{n}/{n}] Done.")

        pred_df = pd.DataFrame(results)
        out_df  = df.reset_index(drop=True).copy()
        for col in ["anomaly_score","is_anomaly","fault_label","fault_name","fault_confidence","rul_cycles","health_score","urgency"]:
            if col in pred_df.columns:
                out_df[col] = pred_df[col].values
        return out_df

    @staticmethod
    def _heuristic_predict(sensors: dict, device_id: str) -> dict:
        """Simple heuristic when models are not loaded — useful for demos."""
        vib   = sensors.get("vibration",  0.5)
        temp  = sensors.get("temperature",65.0)
        cur   = sensors.get("current",    10.0)

        deg_score = min(1.0, (max(0, vib - 0.5) / 2.5 + max(0, temp - 70) / 25 + max(0, cur - 12) / 5) / 3)
        rul       = max(0, 3200 * (1 - deg_score))
        health    = round((1 - deg_score) * 100, 1)
        anomaly   = deg_score > 0.4

        if rul < 100:   urgency = "critical"
        elif rul < 500: urgency = "warning"
        elif rul < 1500:urgency = "moderate"
        else:           urgency = "healthy"

        fault = "normal" if deg_score < 0.3 else ("overheating" if temp > 80 else "bearing_wear")

        return {
            "device_id":       device_id,
            "anomaly_score":   round(deg_score * 0.08, 5),
            "is_anomaly":      anomaly,
            "threshold":       0.04,
            "fault_label":     0 if fault == "normal" else 1,
            "fault_name":      fault,
            "fault_confidence":round(0.6 + 0.4 * deg_score, 3),
            "probabilities":   {"normal": round(1 - deg_score, 3), fault: round(deg_score, 3)},
            "rul_cycles":      round(rul, 1),
            "health_score":    health,
            "urgency":         urgency,
        }


# ── Module-level singleton ─────────────────────────────────────────
_pipeline: Optional[PredMaintenancePipeline] = None


def get_pipeline() -> PredMaintenancePipeline:
    """Return the shared pipeline instance (lazy init)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PredMaintenancePipeline(auto_load=True)
    return _pipeline


if __name__ == "__main__":
    p = get_pipeline()
    test = {
        "vibration": 2.8, "temperature": 89, "current": 15,
        "pressure": 0.7, "rpm": 1200, "humidity": 55, "power_w": 3200,
    }
    result = p.predict_single(test, device_id="motor_01")
    print("\nSingle prediction:")
    for k, v in result.items():
        print(f"  {k:22s}: {v}")
