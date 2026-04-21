"""
api/main.py
FastAPI REST API for predictive maintenance inference.

Endpoints:
  POST /predict/anomaly   — LSTM Autoencoder anomaly detection
  POST /predict/fault     — Random Forest fault classification
  POST /predict/rul       — Gradient Boosting RUL estimation
  POST /predict/full      — Run all three models on one request
  GET  /devices           — List all devices with latest status
  GET  /alerts            — Get active alerts
  GET  /alerts/{id}/resolve — Mark alert as resolved
  GET  /history/{device_id} — Prediction history for a device
  GET  /health            — API liveness check
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.database import (
    init_db, insert_reading, insert_prediction, insert_alert,
    get_recent_readings, get_latest_prediction, get_active_alerts,
    get_all_device_ids, resolve_alert, get_predictions_history,
)

app = FastAPI(
    title="IoT Predictive Maintenance API",
    description="Real-time ML inference for industrial and smart home device health monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded models ─────────────────────────────────────────────
_ae_model    = None
_ae_thresh   = None
_clf         = None
_clf_labels  = None
_rul_model   = None
_scaler      = None
_feature_cols: List[str] = []


def _load_models():
    global _ae_model, _ae_thresh, _clf, _clf_labels, _rul_model, _scaler, _feature_cols
    if _ae_model is not None:
        return

    try:
        from models.autoencoder     import load_autoencoder
        from models.fault_classifier import load_classifier
        from models.rul_estimator   import load_rul_model
        from utils.preprocessing    import load_scaler

        _ae_model, _ae_thresh = load_autoencoder()
        _clf, _clf_labels     = load_classifier()
        _rul_model            = load_rul_model()
        _scaler               = load_scaler()

        # Reconstruct feature cols from scaler
        import joblib
        fc_path = ROOT / "models" / "feature_cols.pkl"
        if fc_path.exists():
            _feature_cols = joblib.load(fc_path)
        else:
            _feature_cols = list(range(_scaler.n_features_in_))

        print("[API] All models loaded successfully.")
    except Exception as e:
        print(f"[API] Warning: Could not load models — {e}")
        print("[API] Running in mock mode. Train models first.")


# ── Pydantic schemas ───────────────────────────────────────────────

class SensorReading(BaseModel):
    device_id:   str
    device_type: str = "unknown"
    domain:      str = "industrial"
    vibration:   float = 0.5
    temperature: float = 65.0
    current:     float = 10.0
    pressure:    float = 1.0
    rpm:         float = 1500.0
    humidity:    float = 50.0
    power_w:     float = 2000.0

class WindowRequest(BaseModel):
    device_id:   str
    device_type: str = "unknown"
    domain:      str = "industrial"
    readings:    List[SensorReading] = Field(..., min_length=1)

class AnomalyResponse(BaseModel):
    device_id:     str
    anomaly_score: float
    is_anomaly:    bool
    threshold:     float
    confidence:    float
    timestamp:     str

class FaultResponse(BaseModel):
    device_id:     str
    fault_label:   int
    fault_name:    str
    confidence:    float
    probabilities: Dict[str, float]
    timestamp:     str

class RULResponse(BaseModel):
    device_id:    str
    rul_cycles:   float
    health_score: float
    urgency:      str
    timestamp:    str

class FullPrediction(BaseModel):
    device_id:     str
    anomaly_score: float
    is_anomaly:    bool
    fault_name:    str
    fault_confidence: float
    rul_cycles:    float
    health_score:  float
    urgency:       str
    timestamp:     str


# ── Helpers ────────────────────────────────────────────────────────

SENSOR_COLS = ["vibration", "temperature", "current", "pressure", "rpm", "humidity", "power_w"]


def _reading_to_vector(r: SensorReading) -> np.ndarray:
    return np.array([[r.vibration, r.temperature, r.current, r.pressure, r.rpm, r.humidity, r.power_w]])


def _mock_prediction(device_id: str) -> dict:
    """Fallback when models are not trained yet."""
    import random
    return {
        "anomaly_score":    round(random.uniform(0.001, 0.08), 5),
        "is_anomaly":       random.random() < 0.15,
        "threshold":        0.05,
        "confidence":       round(random.uniform(0.6, 0.99), 3),
        "fault_label":      0,
        "fault_name":       "normal",
        "fault_confidence": round(random.uniform(0.7, 0.99), 3),
        "probabilities":    {"normal": 0.9},
        "rul_cycles":       round(random.uniform(200, 3000), 1),
        "health_score":     round(random.uniform(40, 100), 1),
        "urgency":          random.choice(["healthy", "healthy", "moderate", "warning"]),
    }


def _auto_alert(device_id: str, pred: dict):
    if pred.get("urgency") == "critical":
        insert_alert(device_id, "rul_critical",   "critical", f"Device {device_id}: RUL critically low ({pred['rul_cycles']:.0f} cycles)")
    elif pred.get("urgency") == "warning":
        insert_alert(device_id, "rul_warning",    "warning",  f"Device {device_id}: RUL warning ({pred['rul_cycles']:.0f} cycles)")
    if pred.get("is_anomaly"):
        insert_alert(device_id, "anomaly_detected","high",    f"Device {device_id}: Anomaly detected (score={pred['anomaly_score']:.4f})")


# ── Startup ────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()
    _load_models()


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "models_loaded": _ae_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(req: SensorReading):
    _load_models()
    if _ae_model is None:
        mock = _mock_prediction(req.device_id)
        return AnomalyResponse(
            device_id=req.device_id, timestamp=datetime.utcnow().isoformat(), **{k: mock[k] for k in ["anomaly_score","is_anomaly","threshold","confidence"]}
        )
    try:
        from models.autoencoder import predict_anomaly as ae_predict
        # Use single reading replicated to fill window
        vec   = _reading_to_vector(req)
        X_win = np.tile(vec, (30, 1))[np.newaxis, :, :7].astype(np.float32)
        result = ae_predict(_ae_model, X_win, _ae_thresh)
        result["device_id"]  = req.device_id
        result["timestamp"]  = datetime.utcnow().isoformat()
        return AnomalyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fault", response_model=FaultResponse)
def predict_fault(req: SensorReading):
    _load_models()
    if _clf is None:
        mock = _mock_prediction(req.device_id)
        return FaultResponse(device_id=req.device_id, timestamp=datetime.utcnow().isoformat(),
                             fault_label=mock["fault_label"], fault_name=mock["fault_name"],
                             confidence=mock["fault_confidence"], probabilities=mock["probabilities"])
    try:
        from models.fault_classifier import predict_fault as clf_predict
        vec    = _reading_to_vector(req)
        result = clf_predict(_clf, _clf_labels, vec)
        result["device_id"]  = req.device_id
        result["timestamp"]  = datetime.utcnow().isoformat()
        return FaultResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rul", response_model=RULResponse)
def predict_rul(req: SensorReading):
    _load_models()
    if _rul_model is None:
        mock = _mock_prediction(req.device_id)
        return RULResponse(device_id=req.device_id, timestamp=datetime.utcnow().isoformat(),
                           rul_cycles=mock["rul_cycles"], health_score=mock["health_score"], urgency=mock["urgency"])
    try:
        from models.rul_estimator import predict_rul as rul_predict
        vec    = _reading_to_vector(req)
        result = rul_predict(_rul_model, vec)
        result["device_id"]  = req.device_id
        result["timestamp"]  = datetime.utcnow().isoformat()
        return RULResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full", response_model=FullPrediction)
def predict_full(req: SensorReading):
    _load_models()
    pred = _mock_prediction(req.device_id)

    if _ae_model and _clf and _rul_model:
        try:
            from models.autoencoder      import predict_anomaly as ae_pred
            from models.fault_classifier import predict_fault   as clf_pred
            from models.rul_estimator    import predict_rul     as rul_pred

            vec   = _reading_to_vector(req)
            X_win = np.tile(vec, (30, 1))[np.newaxis, :, :7].astype(np.float32)
            ae_r  = ae_pred(_ae_model, X_win, _ae_thresh)
            clf_r = clf_pred(_clf, _clf_labels, vec)
            rul_r = rul_pred(_rul_model, vec)

            pred = {**ae_r, **clf_r, **rul_r, "fault_confidence": clf_r["confidence"]}
        except Exception as e:
            print(f"[API] Inference error: {e}")

    # Persist
    sensors = {c: getattr(req, c) for c in SENSOR_COLS}
    insert_reading(req.device_id, req.device_type, req.domain, sensors)
    insert_prediction(req.device_id, pred)
    _auto_alert(req.device_id, pred)

    return FullPrediction(
        device_id       = req.device_id,
        anomaly_score   = pred["anomaly_score"],
        is_anomaly      = pred["is_anomaly"],
        fault_name      = pred.get("fault_name", "normal"),
        fault_confidence= pred.get("fault_confidence", pred.get("confidence", 0.0)),
        rul_cycles      = pred["rul_cycles"],
        health_score    = pred["health_score"],
        urgency         = pred["urgency"],
        timestamp       = datetime.utcnow().isoformat(),
    )


@app.get("/devices")
def list_devices():
    device_ids = get_all_device_ids()
    if not device_ids:
        # Return default device list if DB is empty
        device_ids = [
            "motor_01","motor_02","pump_01","pump_02","compressor_01",
            "ac_01","ac_02","washing_machine_01","washing_machine_02","water_heater_01",
        ]
    result = []
    for did in device_ids:
        latest = get_latest_prediction(did)
        result.append({"device_id": did, "latest_prediction": latest})
    return {"devices": result, "count": len(result)}


@app.get("/alerts")
def list_alerts():
    alerts = get_active_alerts(limit=50)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/alerts/{alert_id}/resolve")
def resolve(alert_id: int):
    resolve_alert(alert_id)
    return {"status": "resolved", "alert_id": alert_id}


@app.get("/history/{device_id}")
def device_history(device_id: str, limit: int = 100):
    readings    = get_recent_readings(device_id, limit=limit)
    predictions = get_predictions_history(device_id, limit=limit)
    return {
        "device_id":   device_id,
        "readings":    readings,
        "predictions": predictions,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
