"""
utils/config.py
Central configuration — all tunable constants in one place.
Override any value via environment variables.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"
LOGS_DIR    = ROOT_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────
INDUSTRIAL_CSV  = DATA_DIR / "industrial.csv"
SMARTHOME_CSV   = DATA_DIR / "smarthome.csv"
COMBINED_CSV    = DATA_DIR / "combined.csv"
DB_PATH         = DATA_DIR / "iot_maintenance.db"

N_SAMPLES_INDUSTRIAL  = int(os.getenv("N_SAMPLES_INDUSTRIAL",  "5000"))
N_SAMPLES_SMARTHOME   = int(os.getenv("N_SAMPLES_SMARTHOME",   "3000"))

# ── Preprocessing ──────────────────────────────────────────────────
SENSOR_COLS  = ["vibration", "temperature", "current", "pressure", "rpm", "humidity", "power_w"]
WINDOW_SIZE  = int(os.getenv("WINDOW_SIZE", "30"))
STEP_SIZE    = int(os.getenv("STEP_SIZE",   "5"))
ROLLING_WIN  = int(os.getenv("ROLLING_WIN", "20"))

# ── Model files ────────────────────────────────────────────────────
AE_MODEL_PATH     = MODELS_DIR / "lstm_autoencoder.keras"
AE_THRESHOLD_PATH = MODELS_DIR / "ae_threshold.pkl"
CLF_MODEL_PATH    = MODELS_DIR / "fault_classifier.pkl"
CLF_LABELS_PATH   = MODELS_DIR / "fault_label_map.pkl"
RUL_MODEL_PATH    = MODELS_DIR / "rul_estimator.pkl"
SCALER_PATH       = MODELS_DIR / "scaler.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"

# ── Training ───────────────────────────────────────────────────────
AE_EPOCHS       = int(os.getenv("AE_EPOCHS",    "30"))
AE_BATCH_SIZE   = int(os.getenv("AE_BATCH",     "64"))
AE_LATENT_DIM   = int(os.getenv("AE_LATENT",    "16"))
AE_LSTM_UNITS   = int(os.getenv("AE_LSTM",      "64"))
AE_THRESHOLD_PCT= float(os.getenv("AE_THRESH",  "95"))

RF_N_ESTIMATORS = int(os.getenv("RF_N_EST",     "200"))
RF_MAX_DEPTH    = int(os.getenv("RF_DEPTH",      "20"))

GB_N_ESTIMATORS = int(os.getenv("GB_N_EST",     "300"))
GB_LR           = float(os.getenv("GB_LR",      "0.05"))
GB_MAX_DEPTH    = int(os.getenv("GB_DEPTH",      "5"))

# ── RUL thresholds ─────────────────────────────────────────────────
RUL_MAX_CYCLES   = float(os.getenv("RUL_MAX",    "3500"))
RUL_CRITICAL     = float(os.getenv("RUL_CRIT",   "100"))
RUL_WARNING      = float(os.getenv("RUL_WARN",   "500"))
RUL_MODERATE     = float(os.getenv("RUL_MOD",    "1500"))

# ── Anomaly threshold ──────────────────────────────────────────────
ANOMALY_SCORE_THRESHOLD = float(os.getenv("ANOMALY_THRESH", "0.04"))

# ── API ────────────────────────────────────────────────────────────
API_HOST        = os.getenv("API_HOST", "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT", "8000"))
API_RELOAD      = os.getenv("API_RELOAD", "true").lower() == "true"

# ── MQTT ───────────────────────────────────────────────────────────
MQTT_BROKER     = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT       = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_BASE = os.getenv("MQTT_TOPIC", "iot/sensors")

# ── Simulator ──────────────────────────────────────────────────────
SIM_INTERVAL_SEC = float(os.getenv("SIM_INTERVAL", "2.0"))

# ── Dashboard ──────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC = int(os.getenv("DASH_REFRESH", "5"))

def summary():
    print("=== PredMaint Config ===")
    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Models dir:  {MODELS_DIR}")
    print(f"  Window:      {WINDOW_SIZE} steps")
    print(f"  AE epochs:   {AE_EPOCHS}")
    print(f"  RF trees:    {RF_N_ESTIMATORS}")
    print(f"  GB trees:    {GB_N_ESTIMATORS}")
    print(f"  API:         {API_HOST}:{API_PORT}")
    print(f"  MQTT:        {MQTT_BROKER}:{MQTT_PORT}")

if __name__ == "__main__":
    summary()
