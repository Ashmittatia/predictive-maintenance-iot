# ⚙️ PredMaint — IoT Predictive Maintenance System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"/>
</p>

<p align="center">
  A full-stack ML system for predicting equipment failures in <strong>industrial machinery</strong> and <strong>smart home devices</strong> — combining anomaly detection, fault classification, and Remaining Useful Life (RUL) estimation with a real-time monitoring dashboard.
</p>

---

## 📸 Dashboard Preview

| Overview | Device Deep Dive |
|----------|-----------------|
| Fleet health grid · anomaly heatmap · KPI cards | Sensor time series · RUL gauge · fault probabilities |

| Alert Center | Model Performance |
|--------------|------------------|
| Severity-ranked alerts · 24hr trend | Training curves · confusion matrix · actual vs predicted RUL |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 — Data Sources                                      │
│  Industrial sensors  │  Smart home sensors  │  Simulator    │
└──────────────────────────────┬──────────────────────────────┘
                               │ MQTT / HTTP
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 2 — Edge Processing                                   │
│  Rolling stats · RMS · FFT features · Sliding windows        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 3 — ML Models                                         │
│  LSTM Autoencoder  │  Random Forest  │  Gradient Boosting   │
│  (anomaly detect.) │  (fault class.) │  (RUL regression)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 4 — FastAPI Backend                                   │
│  /predict/full · /alerts · /devices · /history              │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 5 — Streamlit Dashboard                               │
│  Overview · Device Deep Dive · Alert Center · Model Perf    │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Dual-domain coverage** — 5 industrial devices (motors, pumps, compressors) + 5 smart home devices (ACs, washing machines, water heaters)
- **3 ML models** trained end-to-end on synthetic sensor telemetry:
  - LSTM Autoencoder for unsupervised anomaly detection (threshold at 95th percentile of normal reconstruction error)
  - Random Forest for 11-class fault type classification (99% accuracy)
  - Gradient Boosting for RUL regression (MAE ≈ 12 cycles, R² = 0.998)
- **Feature engineering pipeline** — rolling mean/std/min/max, RMS energy, FFT frequency components, rate of change
- **FastAPI inference server** with 8 REST endpoints, SQLite persistence, and auto-alert generation
- **Streamlit dashboard** — 4 pages: fleet overview, per-device deep dive, alert center, model performance
- **MQTT sensor simulator** — 10 device profiles publishing degrading telemetry
- **Dockerized** — one `docker-compose up` to run the entire stack
- **Configurable** via environment variables (`.env.example` provided)

---

## 📁 Project Structure

```
predictive_maintenance/
├── data/
│   └── generate_dataset.py     # Synthetic sensor data generator
├── models/
│   ├── autoencoder.py          # LSTM Autoencoder (anomaly detection)
│   ├── fault_classifier.py     # Random Forest (fault classification)
│   ├── rul_estimator.py        # Gradient Boosting (RUL regression)
│   └── train_all.py            # Master training script
├── utils/
│   ├── preprocessing.py        # Feature engineering pipeline
│   ├── database.py             # SQLite persistence layer
│   ├── alert_engine.py         # Threshold evaluation + email alerts
│   ├── inference_pipeline.py   # Unified inference interface
│   └── config.py               # Centralised configuration
├── api/
│   └── main.py                 # FastAPI server (8 endpoints)
├── simulator/
│   └── sensor_simulator.py     # MQTT + HTTP IoT data publisher
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── tests/
│   └── test_models.py          # pytest unit tests
├── docker/                     # Dockerfiles + Mosquitto config
├── docker-compose.yml          # Full stack orchestration
├── run.py                      # One-command project launcher
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-iot.git
cd predictive-maintenance-iot
pip install -r requirements.txt
```

### 2. Generate data and train models

```bash
python run.py setup
```

This runs in two steps — generates 40,000 rows of synthetic sensor data across 10 devices, then trains all 3 ML models.

Expected output:
```
[✓] Fault Classifier  — accuracy: 0.99  macro F1: 0.98
[✓] RUL Estimator     — MAE: 12.59  R²: 0.9977
[✓] LSTM Autoencoder  — threshold: 2.676
Training complete! All models saved to models/
```

### 3. Launch everything

```bash
python run.py all
```

Or launch services individually in separate terminals:

```bash
python run.py api         # FastAPI   → http://localhost:8000
python run.py dashboard   # Streamlit → http://localhost:8501
python run.py simulate    # Sensor simulator → pushes to API
```

### 4. (Optional) Docker

```bash
docker-compose up --build
```

---

## 🔌 API Endpoints

Base URL: `http://localhost:8000` · Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + model load status |
| `POST` | `/predict/anomaly` | LSTM Autoencoder anomaly score |
| `POST` | `/predict/fault` | Random Forest fault classification |
| `POST` | `/predict/rul` | Gradient Boosting RUL estimate |
| `POST` | `/predict/full` | All three models in one call |
| `GET` | `/devices` | All devices with latest predictions |
| `GET` | `/alerts` | Active unresolved alerts |
| `GET` | `/history/{device_id}` | Sensor + prediction history |

### Example request

```bash
curl -X POST http://localhost:8000/predict/full \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "motor_01",
    "device_type": "motor",
    "domain": "industrial",
    "vibration": 2.8,
    "temperature": 89.5,
    "current": 14.2,
    "pressure": 0.72,
    "rpm": 1280,
    "humidity": 52.0,
    "power_w": 3100
  }'
```

```json
{
  "device_id": "motor_01",
  "anomaly_score": 0.07823,
  "is_anomaly": true,
  "fault_name": "bearing_wear",
  "fault_confidence": 0.94,
  "rul_cycles": 312.5,
  "health_score": 8.9,
  "urgency": "warning",
  "timestamp": "2024-01-01T12:00:00"
}
```

---

## 🤖 ML Models

### LSTM Autoencoder — Anomaly Detection

Trained exclusively on normal sensor sequences. At inference time, an anomaly score (MSE reconstruction error) is computed — readings above the 95th-percentile threshold of normal errors are flagged.

```
Architecture : LSTM(64) → LSTM(32) → Dense(16) → RepeatVector → LSTM(32) → LSTM(64) → TimeDistributed Dense
Window       : 30 timesteps × 49 features
Parameters   : 76,417
```

### Random Forest — Fault Classification

Multi-class classifier across 11 fault types spanning both domains. Balanced class weights handle the skewed label distribution (normal = 67% of data).

```
Features  : 49 engineered (base sensors + rolling stats + RMS + FFT + rate-of-change)
Classes   : normal, bearing_wear, imbalance, overheating, cavitation, valve_leak,
            refrigerant_leak, compressor_fail, drum_imbalance, pump_fail, heating_element_fail
Accuracy  : 99%   Macro F1: 0.98
```

### Gradient Boosting — RUL Estimation

Regresses time-to-failure in cycles from the engineered feature vector. RUL is capped at the 95th percentile during training to prevent outliers dominating the loss.

```
MAE   : 12.59 cycles
RMSE  : 25.84 cycles
R²    : 0.9977
```

---

## 📊 Dataset

Fully synthetic and reproducible via `data/generate_dataset.py`.

| Split | Devices | Rows | Domain |
|-------|---------|------|--------|
| Industrial | motor ×2, pump ×2, compressor ×1 | 25,000 | Industrial |
| Smart Home | AC ×2, washing machine ×2, water heater ×1 | 15,000 | Smart Home |
| **Combined** | **10 devices** | **40,000** | **Both** |

Each row: `vibration, temperature, current, pressure, rpm, humidity, power_w, fault_label, fault_name, rul, degradation`

Signals follow a sinusoidal base + Gaussian noise, with a degradation curve injected from ~65% of device lifecycle onward that progressively shifts all readings toward fault conditions.

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

Covers preprocessing pipeline, autoencoder architecture, fault classifier, RUL estimator, and database CRUD.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `WINDOW_SIZE` | `30` | LSTM sequence length (timesteps) |
| `AE_EPOCHS` | `30` | Autoencoder training epochs |
| `RUL_CRIT` | `100` | Critical RUL threshold (cycles) |
| `RUL_WARN` | `500` | Warning RUL threshold (cycles) |
| `MQTT_BROKER` | `localhost` | MQTT broker host |
| `SIM_INTERVAL` | `2.0` | Simulator publish interval (seconds) |

---

## 🗺️ Roadmap

- [ ] NASA CMAPSS dataset integration
- [ ] ONNX model export for edge deployment
- [ ] Grafana + InfluxDB dashboard alternative
- [ ] Attention-based LSTM for improved anomaly detection
- [ ] REST webhook support for alert notifications
- [ ] JWT auth on FastAPI
- [ ] GitHub Actions CI pipeline

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ashmit Tatia**  
B.Tech AI & ML · NMIMS University, Mumbai  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

<p align="center">
  Built as a portfolio project demonstrating end-to-end MLOps —<br/>
  data generation → feature engineering → model training → REST API → real-time dashboard → Docker deployment.
</p>
