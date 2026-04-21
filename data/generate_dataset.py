"""
data/generate_dataset.py
Generates labeled synthetic sensor datasets for:
  - Industrial machinery (motors, pumps, compressors)
  - Smart home devices (ACs, washing machines, water heaters)
Outputs: data/industrial.csv, data/smarthome.csv, data/combined.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Industrial: Motor / Pump / Compressor
# ──────────────────────────────────────────────

INDUSTRIAL_DEVICES = [
    {"id": "motor_01",      "type": "motor"},
    {"id": "motor_02",      "type": "motor"},
    {"id": "pump_01",       "type": "pump"},
    {"id": "pump_02",       "type": "pump"},
    {"id": "compressor_01", "type": "compressor"},
]

INDUSTRIAL_FAULT_LABELS = {
    0: "normal",
    1: "bearing_wear",
    2: "imbalance",
    3: "overheating",
    4: "cavitation",      # pump-specific
    5: "valve_leak",      # compressor-specific
}

SMARTHOME_DEVICES = [
    {"id": "ac_01",             "type": "air_conditioner"},
    {"id": "ac_02",             "type": "air_conditioner"},
    {"id": "washing_machine_01","type": "washing_machine"},
    {"id": "washing_machine_02","type": "washing_machine"},
    {"id": "water_heater_01",   "type": "water_heater"},
]

SMARTHOME_FAULT_LABELS = {
    0: "normal",
    1: "refrigerant_leak",    # AC
    2: "compressor_fail",     # AC
    3: "drum_imbalance",      # washing machine
    4: "pump_fail",           # washing machine / water heater
    5: "heating_element_fail",# water heater
}


def _add_degradation(n_samples: int, fault_start_ratio: float = 0.6) -> np.ndarray:
    """Returns a degradation curve 0→1 that starts rising after fault_start_ratio."""
    deg = np.zeros(n_samples)
    start = int(n_samples * fault_start_ratio)
    ramp = np.linspace(0, 1, n_samples - start)
    deg[start:] = ramp
    return deg


def generate_industrial(n_samples_per_device: int = 5000) -> pd.DataFrame:
    records = []
    for dev in INDUSTRIAL_DEVICES:
        t = np.arange(n_samples_per_device)
        deg = _add_degradation(n_samples_per_device, fault_start_ratio=0.65)

        # Base signals
        vibration     = 0.5 + 0.3 * np.sin(2 * np.pi * t / 50) + 0.05 * np.random.randn(n_samples_per_device)
        temperature   = 65  + 5   * np.sin(2 * np.pi * t / 200) + 1.0  * np.random.randn(n_samples_per_device)
        current       = 10  + 1   * np.sin(2 * np.pi * t / 100) + 0.2  * np.random.randn(n_samples_per_device)
        pressure      = 1.0 + 0.1 * np.sin(2 * np.pi * t / 80)  + 0.01 * np.random.randn(n_samples_per_device)
        rpm           = 1500 + 50 * np.sin(2 * np.pi * t / 300)  + 10   * np.random.randn(n_samples_per_device)

        # Inject degradation effects
        vibration   += deg * np.random.uniform(1.5, 3.0)
        temperature += deg * np.random.uniform(10, 25)
        current     += deg * np.random.uniform(2, 5)
        pressure    += deg * np.random.uniform(-0.3, 0.5)

        # Assign fault label
        fault = np.zeros(n_samples_per_device, dtype=int)
        fault_start = int(n_samples_per_device * 0.65)
        if dev["type"] == "motor":
            fault[fault_start:] = np.where(vibration[fault_start:] > 2.5, 2, 1)
        elif dev["type"] == "pump":
            fault[fault_start:] = np.where(pressure[fault_start:] < 0.8, 4, 1)
        elif dev["type"] == "compressor":
            fault[fault_start:] = np.where(current[fault_start:] > 14, 3, 5)

        # RUL: counts down from n_samples at fault_start to 0
        rul = np.maximum(0, (n_samples_per_device - fault_start) - np.maximum(0, t - fault_start))

        for i in range(n_samples_per_device):
            records.append({
                "timestamp":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i),
                "device_id":   dev["id"],
                "device_type": dev["type"],
                "domain":      "industrial",
                "vibration":   round(vibration[i], 4),
                "temperature": round(temperature[i], 2),
                "current":     round(current[i], 3),
                "pressure":    round(pressure[i], 4),
                "rpm":         round(rpm[i], 1),
                "humidity":    round(50 + 5 * np.random.randn(), 2),   # not primary for industrial
                "power_w":     round(current[i] * 220 * 0.85, 2),
                "fault_label": int(fault[i]),
                "fault_name":  INDUSTRIAL_FAULT_LABELS[int(fault[i])],
                "rul":         int(rul[i]),
                "degradation": round(deg[i], 4),
            })

    df = pd.DataFrame(records)
    out = OUTPUT_DIR / "industrial.csv"
    df.to_csv(out, index=False)
    print(f"[✓] Industrial dataset saved → {out}  ({len(df):,} rows)")
    return df


def generate_smarthome(n_samples_per_device: int = 3000) -> pd.DataFrame:
    records = []
    for dev in SMARTHOME_DEVICES:
        t = np.arange(n_samples_per_device)
        deg = _add_degradation(n_samples_per_device, fault_start_ratio=0.7)

        # Base signals
        temperature = 22  + 3   * np.sin(2 * np.pi * t / 100) + 0.5 * np.random.randn(n_samples_per_device)
        humidity    = 55  + 8   * np.sin(2 * np.pi * t / 120) + 1.0 * np.random.randn(n_samples_per_device)
        power_w     = 1200 + 100 * np.sin(2 * np.pi * t / 80)  + 20  * np.random.randn(n_samples_per_device)
        vibration   = 0.1 + 0.05 * np.random.randn(n_samples_per_device)
        noise_db    = 35  + 5   * np.random.randn(n_samples_per_device)
        current     = power_w / 220

        # Device-specific base adjustments
        if dev["type"] == "water_heater":
            temperature = 60 + 5 * np.sin(2 * np.pi * t / 200) + 1.5 * np.random.randn(n_samples_per_device)
            power_w     = 2000 + 150 * np.random.randn(n_samples_per_device)

        # Inject degradation
        power_w     += deg * np.random.uniform(200, 600)
        temperature += deg * np.random.uniform(3, 10)
        vibration   += deg * np.random.uniform(0.2, 0.8)
        noise_db    += deg * np.random.uniform(5, 20)
        current      = power_w / 220

        # Fault labels
        fault = np.zeros(n_samples_per_device, dtype=int)
        fault_start = int(n_samples_per_device * 0.7)
        if dev["type"] == "air_conditioner":
            fault[fault_start:] = np.where(temperature[fault_start:] > 28, 2, 1)
        elif dev["type"] == "washing_machine":
            fault[fault_start:] = np.where(vibration[fault_start:] > 0.6, 3, 4)
        elif dev["type"] == "water_heater":
            fault[fault_start:] = np.where(power_w[fault_start:] > 2500, 5, 4)

        rul = np.maximum(0, (n_samples_per_device - fault_start) - np.maximum(0, t - fault_start))

        for i in range(n_samples_per_device):
            records.append({
                "timestamp":   pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i * 5),
                "device_id":   dev["id"],
                "device_type": dev["type"],
                "domain":      "smarthome",
                "vibration":   round(vibration[i], 4),
                "temperature": round(temperature[i], 2),
                "current":     round(current[i], 3),
                "pressure":    round(1.0 + 0.05 * np.random.randn(), 4),
                "rpm":         round(1200 + 100 * np.random.randn() if dev["type"] == "washing_machine" else 0, 1),
                "humidity":    round(humidity[i], 2),
                "power_w":     round(power_w[i], 2),
                "fault_label": int(fault[i]),
                "fault_name":  SMARTHOME_FAULT_LABELS[int(fault[i])],
                "rul":         int(rul[i]),
                "degradation": round(deg[i], 4),
            })

    df = pd.DataFrame(records)
    out = OUTPUT_DIR / "smarthome.csv"
    df.to_csv(out, index=False)
    print(f"[✓] Smart home dataset saved → {out}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    print("Generating datasets...")
    df_ind = generate_industrial(n_samples_per_device=5000)
    df_sh  = generate_smarthome(n_samples_per_device=3000)
    combined = pd.concat([df_ind, df_sh], ignore_index=True)
    out = OUTPUT_DIR / "combined.csv"
    combined.to_csv(out, index=False)
    print(f"[✓] Combined dataset saved → {out}  ({len(combined):,} rows)")
    print("\nLabel distribution:")
    print(combined["fault_name"].value_counts())
