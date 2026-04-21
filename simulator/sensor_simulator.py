"""
simulator/sensor_simulator.py
Simulates IoT sensor telemetry for all devices.
Publishes readings via MQTT (if broker available) or directly to API via HTTP.

Usage:
    python simulator/sensor_simulator.py --mode http --api http://localhost:8000
    python simulator/sensor_simulator.py --mode mqtt --broker localhost
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import json
import time
import random
import math
import threading
from datetime import datetime


# ── Device profiles ───────────────────────────────────────────────

DEVICES = [
    # Industrial
    {"id": "motor_01",          "type": "motor",           "domain": "industrial", "health": 1.0},
    {"id": "motor_02",          "type": "motor",           "domain": "industrial", "health": 0.85},
    {"id": "pump_01",           "type": "pump",            "domain": "industrial", "health": 0.7},
    {"id": "pump_02",           "type": "pump",            "domain": "industrial", "health": 0.95},
    {"id": "compressor_01",     "type": "compressor",      "domain": "industrial", "health": 0.5},
    # Smart Home
    {"id": "ac_01",             "type": "air_conditioner", "domain": "smarthome",  "health": 0.9},
    {"id": "ac_02",             "type": "air_conditioner", "domain": "smarthome",  "health": 0.4},
    {"id": "washing_machine_01","type": "washing_machine", "domain": "smarthome",  "health": 0.8},
    {"id": "washing_machine_02","type": "washing_machine", "domain": "smarthome",  "health": 0.6},
    {"id": "water_heater_01",   "type": "water_heater",    "domain": "smarthome",  "health": 0.75},
]


def _degrade(health: float) -> float:
    """Randomly degrade health slightly each call."""
    return max(0.05, health - random.uniform(0, 0.0005))


def generate_reading(device: dict, t: float) -> dict:
    """Generate sensor values based on device type and health level."""
    h   = device["health"]
    deg = 1.0 - h              # degradation factor 0 → 1
    noise = lambda s: s * random.gauss(0, 0.02)

    if device["domain"] == "industrial":
        vibration   = 0.5 + 0.3 * math.sin(2 * math.pi * t / 50) + deg * 2.5 + noise(0.5)
        temperature = 65  + 5   * math.sin(2 * math.pi * t / 200) + deg * 20  + noise(2)
        current     = 10  + 1   * math.sin(2 * math.pi * t / 100) + deg * 4   + noise(0.3)
        pressure    = 1.0 + 0.1 * math.sin(2 * math.pi * t / 80)  - deg * 0.3 + noise(0.02)
        rpm         = 1500 - deg * 200 + noise(20)
        humidity    = 50  + noise(5)
        power_w     = current * 220 * 0.85
    else:
        vibration   = 0.1 + deg * 0.6  + noise(0.05)
        temperature = 22  + deg * 10   + noise(1)
        current     = (1200 + deg * 400) / 220 + noise(0.2)
        pressure    = 1.0 + noise(0.05)
        rpm         = (1200 + noise(50)) if device["type"] == "washing_machine" else 0.0
        humidity    = 55 + noise(5)
        power_w     = 1200 + deg * 400 + noise(50)

    return {
        "device_id":   device["id"],
        "device_type": device["type"],
        "domain":      device["domain"],
        "vibration":   round(max(0, vibration), 4),
        "temperature": round(temperature, 2),
        "current":     round(max(0, current), 3),
        "pressure":    round(max(0, pressure), 4),
        "rpm":         round(max(0, rpm), 1),
        "humidity":    round(max(0, min(100, humidity)), 2),
        "power_w":     round(max(0, power_w), 2),
        "timestamp":   datetime.utcnow().isoformat(),
    }


# ── HTTP publisher ─────────────────────────────────────────────────

def publish_http(reading: dict, api_url: str):
    try:
        import requests
        resp = requests.post(f"{api_url}/predict/full", json=reading, timeout=5)
        if resp.status_code == 200:
            r = resp.json()
            print(
                f"[{reading['device_id']:24s}] "
                f"health={reading.get('power_w',0)/3000*100:.0f}%  "
                f"anomaly={r.get('is_anomaly',False)}  "
                f"fault={r.get('fault_name','?'):20s}  "
                f"RUL={r.get('rul_cycles',0):.0f}  "
                f"urgency={r.get('urgency','?')}"
            )
        else:
            print(f"[{reading['device_id']}] API error {resp.status_code}: {resp.text[:80]}")
    except Exception as e:
        print(f"[{reading['device_id']}] HTTP error: {e}")


# ── MQTT publisher ─────────────────────────────────────────────────

def publish_mqtt(reading: dict, client):
    topic = f"iot/sensors/{reading['domain']}/{reading['device_id']}"
    client.publish(topic, json.dumps(reading))
    print(f"[MQTT] Published → {topic}")


def run_mqtt_simulator(broker: str = "localhost", port: int = 1883, interval: float = 2.0):
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect(broker, port, keepalive=60)
        client.loop_start()
        print(f"[MQTT] Connected to {broker}:{port}")
        t = 0
        while True:
            for device in DEVICES:
                device["health"] = _degrade(device["health"])
                reading = generate_reading(device, t)
                publish_mqtt(reading, client)
            t += 1
            time.sleep(interval)
    except Exception as e:
        print(f"[MQTT] Error: {e}. Falling back to console output.")
        run_console_simulator()


def run_http_simulator(api_url: str = "http://localhost:8000", interval: float = 2.0):
    t = 0
    print(f"[HTTP] Sending to {api_url}  (interval={interval}s)")
    while True:
        for device in DEVICES:
            device["health"] = _degrade(device["health"])
            reading = generate_reading(device, t)
            threading.Thread(target=publish_http, args=(reading, api_url), daemon=True).start()
        t += 1
        time.sleep(interval)


def run_console_simulator(interval: float = 1.0):
    """Dry run — just print readings to stdout."""
    t = 0
    print("[Console] Simulating sensor readings...")
    while True:
        for device in DEVICES:
            device["health"] = _degrade(device["health"])
            reading = generate_reading(device, t)
            print(json.dumps(reading))
        t += 1
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Sensor Simulator")
    parser.add_argument("--mode",    choices=["http","mqtt","console"], default="http")
    parser.add_argument("--api",     default="http://localhost:8000")
    parser.add_argument("--broker",  default="localhost")
    parser.add_argument("--port",    type=int, default=1883)
    parser.add_argument("--interval",type=float, default=2.0)
    args = parser.parse_args()

    if args.mode == "http":
        run_http_simulator(args.api, args.interval)
    elif args.mode == "mqtt":
        run_mqtt_simulator(args.broker, args.port, args.interval)
    else:
        run_console_simulator(args.interval)
