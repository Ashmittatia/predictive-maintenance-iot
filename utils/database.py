"""
utils/database.py
SQLite persistence layer for sensor readings, alerts, and predictions.
Uses SQLAlchemy Core (sync) for simplicity; swap to async for production.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = Path(__file__).parent.parent / "data" / "iot_maintenance.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        device_id   TEXT    NOT NULL,
        device_type TEXT,
        domain      TEXT,
        vibration   REAL,
        temperature REAL,
        current     REAL,
        pressure    REAL,
        rpm         REAL,
        humidity    REAL,
        power_w     REAL
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        device_id       TEXT    NOT NULL,
        anomaly_score   REAL,
        is_anomaly      INTEGER,
        fault_label     INTEGER,
        fault_name      TEXT,
        fault_confidence REAL,
        rul_cycles      REAL,
        health_score    REAL,
        urgency         TEXT
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        device_id   TEXT    NOT NULL,
        alert_type  TEXT    NOT NULL,
        severity    TEXT    NOT NULL,
        message     TEXT,
        resolved    INTEGER DEFAULT 0,
        resolved_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_readings_device  ON sensor_readings(device_id);
    CREATE INDEX IF NOT EXISTS idx_readings_ts      ON sensor_readings(timestamp);
    CREATE INDEX IF NOT EXISTS idx_predictions_dev  ON predictions(device_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_device    ON alerts(device_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_resolved  ON alerts(resolved);
    """)

    conn.commit()
    conn.close()
    print(f"[✓] Database initialized → {DB_PATH}")


def insert_reading(device_id: str, device_type: str, domain: str, sensors: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO sensor_readings
            (timestamp, device_id, device_type, domain, vibration, temperature, current, pressure, rpm, humidity, power_w)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        device_id, device_type, domain,
        sensors.get("vibration"), sensors.get("temperature"),
        sensors.get("current"),   sensors.get("pressure"),
        sensors.get("rpm"),       sensors.get("humidity"),
        sensors.get("power_w"),
    ))
    conn.commit()
    conn.close()


def insert_prediction(device_id: str, pred: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO predictions
            (timestamp, device_id, anomaly_score, is_anomaly, fault_label, fault_name,
             fault_confidence, rul_cycles, health_score, urgency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        device_id,
        pred.get("anomaly_score"),
        int(pred.get("is_anomaly", False)),
        pred.get("fault_label"),
        pred.get("fault_name"),
        pred.get("fault_confidence"),
        pred.get("rul_cycles"),
        pred.get("health_score"),
        pred.get("urgency"),
    ))
    conn.commit()
    conn.close()


def insert_alert(device_id: str, alert_type: str, severity: str, message: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO alerts (timestamp, device_id, alert_type, severity, message)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), device_id, alert_type, severity, message))
    conn.commit()
    conn.close()


def get_recent_readings(device_id: str, limit: int = 100) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM sensor_readings WHERE device_id = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (device_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_prediction(device_id: str) -> Optional[Dict]:
    conn = get_connection()
    row  = conn.execute("""
        SELECT * FROM predictions WHERE device_id = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (device_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_active_alerts(limit: int = 50) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM alerts WHERE resolved = 0
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_device_ids() -> List[str]:
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT device_id FROM sensor_readings").fetchall()
    conn.close()
    return [r["device_id"] for r in rows]


def resolve_alert(alert_id: int):
    conn = get_connection()
    conn.execute("""
        UPDATE alerts SET resolved = 1, resolved_at = ? WHERE id = ?
    """, (datetime.utcnow().isoformat(), alert_id))
    conn.commit()
    conn.close()


def get_predictions_history(device_id: str, limit: int = 200) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM predictions WHERE device_id = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (device_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows][::-1]   # chronological order


if __name__ == "__main__":
    init_db()
