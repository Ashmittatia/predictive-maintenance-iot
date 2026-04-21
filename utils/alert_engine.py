"""
utils/alert_engine.py
Alert management: threshold evaluation, deduplication, email notifications.
Integrates with database.py for persistence.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional

from utils.database import insert_alert, get_active_alerts
from utils.config import (
    RUL_CRITICAL, RUL_WARNING, RUL_MODERATE,
    ANOMALY_SCORE_THRESHOLD, LOGS_DIR,
)

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "alerts.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("alert_engine")

# ── Deduplication cache (in-memory, per run) ───────────────────────
# Maps (device_id, alert_type) → last_alerted_time
_alert_cooldown: dict = {}
COOLDOWN_MINUTES = 15  # don't re-raise same alert within this window


def _on_cooldown(device_id: str, alert_type: str) -> bool:
    key  = (device_id, alert_type)
    last = _alert_cooldown.get(key)
    if last is None:
        return False
    return (datetime.utcnow() - last) < timedelta(minutes=COOLDOWN_MINUTES)


def _mark_alerted(device_id: str, alert_type: str):
    _alert_cooldown[(device_id, alert_type)] = datetime.utcnow()


# ── Core evaluation ────────────────────────────────────────────────

def evaluate_and_alert(device_id: str, prediction: dict) -> list:
    """
    Given a device ID and a full prediction dict, evaluate all thresholds
    and raise appropriate alerts. Returns list of raised alert dicts.
    """
    raised = []

    rul          = prediction.get("rul_cycles", float("inf"))
    anomaly_score= prediction.get("anomaly_score", 0.0)
    is_anomaly   = prediction.get("is_anomaly", False)
    urgency      = prediction.get("urgency", "healthy")
    fault_name   = prediction.get("fault_name", "normal")
    health_score = prediction.get("health_score", 100.0)

    # ── RUL alerts ─────────────────────────────────────────────────
    if rul <= RUL_CRITICAL:
        if not _on_cooldown(device_id, "rul_critical"):
            msg = f"CRITICAL: Device {device_id} has only {rul:.0f} cycles remaining. Immediate maintenance required."
            insert_alert(device_id, "rul_critical", "critical", msg)
            _mark_alerted(device_id, "rul_critical")
            log.critical(msg)
            raised.append({"type": "rul_critical", "severity": "critical", "message": msg})

    elif rul <= RUL_WARNING:
        if not _on_cooldown(device_id, "rul_warning"):
            msg = f"WARNING: Device {device_id} RUL is {rul:.0f} cycles. Schedule maintenance soon."
            insert_alert(device_id, "rul_warning", "warning", msg)
            _mark_alerted(device_id, "rul_warning")
            log.warning(msg)
            raised.append({"type": "rul_warning", "severity": "warning", "message": msg})

    elif rul <= RUL_MODERATE:
        if not _on_cooldown(device_id, "rul_moderate"):
            msg = f"NOTICE: Device {device_id} RUL is {rul:.0f} cycles. Plan maintenance in coming weeks."
            insert_alert(device_id, "rul_moderate", "info", msg)
            _mark_alerted(device_id, "rul_moderate")
            log.info(msg)
            raised.append({"type": "rul_moderate", "severity": "info", "message": msg})

    # ── Anomaly alerts ─────────────────────────────────────────────
    if is_anomaly and not _on_cooldown(device_id, "anomaly"):
        msg = (
            f"ANOMALY: Device {device_id} showing abnormal sensor pattern. "
            f"Score={anomaly_score:.5f} (threshold={ANOMALY_SCORE_THRESHOLD})"
        )
        insert_alert(device_id, "anomaly_detected", "high", msg)
        _mark_alerted(device_id, "anomaly")
        log.warning(msg)
        raised.append({"type": "anomaly_detected", "severity": "high", "message": msg})

    # ── Fault type alerts ──────────────────────────────────────────
    if fault_name not in ("normal", "unknown") and not _on_cooldown(device_id, f"fault_{fault_name}"):
        msg = (
            f"FAULT: Device {device_id} classified as [{fault_name.upper().replace('_',' ')}]. "
            f"Health={health_score:.1f}%"
        )
        insert_alert(device_id, f"fault_{fault_name}", "warning", msg)
        _mark_alerted(device_id, f"fault_{fault_name}")
        log.warning(msg)
        raised.append({"type": f"fault_{fault_name}", "severity": "warning", "message": msg})

    return raised


# ── Email notification (optional) ──────────────────────────────────

def send_email_alert(
    to_address:   str,
    subject:      str,
    body:         str,
    smtp_host:    str = "smtp.gmail.com",
    smtp_port:    int = 587,
    from_address: str = "",
    password:     str = "",
) -> bool:
    """
    Send an email alert via SMTP.
    Configure via environment variables:
      ALERT_EMAIL_FROM, ALERT_EMAIL_PASS, ALERT_EMAIL_TO
    Returns True on success.
    """
    import os
    from_address = from_address or os.getenv("ALERT_EMAIL_FROM", "")
    password     = password     or os.getenv("ALERT_EMAIL_PASS", "")

    if not from_address or not password:
        log.debug("Email alert skipped — ALERT_EMAIL_FROM / ALERT_EMAIL_PASS not set.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = from_address
        msg["To"]      = to_address

        html_body = f"""
        <html><body style="font-family:monospace;background:#0f1117;color:#e2e8f0;padding:20px">
          <h2 style="color:#fc8181">⚠️ PredMaint Alert</h2>
          <pre style="background:#1a1d2e;padding:15px;border-radius:8px;color:#a0aec0">{body}</pre>
          <p style="color:#4a5568;font-size:0.8em">Sent by PredMaint — IoT Predictive Maintenance System</p>
        </body></html>
        """
        msg.attach(MIMEText(body,      "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(from_address, password)
            server.sendmail(from_address, to_address, msg.as_string())

        log.info(f"Email alert sent to {to_address}: {subject}")
        return True

    except Exception as e:
        log.error(f"Email alert failed: {e}")
        return False


# ── Alert summary helper ───────────────────────────────────────────

def get_alert_summary() -> dict:
    """Returns a summary dict of currently active alerts grouped by severity."""
    alerts = get_active_alerts(limit=200)
    summary = {"critical": [], "high": [], "warning": [], "info": []}
    for a in alerts:
        sev = a.get("severity", "info")
        summary.setdefault(sev, []).append(a)
    summary["total"] = len(alerts)
    return summary


def log_alert_summary():
    s = get_alert_summary()
    log.info(
        f"Alert summary — Critical:{len(s['critical'])}  "
        f"High:{len(s['high'])}  Warning:{len(s['warning'])}  "
        f"Info:{len(s['info'])}  Total:{s['total']}"
    )


if __name__ == "__main__":
    # Quick test
    from utils.database import init_db
    init_db()

    test_pred = {
        "rul_cycles":    45.0,
        "health_score":  12.0,
        "urgency":       "critical",
        "anomaly_score": 0.085,
        "is_anomaly":    True,
        "fault_name":    "bearing_wear",
    }
    alerts_raised = evaluate_and_alert("motor_test_01", test_pred)
    print(f"Alerts raised: {len(alerts_raised)}")
    for a in alerts_raised:
        print(f"  [{a['severity'].upper()}] {a['message'][:80]}")
    log_alert_summary()
