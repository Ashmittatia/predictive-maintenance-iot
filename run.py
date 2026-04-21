"""
run.py
One-command project runner. Handles setup, training, and launching services.

Usage:
    python run.py setup     — generate data + train all models
    python run.py api       — start FastAPI server
    python run.py dashboard — start Streamlit dashboard
    python run.py simulate  — start sensor simulator (HTTP mode)
    python run.py all       — setup + api + dashboard + simulator (multiprocess)
    python run.py test      — run unit tests
    python run.py status    — check which models/files exist
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_cmd(cmd: list, **kwargs):
    print(f"\n▶  {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kwargs)


def cmd_status():
    from utils.config import (
        COMBINED_CSV, AE_MODEL_PATH, CLF_MODEL_PATH, RUL_MODEL_PATH,
        SCALER_PATH, DB_PATH
    )
    checks = [
        ("Dataset",           COMBINED_CSV),
        ("Scaler",            SCALER_PATH),
        ("LSTM Autoencoder",  AE_MODEL_PATH),
        ("Fault Classifier",  CLF_MODEL_PATH),
        ("RUL Estimator",     RUL_MODEL_PATH),
        ("Database",          DB_PATH),
    ]
    print("\n=== Project Status ===")
    all_ok = True
    for name, path in checks:
        exists = path.exists()
        icon   = "✅" if exists else "❌"
        size   = f"  ({path.stat().st_size / 1024:.0f} KB)" if exists else ""
        print(f"  {icon}  {name:22s}{size}")
        if not exists:
            all_ok = False
    print()
    if all_ok:
        print("All components ready. Run: python run.py all")
    else:
        print("Some components missing. Run: python run.py setup")


def cmd_setup():
    print("\n=== Step 1: Generate datasets ===")
    run_cmd([sys.executable, "data/generate_dataset.py"], cwd=ROOT, check=True)

    print("\n=== Step 2: Train all models ===")
    run_cmd([sys.executable, "models/train_all.py"], cwd=ROOT, check=True)

    print("\n=== Setup complete ===")
    cmd_status()


def cmd_api():
    import uvicorn
    import uvicorn
    print("\n▶  Starting FastAPI on http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    os.chdir(ROOT)
    run_cmd([
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ], cwd=ROOT)


def cmd_dashboard():
    print("\n▶  Starting Streamlit dashboard on http://localhost:8501")
    run_cmd([
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--theme.base", "dark",
    ], cwd=ROOT)


def cmd_simulate(api_url: str = "http://localhost:8000"):
    print(f"\n▶  Starting sensor simulator → {api_url}")
    run_cmd([
        sys.executable, "simulator/sensor_simulator.py",
        "--mode", "http",
        "--api",  api_url,
        "--interval", "2",
    ], cwd=ROOT)


def cmd_all():
    """Launch all services in parallel subprocesses."""
    import time

    procs = []

    def launch(name, cmd):
        print(f"  ▶ Launching {name}...")
        p = subprocess.Popen(cmd, cwd=ROOT)
        procs.append((name, p))
        return p

    # Check setup first
    from utils.config import AE_MODEL_PATH
    if not AE_MODEL_PATH.exists():
        print("Models not found. Running setup first...")
        cmd_setup()

    print("\n=== Launching all services ===")
    launch("FastAPI",    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])
    time.sleep(3)   # give API time to start
    launch("Streamlit",  [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                          "--server.address", "0.0.0.0", "--theme.base", "dark"])
    launch("Simulator",  [sys.executable, "simulator/sensor_simulator.py",
                          "--mode", "http", "--api", "http://localhost:8000", "--interval", "2"])

    print("\n=== All services running ===")
    print("  API:       http://localhost:8000")
    print("  API docs:  http://localhost:8000/docs")
    print("  Dashboard: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services.\n")

    try:
        for name, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for name, p in procs:
            p.terminate()
            print(f"  Stopped {name}")


def cmd_test():
    run_cmd([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], cwd=ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PredMaint runner")
    parser.add_argument("command", choices=["setup","api","dashboard","simulate","all","test","status"],
                        nargs="?", default="status")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    dispatch = {
        "setup":     cmd_setup,
        "api":       cmd_api,
        "dashboard": cmd_dashboard,
        "simulate":  lambda: cmd_simulate(args.api_url),
        "all":       cmd_all,
        "test":      cmd_test,
        "status":    cmd_status,
    }
    dispatch[args.command]()
