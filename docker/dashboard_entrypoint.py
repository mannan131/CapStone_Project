"""Dashboard container entrypoint: wait for a healthy API, then serve Streamlit."""

from __future__ import annotations

import json
import os
import time
import urllib.request

API_URL = os.environ.get("API_URL", "http://api:8000")
TIMEOUT_S = int(os.environ.get("API_WAIT_TIMEOUT_S", "900"))


def _api_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=10) as r:
            body = json.loads(r.read().decode())
        return r.status == 200 and body.get("model_loaded") is True
    except Exception:
        return False


if __name__ == "__main__":
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        if _api_ready():
            print("API is healthy, starting dashboard.", flush=True)
            break
        print("Waiting for API to become healthy...", flush=True)
        time.sleep(10)
    else:
        print("WARNING: API not healthy in time; starting dashboard anyway.", flush=True)
    os.execvp(
        "streamlit",
        [
            "streamlit",
            "run",
            "src/dashboard/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
        ],
    )
