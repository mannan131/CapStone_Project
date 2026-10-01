"""Compose healthcheck: Streamlit must serve its health endpoint."""

import sys
import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=10) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
