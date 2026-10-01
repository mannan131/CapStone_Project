"""Compose healthcheck: API must return 200 with model_loaded=true."""

import sys
import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8000/health", timeout=10) as r:
        import json

        ok = r.status == 200 and json.loads(r.read().decode()).get("model_loaded") is True
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
