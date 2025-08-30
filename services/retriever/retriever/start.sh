#!/usr/bin/env bash
set -euo pipefail

# --- wait for Qdrant ---
python - <<'PY'
import os, time, sys
import requests

q = os.getenv("QDRANT_URL", "http://qdrant:6333")
for _ in range(90):
    try:
        requests.get(q, timeout=2).raise_for_status()
        break
    except Exception:
        time.sleep(1)
else:
    print("Qdrant not reachable", file=sys.stderr)
    sys.exit(1)
PY

# --- auto-seed ---
python - <<'PY'
import os
from retriever.seed import main as seed_main

# AUTO_SEED=1 (default) (seed if missing/stale)
auto = os.getenv("AUTO_SEED", "1") == "1"
if auto:
    seed_main()
PY

# --- run API ---
exec uvicorn retriever.app:app --host 0.0.0.0 --port 8000