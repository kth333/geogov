#!/usr/bin/env bash
set -euo pipefail

# --- wait for Qdrant ---
: "${QDRANT_URL:=http://qdrant:6333}"
for i in {1..90}; do
  if curl -sf "$QDRANT_URL" >/dev/null; then break; fi
  sleep 1
done

# --- auto-seed ---
python - <<'PY'
import os
from retriever.seed import main as seed_main
if os.getenv("AUTO_SEED","1") == "1":
    seed_main()
PY

# --- run API ---
exec uvicorn retriever.app:app --host 0.0.0.0 --port 8000