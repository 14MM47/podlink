#!/usr/bin/env bash
# Launch podlink bound to localhost only. Open http://127.0.0.1:8765 after start.
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn app.server:app --host 127.0.0.1 --port 8765 "$@"
