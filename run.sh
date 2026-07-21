#!/usr/bin/env bash
# Launch podlink bound to localhost only. Open http://127.0.0.1:8765 after start.
# Host/port are fixed here on purpose: no arg pass-through, so a caller cannot
# append --host 0.0.0.0 to defeat the localhost bind (last value would win).
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn app.server:app --host 127.0.0.1 --port 8765
