"""Manual audit entry for any curl/browser hit during debugging.

Keeps the chain of custody honest: if you reached the pod outside the audited
client, log it here with a reason.

Usage:
    python log_manual.py <reason> [--url URL] [--method METHOD] [--status STATUS]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlparse

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "egress.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("reason", help="Why this manual call happened")
    p.add_argument("--url", default="")
    p.add_argument("--method", default="GET")
    p.add_argument("--status", type=int, default=None)
    args = p.parse_args()

    parsed = urlparse(args.url) if args.url else None
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "dst_host": parsed.hostname if parsed else None,
        "dst_port": (parsed.port if parsed and parsed.port else None),
        "scheme": parsed.scheme if parsed else None,
        "path": parsed.path if parsed else None,
        "status": args.status,
        "manual": True,
        "reason": args.reason,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Logged: {record}")


if __name__ == "__main__":
    main()
