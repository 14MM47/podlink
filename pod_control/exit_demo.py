"""Phase 0 exit gate: streaming completion + perf record.

Fires a legal-flavoured prompt at the pod, streams tokens, captures:
  - TTFT (time to first token)
  - total generation time
  - tokens generated
  - tokens/sec

Appends one line to logs/perf.jsonl. Prints a human summary.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import print as rprint

import _secrets
import egress_logger

STATE_PATH = Path(__file__).resolve().parents[1] / "pod_state.json"
PERF_PATH = Path(__file__).resolve().parents[1] / "logs" / "perf.jsonl"

PROMPT = "Explain the doctrine of stare decisis in two sentences."


def main() -> None:
    if not STATE_PATH.exists():
        sys.exit(f"No pod state at {STATE_PATH}; run pod_up.py first.")
    state = json.loads(STATE_PATH.read_text())
    bearer = _secrets.bearer_token()

    url = f"{state['proxy_url']}/v1/chat/completions"
    payload = {
        "model": state["model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 200,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Authorization": f"Bearer {bearer}",
        "Content-Type": "application/json",
    }

    rprint(f"[bold cyan]POST[/] {url}")
    rprint(f"[dim]prompt:[/] {PROMPT}\n")

    t_start = time.perf_counter()
    ttft: float | None = None
    chunks = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    pieces: list[str] = []

    with egress_logger.client(timeout=120.0) as c, \
         c.stream("POST", url, json=payload, headers=headers) as r:
        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code}: {r.read().decode(errors='replace')}")

        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw if isinstance(raw, str) else raw.decode()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            # Final usage-only frame (stream_options.include_usage=True).
            # vLLM emits this with choices=[] and a populated usage block.
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if not content:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t_start
            chunks += 1
            pieces.append(content)
            print(content, end="", flush=True)

    total = time.perf_counter() - t_start
    tps = (completion_tokens / total) if (completion_tokens and total > 0) else None
    cps = chunks / total if total > 0 else 0.0
    text = "".join(pieces)

    print("\n")
    if ttft is not None:
        rprint(f"[bold]TTFT:[/]            {ttft:.3f}s")
    else:
        rprint("[red]no content received[/]")
    rprint(f"[bold]Total wall time:[/] {total:.3f}s")
    if completion_tokens is not None:
        rprint(f"[bold]Completion tokens:[/] {completion_tokens}  "
               f"(prompt {prompt_tokens})")
        rprint(f"[bold]Tokens/sec:[/]       {tps:.1f}")
    else:
        rprint("[yellow]Server did not emit a usage frame; "
               "tokens/sec unavailable.[/]")
    rprint(f"[bold]SSE chunks:[/]      {chunks} ({cps:.1f}/s)")

    PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PERF_PATH.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": state["model"],
            "gpu_type": state.get("gpu_type"),
            "proxy_url": state["proxy_url"],
            "prompt_chars": len(PROMPT),
            "ttft_s": ttft,
            "total_s": total,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_s": tps,
            "chunks": chunks,
            "chunks_per_s": cps,
            "response_chars": len(text),
        }) + "\n")
    rprint(f"\n[green]Appended perf record to[/] {PERF_PATH}")


if __name__ == "__main__":
    main()
