"""Run a full benchmark matrix: tasks x conditions x N repeats.

Usage:
    python -m spendbench.run_matrix --config configs/matrix-orientation.json

Manages the recording proxy itself (spawns one if the port is free), moves the
user's ~/.claude/CLAUDE.md aside for the duration of the batch (see isolation.py),
and interleaves conditions across repeats so cache warmth doesn't systematically
favor whichever condition runs second.
"""

import argparse
import contextlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from .isolation import isolated_user_memory
from .run_one import HARNESSES, run_single


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("localhost", port)) == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--proxy-port", type=int, default=8377)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    tasks: list[str] = cfg["tasks"]
    conditions: dict[str, str | None] = cfg["conditions"]
    model: str = cfg["model"]
    n: int = cfg.get("n", 3)
    timeout: int = cfg.get("timeout", 900)
    harness: str = cfg.get("harness", "claude-code")
    if harness not in HARNESSES:
        raise SystemExit(f"unknown harness {harness!r}; choose from {sorted(HARNESSES)}")

    proxy = None
    if not _port_open(args.proxy_port):
        proxy = subprocess.Popen(
            [sys.executable, "-m", "spendbench.proxy", "--port", str(args.proxy_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

    total = len(tasks) * len(conditions) * n
    done = failed = 0
    started = time.monotonic()
    # The user-global CLAUDE.md only contaminates Claude Code; moving it aside for
    # an Aider/other batch would be pointless (and a needless touch of $HOME).
    isolation = isolated_user_memory() if harness == "claude-code" else contextlib.nullcontext()
    try:
        with isolation:
            # Repeat-major, condition-inner ordering: each repeat cycles through all
            # conditions back-to-back, so every condition experiences a similar mix
            # of warm/cold cache states across the batch.
            for rep in range(n):
                for task in tasks:
                    for label, mcp_config in conditions.items():
                        done += 1
                        tag = f"[{done}/{total}] rep{rep} {Path(task).stem} {label}"
                        try:
                            rec = run_single(task, model, label, mcp_config,
                                             args.proxy_port, timeout, harness)
                            print(f"{tag}: solved={rec['solved']} "
                                  f"neutral=${rec['cost_cache_neutral_usd']} "
                                  f"tokens={rec['total_tokens']:,} "
                                  f"wall={rec['wall_clock_s']}s", flush=True)
                        except Exception as exc:  # keep the batch going
                            failed += 1
                            print(f"{tag}: FAILED — {exc}", flush=True)
    finally:
        if proxy:
            proxy.terminate()

    mins = (time.monotonic() - started) / 60
    print(f"\nbatch complete: {done - failed}/{total} ok, {failed} failed, {mins:.1f} min")
    print("aggregate with: python -m spendbench.aggregate")


if __name__ == "__main__":
    main()
