"""Run one task through a coding agent and produce a run record.

Currently supports orientation tasks via Claude Code headless (`claude -p`).
SWE-bench fix tasks (Docker-graded) are the next milestone.

Usage:
    python -m spendbench.run_one --task tasks/orientation/foo.json \
        --model claude-sonnet-4-6 --label baseline

Requires the recording proxy to be running (python -m spendbench.proxy).
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from .pricing import PRICES_PINNED, cost_cache_neutral_usd, cost_usd

USAGE_LOG = Path("runs/usage.jsonl")
RECORDS_DIR = Path("runs/records")


def load_task(path: Path) -> dict:
    task = json.loads(path.read_text())
    for field in ("id", "type", "repo", "prompt", "expected_regex"):
        if field not in task:
            raise ValueError(f"task {path} missing required field {field!r}")
    return task


def run_claude_code(task: dict, model: str, run_id: str, proxy_port: int, timeout_s: int) -> dict:
    """Launch Claude Code headless against the task repo, routed through the proxy."""
    repo = Path(task["repo"]).expanduser()
    if not repo.is_dir():
        raise FileNotFoundError(f"task repo not found: {repo}")

    cmd = [
        "claude", "-p", task["prompt"],
        "--model", model,
        # stream-json exposes per-turn tool_use blocks, so we can verify whether the
        # condition's MCP tools were actually invoked (condition take-up).
        "--output-format", "stream-json",
        "--verbose",
        # Ignore the host machine's personal MCP servers — conditions must be explicit.
        "--strict-mcp-config",
    ]
    if task.get("_mcp_config"):
        cmd += ["--mcp-config", task["_mcp_config"]]
    env = {
        **__import__("os").environ,
        "ANTHROPIC_BASE_URL": f"http://localhost:{proxy_port}",
        "ANTHROPIC_CUSTOM_HEADERS": f"X-Spendbench-Run: {run_id}",
    }
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout_s)
    wall_s = time.monotonic() - started

    answer = ""
    tool_calls: list[str] = []
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_calls.append(block.get("name", "?"))
        elif event.get("type") == "result":
            answer = event.get("result", "")
    return {
        "exit_code": proc.returncode,
        "wall_clock_s": round(wall_s, 1),
        "answer": answer,
        "tool_calls": tool_calls,
        "stderr_tail": proc.stderr[-2000:],
    }


def grade_orientation(task: dict, answer: str) -> bool:
    return all(re.search(pattern, answer, re.IGNORECASE) for pattern in task["expected_regex"])


def collect_usage(run_id: str, fallback_model: str) -> dict:
    """Aggregate usage for a run, pricing each request by its own model.

    A single run can hit multiple models (e.g. Claude Code spawning Haiku
    subagents), so cost must be computed per request, never from one
    run-level model.
    """
    totals = {"input_tokens": 0, "output_tokens": 0,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    requests, models = 0, set()
    cost = cost_neutral = 0.0
    if USAGE_LOG.exists():
        for line in USAGE_LOG.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("run_id") != run_id:
                continue
            requests += 1
            model = rec.get("model") or fallback_model
            if rec.get("model"):
                models.add(rec["model"])
            inp = rec.get("input_tokens") or 0
            out = rec.get("output_tokens") or 0
            cw = rec.get("cache_creation_input_tokens") or 0
            cr = rec.get("cache_read_input_tokens") or 0
            totals["input_tokens"] += inp
            totals["output_tokens"] += out
            totals["cache_creation_input_tokens"] += cw
            totals["cache_read_input_tokens"] += cr
            cost += cost_usd(model, inp, out, cw, cr)
            cost_neutral += cost_cache_neutral_usd(model, inp, out, cw, cr)
    return {**totals, "api_requests": requests, "models_seen": sorted(models),
            "cost_usd": round(cost, 4), "cost_cache_neutral_usd": round(cost_neutral, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="config label, e.g. baseline | tokenwise-mcp")
    ap.add_argument("--harness", default="claude-code", choices=["claude-code"])
    ap.add_argument("--mcp-config", default=None,
                    help="path to an MCP config JSON defining this condition's context tools")
    ap.add_argument("--proxy-port", type=int, default=8377)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    task = load_task(args.task)
    # The agent subprocess runs with cwd=task repo — make the config path absolute.
    task["_mcp_config"] = str(Path(args.mcp_config).resolve()) if args.mcp_config else None
    run_id = f"{task['id']}__{args.label}__{int(time.time())}"
    print(f"run_id: {run_id}")

    result = run_claude_code(task, args.model, run_id, args.proxy_port, args.timeout)
    usage = collect_usage(run_id, fallback_model=args.model)

    if usage["api_requests"] == 0:
        raise SystemExit(
            "No API requests recorded for this run — is the proxy running on "
            f"port {args.proxy_port}? (python -m spendbench.proxy)"
        )

    solved = grade_orientation(task, result["answer"]) if task["type"] == "orientation" else None
    record = {
        "run_id": run_id,
        "task_id": task["id"],
        "task_type": task["type"],
        "harness": args.harness,
        "model": args.model,
        "label": args.label,
        "solved": solved,
        "total_tokens": usage["input_tokens"] + usage["output_tokens"]
                        + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"],
        "prices_pinned": PRICES_PINNED,
        **usage,
        **result,
    }

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORDS_DIR / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in
                      ("solved", "total_tokens", "cost_usd", "cost_cache_neutral_usd",
                       "api_requests", "wall_clock_s", "tool_calls")},
                     indent=2))
    print(f"record: {out}")


if __name__ == "__main__":
    main()
