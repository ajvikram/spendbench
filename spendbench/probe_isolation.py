"""Empirically test which Claude Code flags isolate a run from the user-global
``~/.claude/CLAUDE.md``.

Why this exists: benchmark conditions must be explicit. The host's user-level
CLAUDE.md currently instructs agents to use the ``tokenwise`` MCP for any code
orientation — if that file leaks into a run, it contaminates *every* condition
(baseline included) and invalidates the comparison. ``isolation.py`` works around
this by physically moving the file aside for the duration of a batch; that hack is
worth dropping the moment a flag does the job. Claude Code's behaviour here has
changed across versions (``--bare`` and per-source ``--setting-sources`` are
recent), so this probe is meant to be re-run whenever the CLI is upgraded.

Method: in a throwaway temp dir (no project/local CLAUDE.md of its own), ask the
agent itself whether its loaded memory mentions ``tokenwise``. The only place that
word can come from is the user-global file, so a YES means that flag combination is
contaminated and a NO means it is clean.

Run (after the API limit resets):
    python -m spendbench.probe_isolation
    python -m spendbench.probe_isolation --model claude-sonnet-4-6 --timeout 90
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

USER_MD = Path.home() / ".claude" / "CLAUDE.md"

# A word that appears in the user-global CLAUDE.md but nowhere a clean run would
# legitimately see it. Keep this in sync with whatever the host's CLAUDE.md says.
SENTINEL = "tokenwise"

PROBE = (
    "Do NOT use any tools. Answer only from instructions/memory already loaded "
    f"into this session. Question: do your loaded instructions mention a tool or "
    f"MCP server named '{SENTINEL}'? Reply with exactly one word — YES or NO — as "
    "the very first word of your response, then optionally one short sentence."
)

# (label, extra CLI args). "default" is the contamination control: if it does not
# come back LEAK, the probe itself is broken (or the sentinel word is stale).
CONDITIONS: list[tuple[str, list[str]]] = [
    ("default (control)", []),
    ("setting-sources=project,local", ["--setting-sources", "project,local"]),
    ("setting-sources=local", ["--setting-sources", "local"]),
    ("bare", ["--bare"]),
]

LIMIT_RE = re.compile(r"hit your limit|rate.?limit|usage limit", re.IGNORECASE)
AUTH_RE = re.compile(r"auth|api[_ ]?key|credential|login|unauthor", re.IGNORECASE)


def run_condition(label: str, extra: list[str], cwd: Path, model: str,
                  timeout: int) -> dict:
    cmd = ["claude", "-p", PROBE, "--strict-mcp-config", "--model", model, *extra]
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"label": label, "verdict": "TIMEOUT", "answer": ""}
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    blob = f"{out}\n{err}".strip()

    if LIMIT_RE.search(blob):
        verdict = "LIMIT"
    elif proc.returncode != 0 and AUTH_RE.search(blob):
        verdict = "AUTH-FAIL"
    elif proc.returncode != 0:
        verdict = f"ERR({proc.returncode})"
    else:
        first = (out.split() or [""])[0].strip(".,:!").upper()
        if first == "YES":
            verdict = "LEAK"
        elif first == "NO":
            verdict = "CLEAN"
        elif SENTINEL.lower() in out.lower():
            verdict = "LEAK?"  # mentioned the sentinel without a clean YES/NO
        else:
            verdict = "UNKNOWN"
    return {"label": label, "verdict": verdict, "answer": out[:200] or err[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    if not USER_MD.exists():
        raise SystemExit(
            f"user CLAUDE.md not found at {USER_MD} — nothing to leak, so this "
            "probe can't distinguish clean from contaminated. (Is a batch mid-run "
            "with the file moved aside? See isolation.py.)"
        )
    if SENTINEL.lower() not in USER_MD.read_text().lower():
        raise SystemExit(
            f"sentinel {SENTINEL!r} not present in {USER_MD}; update SENTINEL in "
            "this file to a word your user CLAUDE.md actually contains."
        )

    print(f"probing isolation flags (model={args.model}, sentinel={SENTINEL!r})")
    print("LEAK = user CLAUDE.md reached the run; CLEAN = isolated\n")

    results = []
    with tempfile.TemporaryDirectory(prefix="spendbench-probe-") as tmp:
        cwd = Path(tmp)
        for label, extra in CONDITIONS:
            res = run_condition(label, extra, cwd, args.model, args.timeout)
            results.append(res)
            print(f"  {res['verdict']:<10} {label}")
            if res["answer"]:
                print(f"             ↳ {res['answer']}")

    control = next((r for r in results if r["label"].startswith("default")), None)
    if control and control["verdict"] != "LEAK":
        print(f"\n⚠️  control condition came back {control['verdict']}, not LEAK — "
              "results are not trustworthy (limit/auth/stale sentinel?).")
        return

    clean = [r["label"] for r in results
             if r["verdict"] == "CLEAN" and not r["label"].startswith("default")]
    print("\n" + ("clean isolation via: " + ", ".join(clean) if clean
                  else "no flag isolated the user CLAUDE.md — keep the "
                       "move-aside hack in isolation.py."))


if __name__ == "__main__":
    main()
