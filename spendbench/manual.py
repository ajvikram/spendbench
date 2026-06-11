"""Record a manually-run benchmark session (Copilot, Windsurf, ...) as a run record.

Subscription tools (VS Code Copilot, Windsurf Cascade) can't be routed through the
recording proxy and don't bill per token, so runs are executed by hand and entered
here. The script grades the pasted final answer with the same expected_regex as
automated runs, and writes a record that aggregates alongside them.

Per-prompt-billed platforms are measured in USER PROMPTS (the billing unit:
Copilot premium requests / Windsurf credits), wall-clock, and solve rate —
not tokens, which are unobservable there.

Usage:
    python -m spendbench.manual --task tasks/orientation/lodash-baseflatten.json \
        --harness copilot-agent --model gpt-4o --label tokenslayer-ext
Then follow the prompts (paste the assistant's final answer, Ctrl-D to end).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .run_one import RECORDS_DIR, extract_answer, grade_orientation, load_task

MANUAL_HARNESSES = ("copilot-agent", "copilot-chat", "windsurf-cascade", "other")


def ask(prompt: str, cast=str, default=None):
    raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
    if not raw and default is not None:
        return default
    return cast(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--harness", required=True, choices=MANUAL_HARNESSES)
    ap.add_argument("--model", required=True, help="model as selected in the tool's picker")
    ap.add_argument("--label", required=True,
                    help="condition label, e.g. baseline | tokenslayer-ext")
    args = ap.parse_args()

    task = load_task(args.task)
    print(f"\nTask prompt to paste into {args.harness}:\n{'-' * 60}")
    print(task["prompt"])
    print("-" * 60)

    user_prompts = ask("user prompts needed (the billing unit)", int, 1)
    wall_clock_s = ask("wall clock seconds (rough)", float)
    truncation = ask("context truncation / 'file too large' events seen? (0/1/2...)", int, 0)
    tool_visibly_used = ask("did the condition's tool visibly run? (y/n/na)", str, "na")
    print("Paste the assistant's FINAL answer, then Ctrl-D:")
    pasted = sys.stdin.read()

    answer, marker_found = extract_answer(pasted)
    solved = grade_orientation(task, answer)
    run_id = f"{task['id']}__{args.label}__{int(time.time())}"

    record = {
        "run_id": run_id,
        "task_id": task["id"],
        "task_type": task["type"],
        "harness": args.harness,
        "model": args.model,
        "label": args.label,
        "solved": solved,
        "entry": "manual",
        "user_prompts": user_prompts,
        "wall_clock_s": wall_clock_s,
        "truncation_events": truncation,
        "tool_visibly_used": tool_visibly_used,
        "answer": answer,
        "answer_marker_found": marker_found,
    }
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    (RECORDS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2))
    print(f"\nsolved={solved}  prompts={user_prompts}  -> runs/records/{run_id}.json")


if __name__ == "__main__":
    main()
