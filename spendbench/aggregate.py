"""Aggregate run records into a per-(task, condition) summary table.

Reports medians with IQR — same-task token usage varies up to 30x between runs
(arXiv:2604.22750), so means over small N are meaningless.

Usage:  python -m spendbench.aggregate [--records runs/records]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .pricing import cost_cache_neutral_usd


def neutral_cost(rec: dict) -> float | None:
    """Cache-neutral cost for a record, backfilled for legacy records.

    Older records (pre cache-neutral schema) stored the raw token breakdown but
    not cost_cache_neutral_usd. Recompute it from those tokens at the record's
    own model so historical runs aggregate alongside new ones rather than
    crashing the table. Manually-entered records (Copilot/Windsurf — no token
    observability) have neither and report None.
    """
    if "cost_cache_neutral_usd" in rec:
        return rec["cost_cache_neutral_usd"]
    if "input_tokens" not in rec:
        return None
    return round(cost_cache_neutral_usd(
        rec["model"],
        rec.get("input_tokens", 0),
        rec.get("output_tokens", 0),
        rec.get("cache_creation_input_tokens", 0),
        rec.get("cache_read_input_tokens", 0),
    ), 4)


def med(recs: list[dict], key, fmt: str) -> str:
    """Median of key over records where it's present; em-dash when absent."""
    values = [v for r in recs if (v := (key(r) if callable(key) else r.get(key))) is not None]
    return fmt.format(statistics.median(values)) if values else "—"


def median_iqr(values: list[float]) -> str:
    if not values:
        return "—"
    med = statistics.median(values)
    if len(values) < 3:
        return f"{med:,.3f} (n<3)"
    qs = statistics.quantiles(values, n=4)
    return f"{med:,.3f} [{qs[0]:,.3f}–{qs[2]:,.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=Path("runs/records"))
    args = ap.parse_args()

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(args.records.glob("*.json")):
        rec = json.loads(path.read_text())
        groups[(rec["task_id"], rec["label"])].append(rec)

    if not groups:
        raise SystemExit(f"no records found in {args.records}")

    header = ["task", "condition", "n", "solve_rate", "neutral_$ med [IQR]",
              "raw_$ med", "tokens med", "reqs/prompts", "wall_s med"]
    rows = []
    for (task, label), recs in sorted(groups.items()):
        solved = [r["solved"] for r in recs if r["solved"] is not None]
        neutral = [v for r in recs if (v := neutral_cost(r)) is not None]
        rows.append([
            task, label, str(len(recs)),
            f"{sum(solved)/len(solved):.0%}" if solved else "—",
            median_iqr(neutral) if neutral else "—",
            med(recs, "cost_usd", "{:.3f}"),
            med(recs, "total_tokens", "{:,.0f}"),
            med(recs, lambda r: r.get("user_prompts", r.get("api_requests")), "{:.0f}"),
            med(recs, "wall_clock_s", "{:.0f}"),
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()
