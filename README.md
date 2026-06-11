# SpendBench

**How much does it cost a coding agent to actually solve a task?**

SWE-bench tells you *whether* an agent solves a task. SpendBench tells you **what it costs** —
tokens and dollars per solved task — across harnesses (Claude Code, Aider, …), models, and
context strategies (e.g. structural compaction MCPs on/off).

> Status: Week 1 — measurement harness. Single-task smoke runs work; SWE-bench Docker matrix is next.

## Why this exists

- Token cost is the #1 complaint of AI-coding users, and input tokens (file reads, orientation)
  dominate it — agentic tasks consume ~1000× more tokens than plain code chat
  ([Wei et al. 2026](https://arxiv.org/abs/2604.22750)).
- No public, reproducible leaderboard measures **$/solved-task** across harnesses and context
  strategies. [ContextBench](https://arxiv.org/abs/2602.05892) measures context retrieval
  recall/precision; [Artificial Analysis](https://artificialanalysis.ai/agents/coding-agents)
  compares agents but not context strategies. SpendBench measures the economics.

## Methodology (the part that has to be bulletproof)

1. **Token counts are captured at the API boundary, not from harness self-reports.**
   A local recording proxy (`spendbench/proxy.py`) sits between the agent and
   `api.anthropic.com`, forwards everything verbatim (including SSE streams), and logs
   per-request usage (`input_tokens`, `output_tokens`, cache creation/read) tagged with a run ID.
2. **Runs are tagged via headers.** The runner launches each agent with
   `ANTHROPIC_BASE_URL` pointed at the proxy and `ANTHROPIC_CUSTOM_HEADERS` carrying
   `X-Spendbench-Run: <run_id>`. Every API call lands in `runs/usage.jsonl` with that tag.
3. **Variance is first-class.** Token usage on the same task can vary up to 30× between runs
   ([Wei et al. 2026](https://arxiv.org/abs/2604.22750)). We report **median ± IQR over N≥3
   runs per cell** and publish every raw transcript.
4. **Headline metric: $/solved-task** (price-weighted, cache reads at ~0.1× input price),
   presented as a Pareto chart of solve-rate vs. tokens. No composite score.

## Task classes

- **Fix tasks** — SWE-bench Verified subset, graded by the official test harness (Docker). *(WIP)*
- **Orientation tasks** — "where is X wired / what does Y expose" questions about real repos
  with regex-verifiable answers (`tasks/orientation/`). This is the phase where context
  strategy matters most and no existing benchmark covers it.

## Quickstart (smoke run)

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

# Terminal 1 — start the recording proxy (default port 8377)
.venv/bin/python -m spendbench.proxy

# Terminal 2 — run one orientation task through Claude Code
.venv/bin/python -m spendbench.run_one \
  --task tasks/orientation/tokenslayer-mcp-entry.json \
  --model claude-sonnet-4-6 --label baseline
```

Results land in `runs/records/<run_id>.json`; raw per-request usage in `runs/usage.jsonl`.

## Conflict of interest

SpendBench's author also builds [TokenSlayer](https://github.com/ajvikram/TokenSlayer) /
tokenwise, which appear on the leaderboard as one context-strategy row among several. All
runners, prompts, transcripts, and grading code are public; PRs adding tools/harnesses welcome.

## Roadmap

- [x] Recording proxy with SSE usage parsing
- [x] Single-task runner (Claude Code headless) + orientation grading
- [ ] SWE-bench Verified subset via official Docker harness
- [ ] Second harness (Aider) through the same proxy
- [ ] N=3 matrix runner + medians/IQR aggregation
- [ ] Static leaderboard site + launch writeup
