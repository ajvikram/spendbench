# Copilot / Windsurf Manual A/B Protocol

Subscription tools can't be proxied or billed per token, so they're measured by hand:
**solve rate, user prompts (the billing unit), truncation events, wall-clock.** The
hypothesis being tested: AST skeletons help most where native retrieval is chunk-based
and context windows are small — i.e. they should improve *correctness and turns*, not
token cost.

## Setup (once)

1. Open the task repo (e.g. `repos/lodash`) as the VS Code workspace.
2. Pick the model in Copilot's picker — **start with a small-context model**
   (e.g. a mini-tier model); that's where the hypothesis predicts the biggest gap.
3. Conditions:
   - **baseline** — TokenSlayer extension disabled (Extensions → Disable (Workspace)).
   - **tokenslayer-ext** — extension enabled; in agent mode make sure the
     `tokenslayer-structural-summary` tool is enabled in the tools picker.
4. Keep one stopwatch (phone is fine).

## Per run (aim for N=3 per cell, fresh chat each time)

1. **New chat** (critical — no history reuse), agent mode.
2. Paste the task prompt printed by:
   `python -m spendbench.manual --task tasks/orientation/<task>.json --harness copilot-agent --model <model> --label <condition>`
   (it prints the prompt first, then waits for your results)
3. While it runs, count:
   - **user prompts** you needed (initial + any follow-ups to get a real answer)
   - **truncation events** (any "file too large", partial reads, or the model
     visibly losing earlier context)
   - whether `tokenslayer-structural-summary` visibly ran (tool call shown in chat)
4. Stop the watch when you have a final answer; paste it into the script (Ctrl-D).
   Grading is automatic (same `expected_regex` as automated runs).

## Run order (interleave conditions, don't batch)

baseline → tokenslayer-ext → baseline → tokenslayer-ext → … per task. This spreads
any workspace-index warm-up evenly across conditions.

## Which tasks

| Task | Why |
|---|---|
| `lodash-baseflatten` | 17k-line single file — the truncation stressor |
| `express-view-sendfile` | multi-file wiring |
| `tokenslayer-mcp-entry` | trivial control — both conditions should tie |

~18 runs total ≈ 60–90 min. Then: `python -m spendbench.aggregate` — manual records
show prompts/wall/solve-rate with "—" in the token/cost columns.

## What would count as a win

- Higher solve rate or fewer truncation events on lodash with the extension, especially
  on small-context models, or
- Fewer user prompts to a correct answer (that's a direct premium-request saving).

If both conditions tie at 100% solve / 1 prompt, the tasks are too easy for this
platform — escalate to a harder multi-hop question before concluding anything.
