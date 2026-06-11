"""Isolate benchmark runs from the host machine's personal Claude Code config.

`--strict-mcp-config` handles MCP servers, but the user-level `~/.claude/CLAUDE.md`
is injected into every headless run regardless of `CLAUDE_CONFIG_DIR` or
`--setting-sources` (verified empirically 2026-06-11: a sandbox config dir loads
its own CLAUDE.md *in addition to* the user one). The only reliable isolation is
moving the file aside for the duration of a benchmark batch.

The context manager restores the file on exit (including on exceptions), and
recovers from a previous crashed run on entry.
"""

import contextlib
from pathlib import Path

USER_MD = Path.home() / ".claude" / "CLAUDE.md"
MOVED = USER_MD.with_name("CLAUDE.md.spendbench-moved")


def recover_if_needed() -> bool:
    """Restore CLAUDE.md if a previous batch crashed mid-isolation."""
    if MOVED.exists() and not USER_MD.exists():
        MOVED.rename(USER_MD)
        return True
    return False


@contextlib.contextmanager
def isolated_user_memory():
    recover_if_needed()
    moved = False
    if USER_MD.exists():
        USER_MD.rename(MOVED)
        moved = True
    try:
        yield
    finally:
        if moved and MOVED.exists() and not USER_MD.exists():
            MOVED.rename(USER_MD)
