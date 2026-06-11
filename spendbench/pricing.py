"""Price table (USD per 1M tokens) and cost computation, across providers.

Prices pinned 2026-06-11 from platform.claude.com (Anthropic, unchanged since
2026-06-10) and openai.com/api/pricing (OpenAI). Every published number must
state its price date — update PRICES_PINNED whenever a rate changes.

Cache economics differ by provider and are folded into each model's Rate:
  - Anthropic: cache *reads* bill at 0.1x input; 5-min-TTL cache *writes* at 1.25x.
  - OpenAI:    cached input is discounted (GPT-5 family 90% off -> 0.1x;
               GPT-4.1 family 75% off -> 0.25x; GPT-4o family 50% off -> 0.5x)
               and there is no separate cache-write surcharge.
"""

from dataclasses import dataclass

PRICES_PINNED = "2026-06-11"


@dataclass(frozen=True)
class Rate:
    """Per-1M-token rates plus this model's cache multipliers (relative to input)."""
    input: float
    output: float
    cache_read_mult: float = 0.1   # cached / cache-read input billed at this x input
    cache_write_mult: float = 1.25  # cache-write (Anthropic 5-min TTL) at this x input


# Match is by longest model-id prefix, so specific variants (e.g. gpt-4o-mini)
# win over their family root (gpt-4o) regardless of dict order.
PRICES: dict[str, Rate] = {
    # --- Anthropic (cache read 0.1x, cache write 1.25x) ---
    "claude-opus-4": Rate(5.00, 25.00),
    "claude-sonnet-4": Rate(3.00, 15.00),
    "claude-haiku-4": Rate(1.00, 5.00),
    # --- OpenAI (no cache-write surcharge -> cache_write_mult 0.0) ---
    # GPT-5 family: 90% cached discount -> 0.1x
    "gpt-5.5": Rate(5.00, 30.00, 0.10, 0.0),
    "gpt-5.4-mini": Rate(0.75, 4.50, 0.10, 0.0),
    "gpt-5.4": Rate(2.50, 15.00, 0.10, 0.0),
    "gpt-5-mini": Rate(0.25, 2.00, 0.10, 0.0),
    "gpt-5": Rate(1.25, 10.00, 0.10, 0.0),
    # GPT-4.1 family: 75% cached discount -> 0.25x
    "gpt-4.1-nano": Rate(0.10, 0.40, 0.25, 0.0),
    "gpt-4.1-mini": Rate(0.40, 1.60, 0.25, 0.0),
    "gpt-4.1": Rate(2.00, 8.00, 0.25, 0.0),
    # GPT-4o family: 50% cached discount -> 0.5x
    "gpt-4o-mini": Rate(0.15, 0.60, 0.50, 0.0),
    "gpt-4o": Rate(2.50, 10.00, 0.50, 0.0),
    # --- Google Gemini (pinned 2026-06-11 from ai.google.dev/gemini-api/docs/pricing) ---
    # Cache reads at 10% of input; no per-token write surcharge (storage billed
    # per hour, which per-run benchmarks don't accrue meaningfully — excluded).
    # Pro-tier rates are the <=200k-prompt tier; our tasks stay under that.
    "gemini-3.5-flash": Rate(1.50, 9.00, 0.10, 0.0),
    "gemini-3.1-pro": Rate(2.00, 12.00, 0.10, 0.0),
    "gemini-3-pro": Rate(2.00, 12.00, 0.10, 0.0),
    "gemini-2.5-pro": Rate(1.25, 10.00, 0.10, 0.0),
    "gemini-2.5-flash-lite": Rate(0.10, 0.40, 0.10, 0.0),
    "gemini-2.5-flash": Rate(0.30, 2.50, 0.10, 0.0),
}


def _rate(model: str) -> Rate:
    matches = [p for p in PRICES if model.startswith(p)]
    if not matches:
        raise KeyError(f"no price pinned for model {model!r} — add it to spendbench/pricing.py")
    return PRICES[max(matches, key=len)]


def ensure_priced(model: str) -> None:
    """Raise before a run starts if we couldn't price it afterwards."""
    _rate(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_creation: int = 0, cache_read: int = 0) -> float:
    r = _rate(model)
    return (
        input_tokens * r.input
        + cache_creation * r.input * r.cache_write_mult
        + cache_read * r.input * r.cache_read_mult
        + output_tokens * r.output
    ) / 1_000_000


def cost_cache_neutral_usd(model: str, input_tokens: int, output_tokens: int,
                           cache_creation: int = 0, cache_read: int = 0) -> float:
    """Cost as if every input token were processed fresh (no cache discounts/premiums).

    Whether a run hits a warm cache depends on what ran in the previous minutes —
    pure luck from a benchmarking standpoint. This metric is the deterministic
    comparator across conditions and providers; report raw cost_usd alongside it.
    """
    r = _rate(model)
    return ((input_tokens + cache_creation + cache_read) * r.input
            + output_tokens * r.output) / 1_000_000
