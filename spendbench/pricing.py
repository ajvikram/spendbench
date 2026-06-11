"""Price table (USD per 1M tokens) and cost computation.

Prices pinned 2026-06-10 from platform.claude.com. Cache reads bill at 0.1x the
input price; 5-minute-TTL cache writes at 1.25x. Update PRICES (and the pin date)
whenever pricing changes — every published number must state its price date.
"""

PRICES_PINNED = "2026-06-10"

# model-id prefix -> (input, output) USD per 1M tokens
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4": (5.00, 25.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (1.00, 5.00),
}

CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25


def _rates(model: str) -> tuple[float, float]:
    for prefix, rates in PRICES.items():
        if model.startswith(prefix):
            return rates
    raise KeyError(f"no price pinned for model {model!r} — add it to spendbench/pricing.py")


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_creation: int = 0, cache_read: int = 0) -> float:
    inp, out = _rates(model)
    return (
        input_tokens * inp
        + cache_creation * inp * CACHE_WRITE_MULT
        + cache_read * inp * CACHE_READ_MULT
        + output_tokens * out
    ) / 1_000_000


def cost_cache_neutral_usd(model: str, input_tokens: int, output_tokens: int,
                           cache_creation: int = 0, cache_read: int = 0) -> float:
    """Cost as if every input token were processed fresh (no cache discounts/premiums).

    Whether a run hits a warm cache depends on what ran in the previous 5 minutes —
    pure luck from a benchmarking standpoint. This metric is the deterministic
    comparator across conditions; report raw cost_usd alongside it.
    """
    inp, out = _rates(model)
    return ((input_tokens + cache_creation + cache_read) * inp + output_tokens * out) / 1_000_000
