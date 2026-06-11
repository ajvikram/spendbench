"""Recording reverse proxy for LLM provider APIs (Anthropic, OpenAI, Gemini).

Forwards every request verbatim (including SSE streams) and appends one JSONL
usage record per model response to SPENDBENCH_LOG. Token counts are normalized to
a single canonical schema (input/output/cache_read/cache_creation) regardless of
provider, so pricing and aggregation stay provider-agnostic.

Run attribution works two ways, so any harness can be tagged:
  - Header:  X-Spendbench-Run: <run_id>   (Claude Code via ANTHROPIC_CUSTOM_HEADERS)
  - Path:    base URL of http://host:port/__sb/<run_id>  (Aider/OpenAI clients that
             only let you override the base URL — the prefix is stripped before
             forwarding upstream)

Provider + upstream are chosen from the request path. Each upstream is
env-overridable (handy for tests / Bedrock-style gateways):
  SPENDBENCH_UPSTREAM (or _ANTHROPIC), SPENDBENCH_UPSTREAM_OPENAI,
  SPENDBENCH_UPSTREAM_GEMINI.

Usage:  python -m spendbench.proxy [--port 8377]
"""

import argparse
import json
import os
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

UP_ANTHROPIC = os.environ.get("SPENDBENCH_UPSTREAM_ANTHROPIC") \
    or os.environ.get("SPENDBENCH_UPSTREAM", "https://api.anthropic.com")
UP_OPENAI = os.environ.get("SPENDBENCH_UPSTREAM_OPENAI", "https://api.openai.com")
UP_GEMINI = os.environ.get("SPENDBENCH_UPSTREAM_GEMINI",
                           "https://generativelanguage.googleapis.com")

LOG_PATH = Path(os.environ.get("SPENDBENCH_LOG", "runs/usage.jsonl"))
RUN_HEADER = "X-Spendbench-Run"
TAG_PREFIX = "/__sb/"

_SKIP_REQ = {"host", "content-length", "accept-encoding", "connection", RUN_HEADER.lower()}
_SKIP_RESP = {"content-length", "transfer-encoding", "content-encoding", "connection"}

_CANON = ("input_tokens", "output_tokens",
          "cache_creation_input_tokens", "cache_read_input_tokens")


def _route(path: str) -> tuple[str, str]:
    """Map an upstream API path to (provider, upstream_base)."""
    if path.startswith("/v1/messages"):
        return "anthropic", UP_ANTHROPIC
    if path.startswith(("/v1/chat/completions", "/v1/completions", "/v1/responses")):
        return "openai", UP_OPENAI
    if path.startswith("/v1beta") or ":generate" in path or ":streamGenerate" in path:
        return "gemini", UP_GEMINI
    return "anthropic", UP_ANTHROPIC  # back-compat default


# --- token normalization ---------------------------------------------------

def _norm_openai(u: dict) -> dict:
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    # The Responses API names them input_tokens/output_tokens directly.
    prompt = u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
    completion = u.get("completion_tokens", u.get("output_tokens", 0)) or 0
    cached = cached or (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": completion,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,  # OpenAI has no cache-write surcharge
    }


def _norm_gemini(u: dict) -> dict:
    prompt = u.get("promptTokenCount", 0) or 0
    cached = u.get("cachedContentTokenCount", 0) or 0
    return {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": u.get("candidatesTokenCount", 0) or 0,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    }


def _norm_anthropic(u: dict) -> dict:
    return {k: u.get(k, 0) or 0 for k in _CANON}


_NORMALIZERS = {"anthropic": _norm_anthropic, "openai": _norm_openai, "gemini": _norm_gemini}


# --- streaming SSE parsers -------------------------------------------------

class _SSEParser:
    """Base: buffers SSE `data:` lines and hands decoded JSON events to on_event."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._buf = ""
        self.model: str | None = None
        self.usage: dict = {}
        self.stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk.decode("utf-8", errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                self.on_event(json.loads(data))
            except json.JSONDecodeError:
                continue

    def on_event(self, event: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class _AnthropicSSE(_SSEParser):
    """Anthropic usage is already canonical, but it arrives in pieces: input and
    cache counts on message_start, output_tokens on message_delta. Overlay only
    the keys each event actually provides so later events don't zero out earlier
    ones."""

    def _overlay(self, usage: dict | None) -> None:
        for k in _CANON:
            v = (usage or {}).get(k)
            if v is not None:
                self.usage[k] = v

    def on_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "message_start":
            msg = event.get("message", {})
            self.model = msg.get("model")
            self._overlay(msg.get("usage"))
        elif etype == "message_delta":
            self._overlay(event.get("usage"))
            self.stop_reason = (event.get("delta") or {}).get("stop_reason") or self.stop_reason


class _OpenAISSE(_SSEParser):
    def on_event(self, event: dict) -> None:
        # Chat Completions streaming: model on every chunk; usage only on the final
        # chunk when stream_options.include_usage is set (the proxy forces this).
        if not self.model and event.get("model"):
            self.model = event["model"]
        if event.get("usage"):
            self.usage = _norm_openai(event["usage"])
        for choice in event.get("choices") or []:
            if choice.get("finish_reason"):
                self.stop_reason = choice["finish_reason"]
        # Responses API streaming: usage rides on the terminal response object.
        resp = event.get("response")
        if isinstance(resp, dict):
            self.model = resp.get("model") or self.model
            if resp.get("usage"):
                self.usage = _norm_openai(resp["usage"])
            self.stop_reason = resp.get("status") or self.stop_reason


class _GeminiSSE(_SSEParser):
    def on_event(self, event: dict) -> None:
        self.model = event.get("modelVersion") or self.model
        if event.get("usageMetadata"):
            self.usage = _norm_gemini(event["usageMetadata"])
        for cand in event.get("candidates") or []:
            if cand.get("finishReason"):
                self.stop_reason = cand["finishReason"]


_SSE = {"anthropic": _AnthropicSSE, "openai": _OpenAISSE, "gemini": _GeminiSSE}


def _parse_json_body(provider: str, payload: bytes):
    """Pull (model, normalized usage, stop_reason) from a non-streaming response."""
    try:
        d = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None, None, None
    if not isinstance(d, dict):
        return None, None, None
    model = d.get("model") or d.get("modelVersion")
    raw = d.get("usage") or d.get("usageMetadata")
    usage = _NORMALIZERS[provider](raw) if raw else None
    stop = d.get("stop_reason") or d.get("status")
    if not stop:
        for choice in d.get("choices") or []:
            if choice.get("finish_reason"):
                stop = choice["finish_reason"]
                break
    return model, usage, stop


def _force_openai_usage(body: bytes) -> bytes:
    """Make streamed OpenAI requests emit a final usage chunk.

    OpenAI omits token usage from SSE responses unless the request opts in via
    stream_options.include_usage. We inject it so cost is captured regardless of
    what the harness asked for. Non-JSON / non-streaming bodies pass through."""
    try:
        d = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body
    if not isinstance(d, dict) or not d.get("stream"):
        return body
    opts = d.get("stream_options")
    if not isinstance(opts, dict):
        opts = {}
    if opts.get("include_usage"):
        return body
    opts["include_usage"] = True
    d["stream_options"] = opts
    return json.dumps(d).encode("utf-8")


def _split_tag(rel) -> tuple[str | None, str]:
    """Return (run_id_from_path, forward_path_with_query) from a relative URL."""
    path = rel.path
    if not path.startswith(TAG_PREFIX):
        return None, str(rel)
    segs = path.split("/")  # ['', '__sb', run_id, *rest]
    run_id = segs[2] if len(segs) > 2 and segs[2] else None
    forward = "/" + "/".join(segs[3:])
    if rel.query_string:
        forward += "?" + rel.query_string
    return run_id, forward


def _usage_record(run_id: str, provider: str, path: str, status: int, latency_ms: int,
                  model: str | None, usage: dict | None, stop_reason: str | None) -> dict:
    usage = usage or {}
    return {
        "ts": time.time(),
        "run_id": run_id,
        "provider": provider,
        "path": path,
        "status": status,
        "latency_ms": latency_ms,
        "model": model,
        **{k: usage.get(k) for k in _CANON},
        "stop_reason": stop_reason,
    }


async def handle(request: web.Request) -> web.StreamResponse:
    tag_run, forward = _split_tag(request.rel_url)
    run_id = request.headers.get(RUN_HEADER) or tag_run or "untagged"
    forward_path = forward.split("?", 1)[0]
    provider, upstream = _route(forward_path)

    body = await request.read()
    if provider == "openai" and request.method == "POST" \
            and forward_path.endswith(("/chat/completions", "/completions")):
        body = _force_openai_usage(body)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQ}
    started = time.monotonic()

    session: ClientSession = request.app["session"]
    async with session.request(
        request.method, upstream + forward, headers=headers, data=body
    ) as upstream_resp:
        resp_headers = {k: v for k, v in upstream_resp.headers.items()
                        if k.lower() not in _SKIP_RESP}

        if "text/event-stream" in upstream_resp.headers.get("Content-Type", ""):
            resp = web.StreamResponse(status=upstream_resp.status, headers=resp_headers)
            await resp.prepare(request)
            parser = _SSE[provider](provider)
            async for chunk in upstream_resp.content.iter_any():
                parser.feed(chunk)
                await resp.write(chunk)
            await resp.write_eof()
            _append_record(_usage_record(
                run_id, provider, forward_path, upstream_resp.status,
                int((time.monotonic() - started) * 1000),
                parser.model, parser.usage, parser.stop_reason,
            ))
            return resp

        payload = await upstream_resp.read()
        latency_ms = int((time.monotonic() - started) * 1000)
        model, usage, stop_reason = _parse_json_body(provider, payload)
        if usage is not None or upstream_resp.status >= 400:
            _append_record(_usage_record(
                run_id, provider, forward_path, upstream_resp.status, latency_ms,
                model, usage, stop_reason,
            ))
        return web.Response(status=upstream_resp.status, headers=resp_headers, body=payload)


def _append_record(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def make_app() -> web.Application:
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app["session"] = ClientSession(timeout=ClientTimeout(total=None, connect=30))
    app.router.add_route("*", "/{tail:.*}", handle)

    async def close_session(app: web.Application) -> None:
        await app["session"].close()

    app.on_cleanup.append(close_session)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("SPENDBENCH_PORT", 8377)))
    args = ap.parse_args()
    print(f"spendbench proxy: localhost:{args.port}  (log: {LOG_PATH})")
    print(f"  anthropic -> {UP_ANTHROPIC}\n  openai    -> {UP_OPENAI}\n  gemini    -> {UP_GEMINI}")
    web.run_app(make_app(), port=args.port, print=None)


if __name__ == "__main__":
    main()
