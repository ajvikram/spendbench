"""Recording reverse proxy for api.anthropic.com.

Forwards every request verbatim (including SSE streams) and appends one JSONL
usage record per model response to SPENDBENCH_LOG. Runs are attributed via the
X-Spendbench-Run request header, which the runner injects through Claude Code's
ANTHROPIC_CUSTOM_HEADERS.

Usage:  python -m spendbench.proxy [--port 8377]
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM = os.environ.get("SPENDBENCH_UPSTREAM", "https://api.anthropic.com")
LOG_PATH = Path(os.environ.get("SPENDBENCH_LOG", "runs/usage.jsonl"))
RUN_HEADER = "X-Spendbench-Run"

# Headers that must not be forwarded verbatim in either direction.
_SKIP_REQ = {"host", "content-length", "accept-encoding", "connection", RUN_HEADER.lower()}
_SKIP_RESP = {"content-length", "transfer-encoding", "content-encoding", "connection"}


def _append_record(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _usage_record(run_id: str, path: str, status: int, latency_ms: int,
                  model: str | None, usage: dict | None, stop_reason: str | None) -> dict:
    usage = usage or {}
    return {
        "ts": time.time(),
        "run_id": run_id,
        "path": path,
        "status": status,
        "latency_ms": latency_ms,
        "model": model,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "stop_reason": stop_reason,
    }


class SSEUsageParser:
    """Incrementally extracts model/usage/stop_reason from an Anthropic SSE stream.

    input_tokens arrive on message_start; output_tokens and stop_reason on
    message_delta (the final one wins).
    """

    def __init__(self) -> None:
        self._buf = ""
        self.model: str | None = None
        self.usage: dict = {}
        self.stop_reason: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk.decode("utf-8", errors="replace")
        # SSE events are newline-delimited; keep the trailing partial line buffered.
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "message_start":
                message = event.get("message", {})
                self.model = message.get("model")
                self.usage.update(message.get("usage") or {})
            elif etype == "message_delta":
                self.usage.update(event.get("usage") or {})
                self.stop_reason = (event.get("delta") or {}).get("stop_reason") or self.stop_reason


async def handle(request: web.Request) -> web.StreamResponse:
    run_id = request.headers.get(RUN_HEADER, "untagged")
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQ}
    started = time.monotonic()

    session: ClientSession = request.app["session"]
    async with session.request(
        request.method, UPSTREAM + str(request.rel_url), headers=headers, data=body
    ) as upstream:
        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _SKIP_RESP}

        if "text/event-stream" in upstream.headers.get("Content-Type", ""):
            resp = web.StreamResponse(status=upstream.status, headers=resp_headers)
            await resp.prepare(request)
            parser = SSEUsageParser()
            async for chunk in upstream.content.iter_any():
                parser.feed(chunk)
                await resp.write(chunk)
            await resp.write_eof()
            _append_record(_usage_record(
                run_id, request.path, upstream.status,
                int((time.monotonic() - started) * 1000),
                parser.model, parser.usage, parser.stop_reason,
            ))
            return resp

        payload = await upstream.read()
        latency_ms = int((time.monotonic() - started) * 1000)
        model = usage = stop_reason = None
        try:
            parsed = json.loads(payload)
            model = parsed.get("model")
            usage = parsed.get("usage")
            stop_reason = parsed.get("stop_reason")
        except (json.JSONDecodeError, AttributeError):
            pass
        if usage is not None or upstream.status >= 400:
            _append_record(_usage_record(
                run_id, request.path, upstream.status, latency_ms, model, usage, stop_reason,
            ))
        return web.Response(status=upstream.status, headers=resp_headers, body=payload)


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
    print(f"spendbench proxy: localhost:{args.port} -> {UPSTREAM}  (log: {LOG_PATH})")
    web.run_app(make_app(), port=args.port, print=None)


if __name__ == "__main__":
    main()
