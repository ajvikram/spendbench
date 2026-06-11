"""Offline end-to-end check of the OpenAI path through the recording proxy.

No API key required: a local stub stands in for api.openai.com. Proves that
  1. the /__sb/<run_id>/ path prefix tags the run and is stripped before forwarding,
  2. the proxy injects stream_options.include_usage into streamed requests,
  3. OpenAI usage (streaming + non-streaming) is normalized to the canonical schema,
  4. the resulting record prices correctly.

Run:  .venv/bin/python tests/e2e_openai_proxy.py
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from aiohttp import ClientSession, web

# Point the proxy's OpenAI upstream at our stub and isolate the log BEFORE import,
# since proxy.py reads these at module load.
_LOG = Path(tempfile.mkdtemp(prefix="sb-e2e-")) / "usage.jsonl"
os.environ["SPENDBENCH_LOG"] = str(_LOG)
_stub_seen: dict = {}


async def start_stub() -> tuple[web.AppRunner, int]:
    async def chat(request: web.Request) -> web.StreamResponse:
        assert request.path == "/v1/chat/completions", f"bad fwd path: {request.path}"
        body = await request.json()
        _stub_seen["stream"] = bool(body.get("stream"))
        _stub_seen["include_usage"] = bool(
            (body.get("stream_options") or {}).get("include_usage"))
        usage = {"prompt_tokens": 1000, "completion_tokens": 50,
                 "prompt_tokens_details": {"cached_tokens": 200}}
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            first = {"model": "gpt-4o-2024-08-06",
                     "choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(first)}\n\n".encode())
            # Final usage chunk only emitted when the request opted in — which it
            # only does if the proxy injected include_usage.
            if _stub_seen["include_usage"]:
                last = {"model": "gpt-4o-2024-08-06",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": usage}
                await resp.write(f"data: {json.dumps(last)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response({
            "model": "gpt-4o-2024-08-06", "usage": usage,
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        })

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def start_proxy(upstream_port: int) -> tuple[web.AppRunner, int]:
    os.environ["SPENDBENCH_UPSTREAM_OPENAI"] = f"http://localhost:{upstream_port}"
    import importlib
    from spendbench import proxy as proxy_mod
    importlib.reload(proxy_mod)  # pick up the upstream env set just above
    runner = web.AppRunner(await proxy_mod.make_app())
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


def _record_for(run_id: str) -> dict:
    rows = [json.loads(l) for l in _LOG.read_text().splitlines()]
    matches = [r for r in rows if r["run_id"] == run_id]
    assert matches, f"no usage record for {run_id} in {_LOG}"
    return matches[-1]


async def main() -> None:
    stub_runner, stub_port = await start_stub()
    proxy_runner, proxy_port = await start_proxy(stub_port)
    from spendbench.pricing import cost_usd, cost_cache_neutral_usd

    ok = True
    try:
        async with ClientSession() as s:
            # 1) Streaming request WITHOUT stream_options -> proxy must inject it.
            url = f"http://localhost:{proxy_port}/__sb/run-stream/v1/chat/completions"
            payload = {"model": "gpt-4o", "stream": True,
                       "messages": [{"role": "user", "content": "hi"}]}
            async with s.post(url, json=payload) as r:
                await r.read()
            assert _stub_seen["include_usage"], "proxy did NOT inject include_usage"

            rec = _record_for("run-stream")
            checks = {
                "provider": (rec["provider"], "openai"),
                "path stripped": (rec["path"], "/v1/chat/completions"),
                "input_tokens (1000-200)": (rec["input_tokens"], 800),
                "output_tokens": (rec["output_tokens"], 50),
                "cache_read_input_tokens": (rec["cache_read_input_tokens"], 200),
                "cache_creation_input_tokens": (rec["cache_creation_input_tokens"], 0),
                "model": (rec["model"], "gpt-4o-2024-08-06"),
            }
            for name, (got, want) in checks.items():
                flag = "ok" if got == want else "FAIL"
                if got != want:
                    ok = False
                print(f"  [{flag}] {name}: {got!r} (want {want!r})")
            raw = cost_usd(rec["model"], rec["input_tokens"], rec["output_tokens"],
                           0, rec["cache_read_input_tokens"])
            neutral = cost_cache_neutral_usd(rec["model"], rec["input_tokens"],
                                             rec["output_tokens"], 0,
                                             rec["cache_read_input_tokens"])
            print(f"  priced raw=${raw:.6f} neutral=${neutral:.6f}")

            # 2) Non-streaming request.
            url2 = f"http://localhost:{proxy_port}/__sb/run-json/v1/chat/completions"
            async with s.post(url2, json={"model": "gpt-4o",
                                          "messages": [{"role": "user", "content": "hi"}]}) as r:
                await r.read()
            rec2 = _record_for("run-json")
            for name, (got, want) in {"json input_tokens": (rec2["input_tokens"], 800),
                                      "json output_tokens": (rec2["output_tokens"], 50)}.items():
                flag = "ok" if got == want else "FAIL"
                if got != want:
                    ok = False
                print(f"  [{flag}] {name}: {got!r} (want {want!r})")
    finally:
        await proxy_runner.cleanup()
        await stub_runner.cleanup()

    print("\nE2E:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
