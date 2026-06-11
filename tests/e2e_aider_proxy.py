"""Offline end-to-end check of the *real* Aider harness through the proxy.

No API key required: a stub stands in for api.openai.com. This proves the part
unit tests can't — that Aider/litellm preserves the proxy's /__sb/<run_id>/v1
base-URL prefix when it builds the request, so the run is actually tagged and its
tokens land in usage.jsonl.

  stub (thread)  <-  proxy (subprocess)  <-  aider (subprocess via run_aider)

Run:  .venv/bin/python tests/e2e_aider_proxy.py
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from aiohttp import web

REPO = Path(__file__).resolve().parents[1]
LOG = Path(tempfile.mkdtemp(prefix="sb-aider-")) / "usage.jsonl"
_stub_paths: list[str] = []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


STUB_PORT, PROXY_PORT = free_port(), free_port()


def run_stub() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def chat(request: web.Request) -> web.StreamResponse:
        _stub_paths.append(request.path)
        body = await request.json()
        usage = {"prompt_tokens": 900, "completion_tokens": 12,
                 "prompt_tokens_details": {"cached_tokens": 0}}
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for piece in ({"role": "assistant"}, {"content": "BANANA"}):
                chunk = {"model": "gpt-4o-2024-08-06",
                         "choices": [{"index": 0, "delta": piece, "finish_reason": None}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            stop = {"model": "gpt-4o-2024-08-06",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            await resp.write(f"data: {json.dumps(stop)}\n\n".encode())
            if (body.get("stream_options") or {}).get("include_usage"):
                final = {"model": "gpt-4o-2024-08-06", "choices": [], "usage": usage}
                await resp.write(f"data: {json.dumps(final)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response({
            "model": "gpt-4o-2024-08-06", "usage": usage,
            "choices": [{"message": {"role": "assistant", "content": "BANANA"},
                         "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/{tail:.*}", chat)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "localhost", STUB_PORT).start())
    loop.run_forever()


def wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("localhost", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"port {port} never opened")


def main() -> None:
    threading.Thread(target=run_stub, daemon=True).start()
    wait_port(STUB_PORT)

    proxy_env = {**os.environ,
                 "SPENDBENCH_UPSTREAM_OPENAI": f"http://localhost:{STUB_PORT}",
                 "SPENDBENCH_LOG": str(LOG)}
    proxy = subprocess.Popen(
        [sys.executable, "-m", "spendbench.proxy", "--port", str(PROXY_PORT)],
        cwd=REPO, env=proxy_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_port(PROXY_PORT)

        # Make `aider` resolvable for the harness subprocess + use a dummy key.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        os.environ["OPENAI_API_KEY"] = "sk-dummy-offline-stub"

        with tempfile.TemporaryDirectory(prefix="sb-aider-repo-") as repo:
            sys.path.insert(0, str(REPO))
            from spendbench.run_one import run_aider
            task = {"repo": repo,
                    "prompt": "Reply with exactly one word: BANANA. Do not edit any files."}
            res = run_aider(task, "gpt-4o", "run-aider", PROXY_PORT, timeout_s=120)
    finally:
        proxy.terminate()

    print("aider exit_code:", res["exit_code"], "wall:", res["wall_clock_s"], "s")
    print("stub saw paths:", _stub_paths)

    ok = True
    if not all(p == "/v1/chat/completions" for p in _stub_paths) or not _stub_paths:
        ok = False
        print("  [FAIL] prefix not stripped to /v1/chat/completions")
    else:
        print("  [ok] proxy stripped /__sb/<run_id> prefix before forwarding")

    rows = [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []
    tagged = [r for r in rows if r["run_id"] == "run-aider"]
    if not tagged:
        ok = False
        print("  [FAIL] no usage.jsonl record tagged run-aider (litellm dropped the prefix?)")
    else:
        rec = tagged[-1]
        print(f"  [ok] tagged record: provider={rec['provider']} model={rec['model']} "
              f"in={rec['input_tokens']} out={rec['output_tokens']}")
        if rec["provider"] != "openai" or not rec["input_tokens"]:
            ok = False
            print("  [FAIL] record missing provider/tokens")

    print("\nAIDER E2E:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
