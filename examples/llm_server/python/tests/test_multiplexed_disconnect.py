# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real sockets: disconnect one SSE request without cancelling its worker peer."""

import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.worker_client import spawn_worker


_WORKER = r"""
import json, os, sys
from pathlib import Path

fd = os.environ.get('EXECUTORCH_LLM_WORKER_CONTROL_FD')
if fd is not None:
    os.close(int(fd))
print(json.dumps(dict(ready=True, multiplexed=True, max_inflight_requests=4, max_named_sessions=4)), flush=True)
active = {}
def send(request_id, **fields):
    print(json.dumps(dict(request_id=request_id, **fields)), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    request_id = request['request_id']
    if request['op'] == 'generate':
        session_id = request['session_id']
        active[session_id] = request_id
        if sys.argv[3] == 'True' and session_id == 'a':
            Path(sys.argv[1] + '.started').write_text('started')
        else:
            send(request_id, token='worker-token-visible-content')
    elif request['op'] == 'cancel':
        assert set(active) == {'a', 'b'}
        assert request['target_request_id'] == active['a']
        send(request_id, cancelled=True)
        if sys.argv[2] == 'True':
            send(active['a'], done=True, cancelled=True, finish_reason='stop')
        Path(sys.argv[1]).write_text(json.dumps(dict(cancelled=active['a'], unaffected=active['b'])))
        send(active['b'], done=True, cancelled=False, finish_reason='stop', prompt_tokens=3, completion_tokens=1, generated_token_ids=[7])
    else:
        raise AssertionError('implicit generation must not send explicit open')
"""


@pytest.mark.parametrize("settles", [True, False])
@pytest.mark.parametrize("preflight", [True, False])
def test_socket_disconnect_cancels_only_owning_multiplexed_request(
    tmp_path, settles, preflight
):
    marker = tmp_path / "cancelled.json"
    worker = spawn_worker(
        [sys.executable, "-u", "-c", _WORKER, str(marker), str(settles), str(preflight)]
    )
    runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
    serving = ServingChat(
        runtime, ChatTemplate(hf_tokenizer_path=None, allow_fallback=True), "test-model"
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(serving, "test-model"),
            log_level="error",
            lifespan="off",
            loop="asyncio",
        )
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    pool = ThreadPoolExecutor(max_workers=1)
    started_b = threading.Event()
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def body(session_id):
        return dict(
            model="test-model",
            session_id=session_id,
            stream=True,
            stream_options=dict(include_usage=True),
            messages=[dict(role="user", content="hi")],
            max_tokens=8,
        )

    def consume_b():
        chunks = []
        with httpx.stream("POST", url, json=body("b"), timeout=5) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if "worker-token" in line:
                    started_b.set()
                if line:
                    chunks.append(line)
        return chunks

    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        peer = pool.submit(consume_b)
        assert started_b.wait(5)
        if preflight:
            payload = json.dumps(body("a")).encode()
            with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
                headers = (
                    f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n"
                ).encode()
                connection.sendall(headers + payload)
                started = marker.with_name(marker.name + ".started")
                deadline = time.monotonic() + 5
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert started.exists()
        else:
            with httpx.stream("POST", url, json=body("a"), timeout=5) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    if "worker-token" in line:
                        break
        chunks = peer.result(timeout=5)
        assert chunks[-1] == "data: [DONE]"
        assert not any('"error"' in chunk for chunk in chunks)
        assert any('"completion_tokens":1' in chunk for chunk in chunks)
        observed = json.loads(marker.read_text())
        assert observed["cancelled"] != observed["unaffected"]
        if not settles:
            deadline = time.monotonic() + 2
            while not runtime._settlements and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(runtime._settlements) == 1
            assert runtime._admitted == 1
        assert runtime.healthy
        assert (
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200
        )
    finally:
        runtime.close_worker()
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        pool.shutdown(wait=True)
        assert not thread.is_alive()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX cancellation pipe")
def test_legacy_mailbox_overflow_drains_terminal_and_preserves_next_request(
    tmp_path, monkeypatch
):
    import asyncio

    from executorch.examples.llm_server.python.session_runtime import _GenerationBridge
    from executorch.examples.llm_server.python.worker_client import WorkerError

    release = tmp_path / "release_burst"
    script = r"""
import json, os, sys, time
from pathlib import Path
fd = int(os.environ['EXECUTORCH_LLM_WORKER_CONTROL_FD'])
print(json.dumps(dict(ready=True, supports_cancel=True)), flush=True)
first = json.loads(sys.stdin.readline())
print(json.dumps(dict(token='ready')), flush=True)
while not Path(sys.argv[1]).exists():
    time.sleep(.001)
for _ in range(257):
    print(json.dumps(dict(token='queued')), flush=True)
assert int.from_bytes(os.read(fd, 8), 'little') == first['cancel_request_id']
print(json.dumps(dict(token='must-be-drained')), flush=True)
print(json.dumps(dict(done=True, cancelled=True, finish_reason='stop')), flush=True)
second = json.loads(sys.stdin.readline())
assert second['prompt'] == 'next'
print(json.dumps(dict(token='next')), flush=True)
print(json.dumps(dict(done=True)), flush=True)
for _ in sys.stdin:
    pass
"""
    overflow = threading.Event()
    token_cb = _GenerationBridge.token_cb

    def observe_overflow(bridge, token):
        try:
            token_cb(bridge, token)
        finally:
            if bridge.drop_tokens.is_set():
                overflow.set()

    monkeypatch.setattr(_GenerationBridge, "token_cb", observe_overflow)

    async def scenario():
        worker = spawn_worker([sys.executable, "-u", "-c", script, str(release)])
        runtime = SessionRuntime(worker)
        options = GenerationOptions(max_new_tokens=1024)
        try:
            assert not worker.supports_multiplexing
            async with runtime.generate_stream(
                "s", PromptInput(text="first"), options
            ) as generation:
                assert await anext(generation) == "ready"
                release.write_text("release")
                assert await asyncio.to_thread(overflow.wait, 5)
                with pytest.raises(WorkerError, match="mailbox overflow") as error:
                    await anext(generation)
                assert error.value.code == "slow_consumer"
            assert generation.stats.cancelled
            assert worker.healthy and runtime.healthy and not worker.failed
            async with runtime.generate_stream(
                "s", PromptInput(text="next"), options
            ) as successor:
                assert [token async for token in successor] == ["next"]
            assert runtime.healthy
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_sse_disconnect_retries_pending_cancel_and_keeps_lease_until_done():
    import asyncio
    from types import SimpleNamespace

    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.server import _ClosingStreamingResponse

    script = r"""
import json, os, sys
fd = os.environ.get('EXECUTORCH_LLM_WORKER_CONTROL_FD')
if fd is not None:
    os.close(int(fd))
print(json.dumps(dict(ready=True, multiplexed=True, max_inflight_requests=2, max_named_sessions=4)), flush=True)
def recv():
    return json.loads(sys.stdin.readline())
def send(request, **fields):
    print(json.dumps(dict(request_id=request['request_id'], **fields)), flush=True)
delayed = []
for _ in range(2):
    seed = recv()
    send(seed, token='seed')
    cancel = recv()
    assert cancel['op'] == 'cancel' and cancel['target_request_id'] == seed['request_id']
    delayed.append(cancel)
    send(seed, done=True, cancelled=True, finish_reason='stop')
send(recv(), opened=True)
target = recv()
assert target['session_id'] == 's'
send(target, token='ready')
release = recv()
assert release['op'] == 'open' and release['session_id'] == 'release_acks'
for cancel in delayed:
    send(cancel, cancelled=True)
send(release, opened=True)
retried = recv()
assert retried['op'] == 'cancel' and retried['target_request_id'] == target['request_id']
send(retried, cancelled=True)
barrier = recv()
assert barrier['op'] == 'open' and barrier['session_id'] == 'cancel_seen'
send(barrier, opened=True)
terminal = recv()
assert terminal['op'] == 'open' and terminal['session_id'] == 'release_terminal', 'same-session successor submitted before done'
send(target, done=True, cancelled=True, finish_reason='stop')
send(terminal, opened=True)
successor = recv()
assert successor['op'] == 'generate' and successor['session_id'] == 's'
send(successor, token='next')
send(successor, done=True)
for _ in sys.stdin:
    pass
"""

    def abandon_seed(_):
        raise ValueError("seed cancellation")

    async def scenario():
        worker = spawn_worker([sys.executable, "-u", "-c", script])
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
        serving = ServingChat(
            runtime,
            ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
            "test-model",
        )
        disconnected, body_started = asyncio.Event(), asyncio.Event()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                body_started.set()

        async def collect_successor():
            async with runtime.generate_stream(
                "s", PromptInput(text="next"), GenerationOptions(max_new_tokens=8)
            ) as generation:
                return [token async for token in generation]

        try:
            for _ in range(2):
                with pytest.raises(ValueError, match="seed cancellation"):
                    await asyncio.to_thread(
                        worker.generate, "seed", SimpleNamespace(), abandon_seed
                    )
            await asyncio.to_thread(worker.open_session, "seed_barrier")
            request = ChatCompletionRequest(
                model="test-model",
                session_id="s",
                stream=True,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
            )
            stream = await serving.create(request)
            target = stream._generation
            response = _ClosingStreamingResponse(stream, media_type="text/event-stream")
            response_task = asyncio.create_task(
                response(
                    {"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send
                )
            )
            await asyncio.wait_for(body_started.wait(), 2)
            disconnected.set()
            await asyncio.wait_for(response_task, 2)
            with worker._lock:
                state = worker._requests[target.request_id]
                assert (
                    state.cancel_pending
                    and not state.cancel_requested
                    and not state.wire_done
                )
                assert len(worker._controls) == 2
            assert runtime._admitted == 1 and len(runtime._settlements) == 1
            successor = asyncio.create_task(collect_successor())
            await asyncio.sleep(0)
            assert runtime._session_locks._entries["s"].users == 2
            await asyncio.to_thread(worker.open_session, "release_acks")
            await asyncio.to_thread(worker.open_session, "cancel_seen")
            with worker._lock:
                assert (
                    state.cancel_requested
                    and not state.cancel_pending
                    and not state.wire_done
                )
            assert not successor.done() and not target._future.done()
            assert runtime._session_locks._entries["s"].users == 2
            await asyncio.to_thread(worker.open_session, "release_terminal")
            assert await asyncio.wait_for(successor, 2) == ["next"]
            assert (await target.result()).cancelled
            await asyncio.sleep(0)
            assert runtime._admitted == 0 and not runtime._session_locks._entries
            assert runtime.healthy
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_local_callback_failure_keeps_session_lease_until_wire_terminal():
    import asyncio

    from executorch.examples.llm_server.python.worker_client import WorkerError

    script = r"""
import json, os, sys
fd = os.environ.get('EXECUTORCH_LLM_WORKER_CONTROL_FD')
if fd is not None:
    os.close(int(fd))
print(json.dumps(dict(ready=True, multiplexed=True, max_inflight_requests=4, max_named_sessions=2)), flush=True)
def recv():
    return json.loads(sys.stdin.readline())
def send(request, **fields):
    print(json.dumps(dict(request_id=request['request_id'], **fields)), flush=True)
first = recv()
assert first['session_id'] == 's'
send(first, token='x' * 32)
cancel = recv()
assert cancel['op'] == 'cancel' and cancel['target_request_id'] == first['request_id']
send(cancel, cancelled=True)
other = recv()
assert other['session_id'] == 'other', 'same session admitted before its old wire terminal'
send(other, token='ok')
send(other, done=True)
send(first, done=True, cancelled=True, finish_reason='stop')
successor = recv()
assert successor['session_id'] == 's'
send(successor, token='ok')
send(successor, done=True)
for line in sys.stdin:
    pass
"""

    async def scenario():
        worker = spawn_worker([sys.executable, "-u", "-c", script])
        runtime = SessionRuntime(
            worker, max_buffered_chars=16, cancel_grace_seconds=0.01
        )

        async def collect(session_id):
            async with runtime.generate_stream(
                session_id, PromptInput(text="hi"), GenerationOptions(max_new_tokens=8)
            ) as generation:
                return [token async for token in generation]

        try:
            with pytest.raises(WorkerError, match="mailbox overflow"):
                await collect("s")
            assert runtime._admitted == 1
            assert len(runtime._settlements) == 1
            successor = asyncio.create_task(collect("s"))
            await asyncio.sleep(0)
            other = asyncio.create_task(collect("other"))
            assert await asyncio.wait_for(asyncio.gather(successor, other), 2) == [
                ["ok"],
                ["ok"],
            ]
            await asyncio.sleep(0)
            assert runtime._admitted == 0
            assert not runtime._session_locks._entries
            assert runtime.healthy
        finally:
            runtime.close_worker()

    asyncio.run(scenario())
