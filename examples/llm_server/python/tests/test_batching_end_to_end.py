# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in subprocess checks against the real C++ transport and serving runtime."""

import asyncio
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.worker_client import (
    spawn_worker,
    WorkerError,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def native_worker(tmp_path):
    binary = os.environ.get("EXECUTORCH_BATCHING_TEST_WORKER")
    if not binary:
        pytest.skip("Set EXECUTORCH_BATCHING_TEST_WORKER to the native test worker")
    clients = []
    processes = []
    logs = []
    pool = ThreadPoolExecutor(max_workers=4)

    def create(*, gate=False, stop_immediately=False, prefix_cache=False):
        trace = tmp_path / f"batches-{len(processes)}.txt"
        log = (tmp_path / f"worker-{len(processes)}.log").open("w")
        logs.append(log)
        timers = []

        def popen(*args, **kwargs):
            kwargs["stderr"] = log
            process = subprocess.Popen(*args, **kwargs)
            processes.append(process)
            timer = threading.Timer(10, process.kill)
            timer.start()
            timers.append(timer)
            return process

        command = [binary, "--trace", str(trace)]
        if gate:
            command.append("--gate-two")
        if stop_immediately:
            command.append("--stop-immediately")
        if prefix_cache:
            command.append("--prefix-cache")
        try:
            client = spawn_worker(command, popen=popen, require_multiplexing=True)
        finally:
            for timer in timers:
                timer.cancel()
        clients.append(client)
        assert client.supports_multiplexing
        return client, pool, trace

    yield create
    for client in clients:
        client.close()
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    pool.shutdown(wait=True)
    for log in logs:
        log.close()


def collect(
    client,
    prompt,
    *,
    key=None,
    count=4,
    request_id=None,
    callback=None,
    segments=None,
    stop=None,
):
    pieces = []
    stats = []

    def on_token(piece):
        pieces.append(piece)
        if callback:
            callback(piece)

    config = SimpleNamespace(
        session_id=key,
        max_new_tokens=count,
        temperature=0,
        top_p=1,
        top_k=0,
        seed=123,
        stop=stop or [],
    )
    if segments is not None:
        config.prompt_segments = segments
    client.generate(
        prompt,
        config,
        token_callback=on_token,
        stats_callback=stats.append,
        request_id=request_id,
    )
    assert len(stats) == 1
    return "".join(pieces), stats[0]


def test_two_python_requests_reach_one_executor_batch(native_worker):
    client, pool, trace = native_worker(gate=True)
    first = pool.submit(collect, client, "first", key="a")
    second = pool.submit(collect, client, "second", key="b")
    for result in (first.result(timeout=10), second.result(timeout=10)):
        text, stats = result
        assert len(text) == stats.num_generated_tokens == 4
        assert not stats.cancelled
    assert any(int(line) >= 2 for line in trace.read_text().splitlines())


@pytest.mark.anyio
async def test_async_runtime_does_not_serialize_native_generations(native_worker):
    client, _, trace = native_worker(gate=True)
    runtime = SessionRuntime(client, max_concurrent_requests=4)

    async def generate(key):
        async with runtime.generate_stream(
            key,
            PromptInput(text=f"prompt for {key}"),
            GenerationOptions(max_new_tokens=4, seed=123),
        ) as generation:
            pieces = [piece async for piece in generation]
            stats = await generation.result()
        assert len("".join(pieces)) == stats.completion_tokens == 4

    try:
        await asyncio.wait_for(asyncio.gather(generate("a"), generate("b")), 10)
        assert any(int(line) >= 2 for line in trace.read_text().splitlines())
    finally:
        runtime.close_worker()


@pytest.mark.anyio
async def test_openai_streaming_and_nonstreaming_share_native_runner(native_worker):
    worker, _, trace = native_worker(gate=True)
    runtime = SessionRuntime(worker, max_concurrent_requests=4)
    serving = ServingChat(
        runtime,
        ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
        "test-model",
    )
    transport = httpx.ASGITransport(app=build_app(serving, "test-model"))
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            ordinary = {
                "model": "test-model",
                "session_id": "http-a",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            }
            streaming = {
                **ordinary,
                "session_id": "http-b",
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            normal, stream = await asyncio.wait_for(
                asyncio.gather(
                    client.post("/v1/chat/completions", json=ordinary),
                    client.post("/v1/chat/completions", json=streaming),
                ),
                10,
            )
        assert normal.status_code == stream.status_code == 200
        response = normal.json()
        assert len(response["choices"][0]["message"]["content"]) == 4
        assert response["usage"]["completion_tokens"] == 4
        records = [
            line[6:] for line in stream.text.splitlines() if line.startswith("data: ")
        ]
        assert records[-1] == "[DONE]"
        chunks = [json.loads(record) for record in records[:-1]]
        text = "".join(
            choice.get("delta", {}).get("content", "")
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert len(text) == 4
        usage = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
        assert usage[-1]["completion_tokens"] == 4
        assert any(int(line) >= 2 for line in trace.read_text().splitlines())
    finally:
        runtime.close_worker()


@pytest.mark.anyio
async def test_runtime_prefix_cache_is_creation_only(native_worker):
    worker, _, _ = native_worker(prefix_cache=True)
    runtime = SessionRuntime(worker)

    async def generate(key):
        async with runtime.generate_stream(
            key,
            PromptInput(text="hello"),
            GenerationOptions(max_new_tokens=1, temperature=0.0, seed=123),
        ) as generation:
            pieces = [piece async for piece in generation]
            stats = await generation.result()
        assert len("".join(pieces)) == stats.completion_tokens == 1
        return stats

    try:
        cold = await generate("first")
        warm = await generate("second")
        assert cold.reused_prompt_tokens == 0
        assert warm.reused_prompt_tokens == 4
        assert warm.prefilled_prompt_tokens == 1
        await asyncio.to_thread(worker.reset_session, "first")
        reset = await generate("first")
        assert reset.reused_prompt_tokens == 0
        await asyncio.to_thread(worker.open_session, "explicit")
        explicit = await generate("explicit")
        assert explicit.reused_prompt_tokens == 0
    finally:
        runtime.close_worker()


def test_cancel_is_request_scoped(native_worker):
    client, pool, _ = native_worker()
    started = threading.Event()
    request_id = client.reserve_request()
    first = pool.submit(
        collect,
        client,
        "long request",
        key="a",
        count=1024,
        request_id=request_id,
        callback=lambda _: started.set(),
    )
    assert started.wait(timeout=5)
    second = pool.submit(collect, client, "other request", key="b")
    assert client.cancel(request_id)
    _, cancelled = first.result(timeout=10)
    text, completed = second.result(timeout=10)
    assert cancelled.cancelled
    assert len(text) == completed.num_generated_tokens == 4
    assert not completed.cancelled


def test_full_prompt_with_exact_completion_ids_continues_session(native_worker):
    client, _, _ = native_worker()
    _, first = collect(client, "hello", key="named")
    assert len(first.generated_token_ids) == 4
    _, second = collect(
        client,
        "",
        key="named",
        count=2,
        segments=[
            {"text": "hello"},
            {"ids": first.generated_token_ids},
            {"text": " again"},
        ],
    )
    assert second.session_reset_reason == "exact_prefix"
    assert second.reused_prompt_tokens > 0
    assert second.prefilled_prompt_tokens > 0
    assert (
        second.reused_prompt_tokens + second.prefilled_prompt_tokens
        == second.num_prompt_tokens
    )
    assert second.num_generated_tokens == 2


def test_oversized_stop_preserves_resident_session(native_worker):
    client, pool, _ = native_worker()
    _, first = pool.submit(collect, client, "hello", key="named").result(timeout=10)
    assert len(first.generated_token_ids) == 4
    segments = [
        {"text": "hello"},
        {"ids": first.generated_token_ids},
        {"text": " again"},
    ]
    request_id = client.reserve_request()
    rejected = pool.submit(
        collect,
        client,
        "",
        key="named",
        count=2,
        segments=segments,
        stop=["x" * (1024 * 1024)],
        request_id=request_id,
    )
    with pytest.raises(WorkerError) as error:
        rejected.result(timeout=10)
    assert error.value.code == "invalid_argument"
    pool.submit(client.wait_for_request, request_id).result(timeout=5)
    assert client.healthy
    _, continued = pool.submit(
        collect, client, "", key="named", count=2, segments=segments
    ).result(timeout=10)
    assert continued.session_reset_reason == "exact_prefix"
    assert continued.reused_prompt_tokens > 0
    assert continued.prefilled_prompt_tokens > 0
    assert (
        continued.reused_prompt_tokens + continued.prefilled_prompt_tokens
        == continued.num_prompt_tokens
    )
    assert continued.num_generated_tokens == 2


def test_terminal_only_completion_preserves_known_empty_ids(native_worker):
    client, _, _ = native_worker(stop_immediately=True)
    text, stats = collect(client, "hello")
    assert text == ""
    assert stats.num_generated_tokens == 0
    assert stats.finish_reason == "stop"
    assert stats.generated_token_ids == []


def test_empty_visible_string_stop_has_unknown_replay_ids(native_worker):
    client, _, _ = native_worker()
    first_text, first = collect(client, "hello", key="named", count=1)
    text, stats = collect(
        client,
        "",
        key="named",
        count=1,
        segments=[
            {"text": "hello"},
            {"ids": first.generated_token_ids},
            {"text": " next"},
        ],
        stop=[first_text],
    )
    assert text == ""
    assert stats.num_generated_tokens == 1
    assert stats.finish_reason == "stop"
    assert stats.generated_token_ids is None


def test_reset_and_close_keep_public_session_key_usable(native_worker):
    client, _, _ = native_worker()
    client.open_session("named")
    collect(client, "before reset", key="named")
    client.reset_session("named")
    _, reset = collect(client, "after reset", key="named")
    assert reset.reused_prompt_tokens == 0
    client.close_session("named")
    _, reopened = collect(client, "reopened", key="named")
    assert reopened.reused_prompt_tokens == 0
    assert reopened.num_generated_tokens == 4


def test_reserved_older_id_can_arrive_after_lifecycle_completion(native_worker):
    client, _, _ = native_worker()
    older_id = client.reserve_request()
    client.open_session("named")
    text, stats = collect(client, "hello", key="named", request_id=older_id)
    assert len(text) == stats.num_generated_tokens == 4


def test_slow_consumer_does_not_stall_another_request(native_worker):
    client, pool, _ = native_worker()
    started = threading.Event()
    release = threading.Event()

    def hold(_):
        started.set()
        release.wait(timeout=5)

    first = pool.submit(collect, client, "slow", key="a", count=1024, callback=hold)
    try:
        assert started.wait(timeout=5)
        # Enough decode steps to fill A's bounded mailbox while its caller waits.
        second = pool.submit(collect, client, "fast", key="b", count=128)
        text, stats = second.result(timeout=5)
        assert len(text) == stats.num_generated_tokens == 128
        assert not stats.cancelled
    finally:
        release.set()
    with pytest.raises(WorkerError):
        first.result(timeout=10)
    text, stats = collect(client, "still healthy", key="c", count=2)
    assert len(text) == stats.num_generated_tokens == 2
