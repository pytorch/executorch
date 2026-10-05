# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image HTTP admission failures, chunked bodies, and native ownership."""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from copy import deepcopy

import httpx
import pytest

from executorch.examples.llm_server.python import session_runtime
from executorch.examples.llm_server.python.image_input import validate_image
from executorch.examples.llm_server.python.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.tests.conftest import FakeRunner
from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
    _call,
    _NativeWorker,
)
from executorch.examples.llm_server.python.tests.test_image_input import (
    _record,
    encoded,
    image_part,
    LIMITS,
    png,
    template,
)
from executorch.examples.llm_server.python.tests.test_multiplexed_worker_client import (
    _run,
)
from executorch.examples.llm_server.python.tool_parsers import HermesDetector
from executorch.examples.llm_server.python.worker_client import WorkerError


@pytest.mark.parametrize(
    "replacement", ["", "{{p.text}}{{p.text}}", "{{p.text|lower}}"]
)
def test_jinja_cannot_drop_duplicate_or_alter_binding(replacement):
    async def scenario():
        worker = _NativeWorker(image_limits=LIMITS)
        runtime = SessionRuntime(worker)
        chat_template = template()
        chat_template._hf.chat_template = chat_template._hf.chat_template.replace(
            "{{p.text}}", replacement
        )
        serving = ServingChat(runtime, chat_template, "test-model")
        serving._transcript.record_assistant_turn(
            session_id="s",
            content="prior",
            tool_calls=None,
            generated_token_ids=[1],
            prior_turns=0,
        )
        previous = dict(serving._transcript._turns["s"])
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",
            ) as http:
                response = await http.post(
                    "/v1/chat/completions",
                    json={
                        "session_id": "s",
                        "messages": [
                            {"role": "assistant", "content": "edited"},
                            {"role": "user", "content": [image_part()]},
                        ],
                        "stream": True,
                    },
                )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "invalid_image"
            assert worker.calls.empty() and not worker._requests
            assert serving._transcript._turns["s"] == previous
        finally:
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "failure", ["initial_raise", "second_raise", "drop", "duplicate", "alter"]
)
def test_image_render_and_binding_errors_do_not_prune_stale_tail(
    stream, failure, monkeypatch
):
    async def scenario():
        worker = _NativeWorker(image_limits=LIMITS)
        runtime = SessionRuntime(worker)
        chat_template = template()
        serving = ServingChat(runtime, chat_template, "test-model")
        source = [ChatMessage(role="user", content=[image_part()])]
        _record(serving._transcript, source)
        source += [
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="next"),
        ]
        _record(serving._transcript, source, content="last", ids=[800])
        source += [ChatMessage(role="assistant", content="edited")]
        previous = deepcopy(serving._transcript._turns)
        render = chat_template.render
        calls = []

        def failing_render(*args, **kwargs):
            calls.append(args[0])
            if (failure == "initial_raise" and len(calls) == 1) or (
                failure == "second_raise" and len(calls) == 2
            ):
                raise ValueError("template failed")
            result = render(*args, **kwargs)
            if len(calls) == 2:
                marker = re.search(r"ETIMAGE_[a-z0-9]+_END", result).group()
                replacement = {
                    "drop": "",
                    "duplicate": marker + marker,
                    "alter": marker.lower(),
                }[failure]
                result = result.replace(marker, replacement)
            return result

        monkeypatch.setattr(chat_template, "render", failing_render)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",
            ) as http:
                body = {
                    "session_id": "s",
                    "messages": [m.model_dump() for m in source],
                    "stream": stream,
                }
                response = await http.post("/v1/chat/completions", json=body)
                assert response.status_code == 400
                assert response.json()["error"]["code"] == "invalid_image"
                assert len(calls) == (1 if failure == "initial_raise" else 2)
                assert serving._transcript._turns == previous
                assert worker.calls.empty() and not worker._requests
                assert not serving._transactions._entries
                # Once construction succeeds, invalidate exactly the stale tail.
                monkeypatch.setattr(chat_template, "render", render)
                pending = asyncio.create_task(
                    http.post("/v1/chat/completions", json=body)
                )
                _, request_id, config = await _call(worker)
                assert [s["ids"] for s in config.prompt_segments if "ids" in s] == [
                    [700, 701]
                ]
                assert set(serving._transcript._turns["s"]) == {0}
                worker.finish(request_id)
                assert (await pending).status_code == 200
        finally:
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_image_tool_response_roundtrip_keeps_exact_ids(stream):
    async def scenario():
        raw = '<tool_call>{"name":"look", "arguments":{"a":1,"b":2}}</tool_call>'
        worker = _NativeWorker(image_limits=LIMITS, tokens=(raw,))
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime, template(), "test-model", tool_detector_cls=HermesDetector
        )
        body = {
            "session_id": "s",
            "messages": [{"role": "user", "content": [image_part()]}],
            "stream": stream,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "look", "parameters": {"type": "object"}},
                }
            ],
        }
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",
            ) as http:
                pending = asyncio.create_task(
                    http.post("/v1/chat/completions", json=body)
                )
                _, request_id, _ = await _call(worker)
                worker.finish(request_id, generated_token_ids=[700, 701])
                response = await pending
                assert response.status_code == 200
                if stream:
                    chunks = [
                        json.loads(line[6:])
                        for line in response.text.splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    ]
                    calls = [
                        call
                        for chunk in chunks
                        for choice in chunk["choices"]
                        for call in choice["delta"].get("tool_calls", [])
                    ]
                else:
                    calls = response.json()["choices"][0]["message"]["tool_calls"]
                assert len(calls) == 1 and calls[0]["function"]["name"] == "look"
                calls[0]["function"]["arguments"] = ' { "b": 2, "a": 1 } '
                calls[0]["id"] = "echoed-call-id"
                body["messages"] += [
                    {"role": "assistant", "tool_calls": calls},
                    {
                        "role": "tool",
                        "tool_call_id": "echoed-call-id",
                        "content": "tool output",
                    },
                ]
                pending = asyncio.create_task(
                    http.post("/v1/chat/completions", json=body)
                )
                _, request_id, config = await _call(worker)
                assert [s["ids"] for s in config.prompt_segments if "ids" in s] == [
                    [700, 701]
                ]
                assert sum("image" in s for s in config.prompt_segments) == 1
                assert "tool output" in "".join(
                    s.get("text", "") for s in config.prompt_segments
                )
                worker.finish(request_id, generated_token_ids=[702])
                assert (await pending).status_code == 200
                assert serving._transcript._turns["s"][1]["ids"] == [702]
        finally:
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("chunked", [False, True])
def test_http_rejects_oversized_body_before_json_validation(chunked):
    async def scenario():
        worker = _NativeWorker(image_limits=LIMITS)
        runtime = SessionRuntime(worker)
        serving = ServingChat(runtime, template(), "test-model")
        consumed = []

        async def chunks():
            for index in range(18):
                consumed.append(index)
                yield b"x" * (64 * 1024)

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",
            ) as http:
                response = await http.post(
                    "/v1/chat/completions",
                    content=chunks() if chunked else b"x" * (1024 * 1024 + 1),
                    headers={"content-type": "application/json"},
                )
                assert response.status_code == 413
                assert len(consumed) == (17 if chunked else 0)
                assert worker.calls.empty()
        finally:
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("factory_mode", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_large_body_policy_resolves_active_adapter(native, factory_mode, chunked):
    async def scenario():
        received = []
        resources = []
        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_part(data=b"x" * (1024 * 1024)),
                        ],
                    }
                ]
            }
        ).encode()
        assert 1024 * 1024 < len(body) < 20 * 1024 * 1024

        class ObservedServing(ServingChat):
            async def create(self, req):
                received.append(req)
                return ChatCompletionResponse(model="test-model", choices=[])

        def make_serving():
            worker = _NativeWorker() if native else FakeRunner(["reply"])
            runtime = SessionRuntime(worker)
            resources.append(runtime)
            return ObservedServing(runtime, template(), "test-model")

        @asynccontextmanager
        async def factory():
            serving = make_serving()
            try:
                yield serving
            finally:
                await serving._runtime.aclose_worker()

        async def chunks(data):
            for index in range(0, len(data), 64 * 1024):
                yield data[index : index + 64 * 1024]

        app = (
            build_app(None, "test-model", serving_factory=factory)
            if factory_mode
            else build_app(make_serving(), "test-model")
        )
        assert bool(resources) is not factory_mode
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as http:
                    response = await http.post(
                        "/v1/chat/completions",
                        content=chunks(body) if chunked else body,
                        headers={"content-type": "application/json"},
                    )
                    assert response.status_code == (413 if native else 200)
                    assert len(received) == (0 if native else 1)
                    if native:
                        # Invalid JSON proves rejection precedes FastAPI parsing.
                        invalid = b"x" * len(body)
                        response = await http.post(
                            "/v1/chat/completions",
                            content=chunks(invalid) if chunked else invalid,
                            headers={"content-type": "application/json"},
                        )
                        assert response.status_code == 413
                    else:
                        assert (
                            received[0]
                            .messages[0]
                            .content[0]["image_url"]["url"]
                            .startswith("data:image/png;base64,")
                        )
        finally:
            if not factory_mode:
                await resources[0].aclose_worker()

    _run(scenario())


def test_legacy_runtime_retains_adapter_owned_image_forwarding():
    async def scenario():
        worker = FakeRunner(["reply"])
        runtime = SessionRuntime(worker)
        segments = [
            {"text": "before"},
            {
                "image": {
                    "encoding": "base64",
                    "mime_type": "image/png",
                    "data": encoded(b"x" * (1024 * 1024)),
                }
            },
        ]
        generation = runtime.generate_stream(
            None, PromptInput(segments=segments), GenerationOptions(max_new_tokens=1)
        )
        try:
            async for _ in generation:
                pass
            await generation.result()
            assert worker.captured_config.prompt_segments == segments
            assert not runtime.uses_native_transport
        finally:
            await generation.aclose()
            await runtime.aclose_worker()

    _run(scenario())


def test_chunked_image_http_uses_direct_native_stream(monkeypatch):
    async def scenario():
        def forbidden(*args, **kwargs):
            raise AssertionError(
                "native image path cannot use a bridge, thread, or executor"
            )

        monkeypatch.setattr(session_runtime, "_GenerationBridge", forbidden)
        monkeypatch.setattr(session_runtime, "ThreadPoolExecutor", forbidden)
        monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", forbidden)
        worker = _NativeWorker(image_limits=LIMITS)
        runtime = SessionRuntime(worker)
        serving = ServingChat(runtime, template(), "test-model")
        body = json.dumps(
            {"messages": [{"role": "user", "content": [image_part()]}]}
        ).encode()

        async def chunks():
            for index in range(0, len(body), 7):
                yield body[index : index + 7]

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",
            ) as http:
                pending = asyncio.create_task(
                    http.post(
                        "/v1/chat/completions",
                        content=chunks(),
                        headers={"content-type": "application/json"},
                    )
                )
                _, request_id, config = await _call(worker)
                assert any("image" in s for s in config.prompt_segments)
                worker.finish(request_id)
                assert (await pending).status_code == 200
                assert runtime._executor is None
        finally:
            await runtime.aclose_worker()
        assert worker._reader.done() and worker._writer.done() and worker._proc.reaped

    _run(scenario())


def test_runtime_preserves_image_position_stats_and_every_generated_id():
    async def scenario():
        worker = _NativeWorker(image_limits=LIMITS)
        runtime = SessionRuntime(worker)
        generation = runtime.generate_stream(
            "s",
            PromptInput(
                segments=[
                    {"text": "before"},
                    validate_image("image/png", encoded(png()), LIMITS),
                    {"ids": [1, 2]},
                ]
            ),
            GenerationOptions(max_new_tokens=3),
        )
        try:
            assert await anext(generation) == "reply"
            _, request_id, _ = await _call(worker)
            worker.finish(
                request_id,
                completion_tokens=3,
                generated_token_ids=[3, 4, 5],
                prompt_tokens=8,
                prompt_positions=20,
                reused_prompt_positions=0,
                prefilled_prompt_positions=20,
            )
            stats = await generation.result()
            assert stats.generated_token_ids == [3, 4, 5]
            assert stats.completion_tokens == 3 and stats.prompt_tokens == 8
            assert stats.prompt_positions == stats.prefilled_prompt_positions == 20
            assert stats.reused_prompt_positions == stats.reused_prompt_tokens == 0
        finally:
            await generation.aclose()
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_image_stream_close_fences_native_lifecycle(operation):
    async def scenario():
        worker = _NativeWorker(image_limits=LIMITS, cooperative=False)
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
        serving = ServingChat(runtime, template(), "test-model")
        request = ChatCompletionRequest(
            session_id="s",
            messages=[{"role": "user", "content": [image_part()]}],
            stream=True,
        )
        try:
            stream = await serving.create(request)
            _, request_id, _ = await _call(worker)
            lifecycle = asyncio.create_task(
                getattr(serving, operation + "_session")("s")
            )
            await stream.aclose()
            assert worker.cancelled == [request_id]
            assert not lifecycle.done() and worker.calls.empty()
            worker.finish(request_id)
            assert await _call(worker) == (operation, "s")
            await lifecycle
            assert (
                not serving._transactions._entries
                and not runtime._session_locks._entries
            )
            assert runtime.healthy
        finally:
            await runtime.aclose_worker()

    _run(scenario())


def test_native_image_capability_rejection_releases_reservation():
    async def scenario():
        worker = _NativeWorker()
        try:
            request_id = worker.reserve_request()
            with pytest.raises(WorkerError, match="support images"):
                worker.generate(
                    "",
                    type(
                        "Config",
                        (),
                        {
                            "prompt_segments": [
                                validate_image("image/png", encoded(png()), LIMITS)
                            ]
                        },
                    )(),
                    request_id,
                )
            assert not worker._requests and not worker._writes
        finally:
            await worker.close()

    _run(scenario())
