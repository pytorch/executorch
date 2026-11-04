# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatMessage
from executorch.examples.llm_server.python.server import _with_heartbeats


def test_unknown_context_is_omitted_from_model_metadata(make_client):
    client, _ = make_client()
    assert "context_length" not in client.get("/v1/models").json()["data"][0]


@pytest.mark.parametrize("failed", [False, True])
def test_heartbeat_cleanup_while_waiting_for_source(failed):
    async def scenario():
        gate = asyncio.Event()
        closed = []

        async def source():
            try:
                await gate.wait()
                raise RuntimeError("source failed")
                yield
            finally:
                closed.append(True)

        stream = _with_heartbeats(source(), interval=0.001)
        assert await anext(stream) == ": keep-alive\n\n"
        if failed:
            gate.set()
            with pytest.raises(RuntimeError, match="source failed"):
                await anext(stream)
        await stream.aclose()
        assert closed == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "bogus", "content": "keep me"}]},
        {"tools": [{"type": "function", "function": {"name": []}}]},
        {"tools": [{"type": "function", "function": "bad"}]},
        {"max_tokens": True},
        {"max_tokens": "12"},
        {
            "messages": [
                {"role": "tool", "tool_call_id": "missing", "content": "result"}
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "x", "function": {"name": "f", "arguments": "{broken"}}
                    ],
                }
            ]
        },
    ],
)
def test_bad_structures_are_rejected_before_generation(make_client, body):
    client, worker = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **body},
    )
    assert response.status_code == 422
    assert worker.captured_config is None


def test_model_metadata_reports_context_limit(make_client):
    client, _ = make_client(max_context=65536)
    model = client.get("/v1/models").json()["data"][0]
    assert model["context_length"] == 65536


def test_heartbeat_does_not_cancel_or_duplicate_pending_token():
    async def scenario():
        gate = asyncio.Event()
        closed = []

        async def source():
            try:
                await gate.wait()
                yield "data: token\n\n"
            finally:
                closed.append(True)

        stream = _with_heartbeats(source(), interval=0.01)
        assert await anext(stream) == ": keep-alive\n\n"
        gate.set()
        assert await anext(stream) == "data: token\n\n"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert closed == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False, True])
def test_bounded_streaming_preflight_keeps_generation_alive(failed):
    import threading

    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
        _Worker,
    )

    class SlowPrefill(_Worker):
        def __init__(self):
            super().__init__()
            self.prefill = threading.Event()

        def generate(self, prompt, config, token_callback, stats_callback, request_id):
            self.calls.put(("prefill", request_id))
            assert self.prefill.wait(5)
            if failed:
                from executorch.examples.llm_server.python.worker_client import (
                    WorkerError,
                )

                raise WorkerError("late worker error", code="invalid_argument")
            return super().generate(
                prompt, config, token_callback, stats_callback, request_id
            )

    async def scenario():
        worker = SlowPrefill()
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            streaming_preflight_timeout=0.02,
        )
        stream = None
        try:
            stream = await asyncio.wait_for(
                serving.create(
                    ChatCompletionRequest(
                        messages=[ChatMessage(role="user", content="hi")], stream=True
                    )
                ),
                timeout=1,
            )
            assert '"role":"assistant"' in await anext(stream)
            _, request_id = worker.calls.get_nowait()
            assert not worker.cancelled
            worker.prefill.set()
            worker.finish(request_id)
            output = "".join([chunk async for chunk in stream])
            assert ("invalid_argument" if failed else "reply") in output
            assert "[DONE]" in output
        finally:
            worker.prefill.set()
            if stream is not None:
                await stream.aclose()
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("include_usage", [False, True])
def test_openrouter_usage_arrives_before_client_stops_at_finish(
    make_client, inline, include_usage
):
    import json

    client, _ = make_client(
        api_dialect="openrouter" if inline else "openai", tokens=["one", "two"], reuse=2
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": include_usage},
        },
    )
    chunks = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == int(include_usage)
    if include_usage:
        usage_chunk = usage_chunks[0]
        assert usage_chunk["usage"]["completion_tokens"] == 2
        assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
        if inline:
            assert usage_chunk["choices"][0]["finish_reason"]
            assert chunks[-1] is usage_chunk
        else:
            assert usage_chunk["choices"] == []


@pytest.mark.parametrize("arguments", ["", None, "{}"])
def test_empty_historical_arguments_and_null_schema_are_accepted(
    make_client, arguments
):
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "f", "arguments": arguments},
                        }
                    ],
                }
            ],
            "tools": [
                {"type": "function", "function": {"name": "f", "parameters": None}}
            ],
        },
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
def test_hermes_encoded_object_round_trips(make_client, dialect):
    raw = (
        "<tool_call>"
        + json.dumps({"name": "f", "arguments": json.dumps({"city": "Paris"})})
        + "</tool_call>"
    )
    client, _ = make_client(tokens=[raw], api_dialect=dialect)
    body = {
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    first = client.post("/v1/chat/completions", json=body)
    assert first.status_code == 200
    message = first.json()["choices"][0]["message"]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Paris"
    }
    body["messages"].extend(
        [
            message,
            {
                "role": "tool",
                "tool_call_id": message["tool_calls"][0]["id"],
                "content": "sunny",
            },
        ]
    )
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    error = client.post(
        "/v1/chat/completions", json={"messages": [], "max_tokens": True}
    )
    assert error.status_code == 422
    assert error.json()["error"]["code"] == "invalid_value"


@pytest.mark.parametrize("arguments", [[], 42, "not json", "[]"])
def test_hermes_nonobject_arguments_are_not_emitted(make_client, arguments):
    raw = (
        "<tool_call>"
        + json.dumps({"name": "f", "arguments": arguments})
        + "</tool_call>"
    )
    client, _ = make_client(tokens=[raw])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        },
    )
    assert response.status_code == 200
    assert not response.json()["choices"][0]["message"].get("tool_calls")


@pytest.mark.parametrize("reasoning", [{}, {"exclude": False}])
def test_noop_reasoning_is_accepted(make_client, reasoning):
    client, _ = make_client()
    assert (
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": reasoning,
            },
        ).status_code
        == 200
    )


def test_template_input_errors_use_structured_400(make_client):
    from types import SimpleNamespace

    from jinja2 import TemplateError

    client, _ = make_client()

    def render(*args, **kwargs):
        raise TemplateError("unsupported message ordering")

    client.app.state.serving._template._hf = SimpleNamespace(apply_chat_template=render)
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_value"


def test_closing_response_sends_heartbeat_and_closes_source():
    from executorch.examples.llm_server.python.server import _ClosingStreamingResponse

    async def scenario():
        gate = asyncio.Event()
        closed = []
        bodies = []

        async def source():
            try:
                await gate.wait()
                yield "data: [DONE]\n\n"
            finally:
                closed.append(True)

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                bodies.append(message.get("body", b""))
                gate.set()

        response = _ClosingStreamingResponse(source(), heartbeat_interval=0.001)
        await asyncio.wait_for(
            response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send),
            1,
        )
        assert bodies[:2] == [b": keep-alive\n\n", b"data: [DONE]\n\n"]
        assert closed == [True]

    asyncio.run(scenario())
