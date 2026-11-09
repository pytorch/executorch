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
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "duplicate",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                        for _ in range(2)
                    ],
                }
            ]
        },
        {"messages": [{"role": "tool", "content": "result"}]},
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


@pytest.mark.parametrize("arguments", ["", None, "omitted"])
@pytest.mark.parametrize(
    "render_args",
    [
        "{{ call.function.arguments | items | list | tojson }}",
        "{{ call.function.arguments | tojson }}",
    ],
)
def test_empty_arguments_render_as_objects_in_hf_templates(
    make_client, monkeypatch, arguments, render_args
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{% for call in message.tool_calls | default([]) %}"
        + render_args
        + "{% endfor %}{% endfor %}"
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    client, worker = make_client()
    client.app.state.serving._template = ChatTemplate("local")
    function = {"name": "f"}
    if arguments != "omitted":
        function["arguments"] = arguments
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c", "function": function}],
                },
                {"role": "tool", "tool_call_id": "c", "content": "done"},
            ]
        },
    )
    assert response.status_code == 200, response.text
    expected = "[]" if "items" in render_args else "{}"
    assert (
        client.app.state.serving._template.render(
            [
                ChatMessage(
                    role="assistant", tool_calls=[{"id": "c", "function": function}]
                )
            ]
        )
        == expected
    )
    assert worker.captured_config is not None


def test_hermes_empty_arguments_preserve_the_complete_call_set(make_client):
    raw = '<tool_call>[{"name":"f","arguments":""},{"name":"f","arguments":{}}]</tool_call>'
    client, _ = make_client(tokens=[raw])
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call f twice"}],
            "tools": [
                {"type": "function", "function": {"name": "f", "parameters": {}}}
            ],
        },
    )
    assert response.status_code == 200
    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [{}, {}]


def test_heartbeat_disconnect_closes_real_generation_and_session_lease():
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.server import _ClosingStreamingResponse
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
        _NativeWorker,
    )

    async def scenario():
        worker = _NativeWorker(tokens=())
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            streaming_preflight_timeout=0,
        )
        disconnected = asyncio.Event()
        bodies = []

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                bodies.append(message.get("body", b""))
                if bodies[-1] == b": keep-alive\n\n":
                    disconnected.set()

        try:
            stream = await serving.create(
                ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")],
                    session_id="s",
                    stream=True,
                )
            )
            response = _ClosingStreamingResponse(stream, heartbeat_interval=0.001)
            await asyncio.wait_for(
                response(
                    {"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send
                ),
                2,
            )
            assert b": keep-alive\n\n" in bodies
            assert len(worker.cancelled) == 1
            assert runtime._admitted == 0

            async def reacquire():
                async with runtime._session_locks.hold("s"):
                    async with serving._transactions.hold("s"):
                        pass

            await asyncio.wait_for(reacquire(), 1)
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "history",
    [
        "orphan",
        "summarized",
        "reused-id-missing-result",
        "duplicate-result",
        "complete-reused-id",
    ],
)
def test_compacted_tool_histories_preserve_messages(
    make_client, monkeypatch, dialect, stream, history
):
    from copy import deepcopy

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"a.py"}'},
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "call_0",
        "name": "read",
        "content": "file contents",
    }
    if history == "orphan":
        messages = [result]
    elif history == "summarized":
        messages = [
            {"role": "system", "content": "Earlier work was compacted."},
            result,
        ]
    elif history == "reused-id-missing-result":
        messages = [
            call,
            {"role": "user", "content": "Earlier result was summarized. Read again."},
            deepcopy(call),
            result,
        ]
    elif history == "duplicate-result":
        messages = [call, result, deepcopy(result)]
    else:
        messages = [
            call,
            result,
            {"role": "user", "content": "Read again."},
            deepcopy(call),
            deepcopy(result),
        ]
    messages.append({"role": "user", "content": "Continue from the retained results."})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    tokenizer.chat_template = "{{ messages | tojson }}"
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    client, worker = make_client(api_dialect=dialect)
    client.app.state.serving._template = ChatTemplate("local")
    prompts = []
    generate = worker.generate

    def capture(prompt, *args, **kwargs):
        prompts.append(json.loads(prompt))
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(worker, "generate", capture)
    response = client.post(
        "/v1/chat/completions", json={"messages": messages, "stream": stream}
    )
    assert response.status_code == 200, response.text
    assert len(prompts) == 1
    rendered = prompts[0]
    assert len(rendered) == len(messages)
    for original, actual in zip(messages, rendered):
        for key in ("role", "content", "name", "tool_call_id"):
            if key in original:
                assert actual[key] == original[key]
        if original.get("tool_calls"):
            assert actual["tool_calls"][0]["id"] == "call_0"
            assert actual["tool_calls"][0]["function"] == {
                "name": "read",
                "arguments": {"path": "a.py"},
            }


@pytest.mark.parametrize("close_early", [False, True])
def test_heartbeat_source_and_cleanup_stay_in_one_task_with_bounded_read_ahead(
    close_early,
):
    async def scenario():
        owners = set()
        produced = []
        closed = []

        async def source():
            try:
                for i in range(100):
                    owners.add(asyncio.current_task())
                    produced.append(i)
                    yield str(i)
            finally:
                owners.add(asyncio.current_task())
                closed.append(True)

        stream = _with_heartbeats(source())
        assert await anext(stream) == "0"
        if close_early:
            for _ in range(5):
                await asyncio.sleep(0)
            assert len(produced) <= 3
            await stream.aclose()
        else:
            assert [chunk async for chunk in stream] == [str(i) for i in range(1, 100)]
        assert closed == [True]
        assert len(owners) == 1

    asyncio.run(scenario())


def test_heartbeat_propagates_source_cancellation():
    async def scenario():
        async def source():
            raise asyncio.CancelledError()
            yield

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(anext(_with_heartbeats(source())), 1)

    asyncio.run(scenario())
