# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.muse import command
from executorch.examples.llm_server.python.protocol import ChatMessage
from executorch.examples.llm_server.python.server import _with_heartbeats


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


def test_model_metadata_drives_muse_compaction_budget(make_client):
    client, _ = make_client(max_context=65536)
    model = client.get("/v1/models").json()["data"][0]
    argv = command("muse", "http://127.0.0.1:8088/v1", model, ["exec", "--json"])
    assert argv[:2] == ["muse", "exec"]
    assert "context_compaction.provider_context_limit_tokens=65536" in argv
    assert argv[-1] == "--json"


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


def test_bounded_streaming_preflight_keeps_generation_alive():
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
            assert "reply" in output and "[DONE]" in output
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
        stream_usage_in_finish_chunk=inline, tokens=["one", "two"], reuse=2
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
