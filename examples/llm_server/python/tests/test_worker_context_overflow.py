# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json

import httpx
import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
)
from executorch.examples.llm_server.python.protocol import ChatMessage
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.worker_client import WorkerError


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream,timeout", [(False, None), (True, None), (True, 0)])
def test_native_overflow_counts_reach_http(active_peer, dialect, stream, timeout):
    async def scenario():
        async with active_peer() as (worker, peer, gate):
            await asyncio.get_running_loop().sock_sendall(gate, b"R")
            async for _ in peer:
                pass
            await peer.wait()
            template = ChatTemplate(allow_fallback=True)
            # No HF tokenizer: only the native tokenizer knows the true size.
            serving = ServingChat(
                SessionRuntime(worker),
                template,
                "test",
                max_context=128,
                api_dialect=dialect,
                streaming_preflight_timeout=timeout,
            )
            messages = [ChatMessage(role="user", content="x" * 200)]
            expected = len(template.render(messages).encode("utf-8"))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test")),
                base_url="http://test",  # @lint-ignore: in-process ASGI transport
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [m.model_dump(exclude_none=True) for m in messages],
                        "max_tokens": 5,
                        "stream": stream,
                        "session_id": "overflow",
                    },
                )
                if stream and timeout == 0:
                    assert response.status_code == 200
                    errors = [
                        json.loads(line[6:])["error"]
                        for line in response.text.splitlines()
                        if line.startswith('data: {"error"')
                    ]
                    assert len(errors) == 1
                    error = errors[0]
                    assert '"finish_reason"' not in response.text
                else:
                    assert response.status_code == 400
                    error = response.json()["error"]
                assert error["code"] == "context_length_exceeded"
                if dialect == "openrouter":
                    assert error["metadata"] == {
                        "error_type": "context_length_exceeded",
                        "provider_code": "context_length_exceeded",
                        "context_length": 128,
                        "prompt_tokens": expected,
                        "completion_tokens": 5,
                    }
                    assert "maximum context length is 123 tokens." in error["message"]
                    assert f"messages resulted in {expected} tokens" in error["message"]
                else:
                    assert "metadata" not in error
                    assert f"request has {expected} prompt tokens" in error["message"]
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "session_id": "overflow",
                    },
                )
                assert response.status_code == 200
                assert worker.healthy

    asyncio.run(asyncio.wait_for(scenario(), 15))


@pytest.mark.parametrize(
    "counts",
    [
        None,
        {},
        {"prompt_tokens": True, "context_length": 128, "completion_tokens": 1},
        {"prompt_tokens": 200, "context_length": 0, "completion_tokens": 1},
    ],
)
def test_malformed_worker_overflow_counts_are_rejected(counts):
    with pytest.raises(WorkerError, match="context overflow counts"):
        MultiplexedWorkerClient._validate_message(
            {
                "request_id": 1,
                "error": "overflow",
                "code": "context_length_exceeded",
                "context_overflow": counts,
            }
        )


def test_old_worker_overflow_without_counts_does_not_invent_metadata():
    error = ServingChat._generation_error(
        WorkerError("too long", code="context_length_exceeded")
    )
    assert error.status == 400
    assert "metadata" not in error.body(openrouter=True)["error"]
