# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import re
import threading
from contextlib import AsyncExitStack

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tests.test_concurrent_serving import _Worker
from executorch.examples.llm_server.python.worker_client import WorkerError, WorkerStats


def muse_overflow_counts(body):
    error = body["error"]
    assert error["metadata"]["error_type"] == "context_length_exceeded"
    # Muse's OpenRouter contract: metadata classifies it, message gives the
    # input size and usable limit for its one-shot changed-input retry.
    match = re.match(
        r"This model's maximum context length is (\d+) tokens\. "
        r"However, your messages resulted in (\d+) tokens[. (]",
        error["message"],
    )
    return tuple(map(int, match.groups())) if match else None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "prompt,completion,budget",
    [(110, None, 99), (100, None, 99), (80, 30, 70), (110, 30, 70)],
)
def test_openrouter_http_overflow_supplies_actionable_prompt_counts(
    make_client, stream, prompt, completion, budget
):
    client, worker = make_client(
        max_context=100, prompt_tokens=prompt, api_dialect="openrouter"
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            **({"max_tokens": completion} if completion else {}),
        },
    )
    assert response.status_code == 400
    assert muse_overflow_counts(response.json()) == (budget, prompt)
    assert response.json()["error"]["metadata"]["context_length"] == 100
    assert worker.captured_config is None


@pytest.mark.parametrize("prompt", [20, 110])
def test_impossible_output_reservation_does_not_invent_recoverable_counts(
    make_client, prompt
):
    client, _ = make_client(
        max_context=100, prompt_tokens=prompt, api_dialect="openrouter"
    )
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100},
    )
    assert response.status_code == 400
    assert muse_overflow_counts(response.json()) is None


@pytest.mark.parametrize("completion", [None, 30])
def test_default_openai_error_contract_is_unchanged(make_client, completion):
    client, _ = make_client(max_context=100, prompt_tokens=110)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    if completion is not None:
        body["max_tokens"] = completion
    response = client.post("/v1/chat/completions", json=body)
    message = "This model's maximum context length is 100 tokens, but the request has 110 prompt tokens."
    assert response.json() == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }


def test_late_context_error_has_same_openrouter_contract(make_client):
    client, worker = make_client(api_dialect="openrouter")

    def fail(*args, **kwargs):
        raise WorkerError(
            "context length exceeded",
            code="context_length_exceeded",
            context_overflow={
                "prompt_tokens": 110,
                "context_length": 100,
                "completion_tokens": 1,
            },
        )

    worker.generate = fail
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    errors = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith('data: {"error"')
    ]
    assert len(errors) == 1
    assert muse_overflow_counts(errors[0]) == (99, 110)
    assert '"finish_reason"' not in response.text


class CapacityWorker(_Worker):
    def __init__(self, late, after_output=False):
        super().__init__()
        self.sessions = set()
        self.closed_sessions = []
        self.attempts = []
        self.admission = threading.Event()
        self.after_output = after_output
        if not late:
            self.admission.set()

    def generate(self, prompt, config, token_callback, stats_callback, request_id):
        try:
            sid = config.session_id
            self.attempts.append(sid)
            if sid not in self.sessions and len(self.sessions) >= 2:
                assert self.admission.wait(5)
                if self.after_output:
                    token_callback("partial " * 8)
                raise WorkerError(
                    "all sessions full", code="session_capacity_exhausted"
                )
            self.sessions.add(sid)
            token_callback("ok")
            stats_callback(
                WorkerStats(
                    num_prompt_tokens=3, num_generated_tokens=1, finish_reason="stop"
                )
            )
        finally:
            self.release_request(request_id)

    def close_session(self, sid):
        existed = sid in self.sessions
        self.sessions.discard(sid)
        self.closed_sessions.append(sid)
        return existed


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("limit", [1, 4])
def test_stream_admission_evicts_idle_sessions_before_or_after_preflight(late, limit):
    async def scenario():
        worker = CapacityWorker(late)
        runtime = SessionRuntime(worker, max_concurrent_requests=limit)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            evict_idle_sessions=True,
            streaming_preflight_timeout=0.01,
        )

        def request(sid, stream=False):
            return ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                session_id=sid,
                stream=stream,
            )

        try:
            for sid in ["a", "b"]:
                await serving.create(request(sid))
            stream = await serving.create(request("c", True))
            worker.admission.set()
            output = "".join([chunk async for chunk in stream])
            assert '"content":"ok"' in output
            assert '"error"' not in output
            assert worker.closed_sessions == ["a"]
            assert worker.sessions == set(serving._idle_sessions) == {"b", "c"}
            assert not serving._cleanup_sessions
        finally:
            worker.admission.set()
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("after_output", [False, True])
def test_late_error_never_evicts_active_sessions_or_replays_output(after_output):
    async def scenario():
        worker = CapacityWorker(True, after_output)
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            evict_idle_sessions=True,
            streaming_preflight_timeout=0.01,
        )

        def request(sid, stream=False):
            return ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                session_id=sid,
                stream=stream,
            )

        try:
            for sid in ["a", "b"]:
                await serving.create(request(sid))
            async with AsyncExitStack() as stack:
                if not after_output:
                    await stack.enter_async_context(serving._transactions.hold("a"))
                    await stack.enter_async_context(serving._transactions.hold("b"))
                stream = await serving.create(request("c", True))
                worker.admission.set()
                output = "".join([chunk async for chunk in stream])
            errors = [
                json.loads(line[6:])["error"]
                for line in output.splitlines()
                if line.startswith('data: {"error"')
            ]
            assert len(errors) == 1
            assert errors[0]["code"] == "capacity_exhausted"
            assert not worker.closed_sessions
            assert worker.attempts.count("c") == 1
            if not after_output:
                assert set(serving._idle_sessions) == {"a", "b"}
            else:
                assert '"content":"partial ' in output
        finally:
            worker.admission.set()
            await runtime.aclose_worker()

    asyncio.run(scenario())
