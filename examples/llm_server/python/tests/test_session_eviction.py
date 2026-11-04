# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.openai_transcript import (
    OpenAITranscriptState,
)
from executorch.examples.llm_server.python.protocol import ChatMessage
from executorch.examples.llm_server.python.worker_client import WorkerError


def test_request_overload_never_evicts_sessions(make_client):
    client, worker = make_client(max_named_sessions=3, evict_idle_sessions=True)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    for sid in ("a", "b", "c"):
        assert (
            client.post(
                "/v1/chat/completions", json={**body, "session_id": sid}
            ).status_code
            == 200
        )
    attempts = []

    def overloaded(*args, **kwargs):
        attempts.append(True)
        raise WorkerError("request limit reached", code="capacity_exhausted")

    worker.generate = overloaded
    assert client.post("/v1/chat/completions", json=body).status_code == 429
    assert len(attempts) == 1
    assert worker.open_named == {"a", "b", "c"}


def test_close_failure_preserves_session_and_rotates_victim(make_client):
    client, worker = make_client(
        max_named_sessions=2, evict_idle_sessions=True, gen_ids=[1]
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}
    for sid in ("a", "b"):
        assert (
            client.post(
                "/v1/chat/completions", json={**body, "session_id": sid}
            ).status_code
            == 200
        )
    close = worker.close_session

    def fail_a(sid):
        if sid == "a":
            raise WorkerError("close overloaded", code="capacity_exhausted")
        return close(sid)

    worker.close_session = fail_a
    serving = client.app.state.serving
    assert (
        client.post(
            "/v1/chat/completions", json={**body, "session_id": "c"}
        ).status_code
        == 429
    )
    assert "a" in serving._transcript._turns
    assert list(serving._idle_sessions) == ["b", "a"]
    assert (
        client.post(
            "/v1/chat/completions", json={**body, "session_id": "c"}
        ).status_code
        == 200
    )
    assert worker.open_named == {"a", "c"}


def test_failed_admissions_do_not_register_phantom_idle_sessions(make_client):
    client, worker = make_client(evict_idle_sessions=True)
    serving = client.app.state.serving
    serving._multiplexed = True

    async def rejected(*args, **kwargs):
        raise WorkerError("invalid prompt", code="invalid_argument")
        yield

    serving._runtime.generate_stream = rejected
    for sid in ("a", "b", "c"):
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "session_id": sid},
        )
        assert response.status_code == 400
    assert not serving._idle_sessions
    assert not worker.open_named


@pytest.mark.parametrize("close", [False, True])
def test_cancellation_during_eviction_does_not_create_replacement(close):
    from executorch.examples.llm_server.python.serving_chat import _AdmissionRetryStream

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        replacements = []

        class Serving:
            _pending_sessions = set()

            async def _evict_idle_session(self):
                entered.set()
                await release.wait()
                return True

        class Rejected:
            async def __anext__(self):
                raise WorkerError("slots full", code="session_capacity_exhausted")

            async def aclose(self):
                pass

            def cancel(self):
                return True

        stream = _AdmissionRetryStream(
            Serving(), "s", Rejected(), lambda: replacements.append(True)
        )
        read = asyncio.create_task(anext(stream))
        await entered.wait()
        if close:
            await stream.aclose()
        else:
            stream.cancel()
        release.set()
        with pytest.raises(StopAsyncIteration):
            await read
        assert not replacements

    asyncio.run(scenario())


def test_history_prefix_hashing_serializes_each_message_once(monkeypatch):
    state = OpenAITranscriptState(ChatTemplate(allow_fallback=True))
    messages = [ChatMessage(role="user", content="large image source" * 100)]
    for i in range(20):
        state.record_assistant_turn(
            session_id="s",
            request_messages=messages,
            content="OK",
            tool_calls=None,
            generated_token_ids=[100 + i],
            prior_turns=i,
        )
        messages.extend(
            [
                ChatMessage(role="assistant", content="OK"),
                ChatMessage(role="user", content="continue"),
            ]
        )
    original = state._assistant_fingerprint
    calls = []

    def counted(content, tools):
        calls.append(content)
        return original(content, tools)

    monkeypatch.setattr(
        OpenAITranscriptState, "_assistant_fingerprint", staticmethod(counted)
    )
    state.build_prompt_input(
        session_id="s",
        messages=messages,
        rendered_prompt=state._template.render(messages),
        tools=None,
        template_kwargs=None,
    )
    assert calls.count(messages[0].content) == 1


def test_idle_session_eviction_releases_capacity_and_transcript(make_client):
    client, worker = make_client(
        max_named_sessions=2, evict_idle_sessions=True, gen_ids=[1]
    )

    def chat(sid):
        return client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "session_id": sid},
        )

    assert chat("a").status_code == chat("b").status_code == 200
    assert chat("a").status_code == 200  # a is newer than b
    assert chat("c").status_code == 200
    assert worker.open_named == {"a", "c"}
    serving = client.app.state.serving
    assert "b" not in serving._transcript._turns
    for i in range(20):
        assert chat(f"next-{i}").status_code == 200
    assert len(worker.open_named) == len(serving._idle_sessions) == 2


def test_history_identity_prevents_same_answer_collision():
    template = ChatTemplate(allow_fallback=True)
    state = OpenAITranscriptState(template)
    u1 = ChatMessage(role="user", content="removed question")
    u2 = ChatMessage(role="user", content="retained question")
    answer = ChatMessage(role="assistant", content="OK")
    for i, history in enumerate(([u1], [u1, answer, u2])):
        state.record_assistant_turn(
            session_id="s",
            request_messages=history,
            content="OK",
            tool_calls=None,
            generated_token_ids=[100 + i],
            prior_turns=i,
        )
    messages = [u2, answer, ChatMessage(role="user", content="continue")]
    rendered = template.render(messages)
    prompt = state.build_prompt_input(
        session_id="s",
        messages=messages,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assert prompt.text == rendered
    assert prompt.segments is None
    assert not state._turns["s"]


@pytest.mark.parametrize("changed_role", ["system", "developer", "user", "tool"])
def test_changed_preceding_context_invalidates_stored_reply(changed_role):
    template = ChatTemplate(allow_fallback=True)
    state = OpenAITranscriptState(template)
    history = [ChatMessage(role=changed_role, content="before")]
    state.record_assistant_turn(
        session_id="s",
        request_messages=history,
        content="OK",
        tool_calls=None,
        generated_token_ids=[123],
        prior_turns=0,
    )
    messages = [
        ChatMessage(role=changed_role, content="after"),
        ChatMessage(role="assistant", content="OK"),
    ]
    prompt = state.build_prompt_input(
        session_id="s",
        messages=messages,
        rendered_prompt=template.render(messages),
        tools=None,
        template_kwargs=None,
    )
    assert prompt.segments is None


def test_idle_eviction_protects_active_transactions(make_client):
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest

    client, worker = make_client(max_named_sessions=2, evict_idle_sessions=True)
    serving = client.app.state.serving

    async def scenario():
        def request(sid):
            return ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], session_id=sid
            )

        await serving.create(request("a"))
        await serving.create(request("b"))
        async with serving._transactions.hold("a"):
            await serving.create(request("c"))
            assert worker.open_named == {"a", "c"}
            async with serving._transactions.hold("c"):
                assert not await serving._evict_idle_session()

    asyncio.run(scenario())
