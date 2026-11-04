# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from contextlib import aclosing

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


@pytest.mark.parametrize("opened", [False, True])
def test_unconfirmed_native_admission_cleanup_is_retried(opened):
    from executorch.examples.llm_server.python.errors import APIError
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_openrouter_recovery import (
        CapacityWorker,
    )

    class Worker(CapacityWorker):
        def __init__(self):
            super().__init__(False)
            self.fail_close = opened

        def generate(self, prompt, config, token_callback, stats_callback, request_id):
            if config.session_id == "bad":
                if opened:
                    self.sessions.add("bad")
                self.release_request(request_id)
                raise WorkerError(
                    "generation failed before output", code="invalid_argument"
                )
            return super().generate(
                prompt, config, token_callback, stats_callback, request_id
            )

        def close_session(self, sid):
            if self.fail_close:
                self.fail_close = False
                raise WorkerError("close queue full", code="capacity_exhausted")
            if sid not in self.sessions:
                raise WorkerError("no session", code="session_not_found")
            return super().close_session(sid)

    async def scenario():
        worker = Worker()
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime, ChatTemplate(allow_fallback=True), "test", evict_idle_sessions=True
        )

        def request(sid):
            return ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], session_id=sid
            )

        try:
            with pytest.raises(APIError):
                await serving.create(request("bad"))
            assert not serving._idle_sessions
            assert list(serving._cleanup_sessions) == (["bad"] if opened else [])
            await serving.create(request("b"))
            await serving.create(request("c"))
            assert worker.sessions == {"b", "c"}
            assert not serving._cleanup_sessions
            assert worker.attempts.count("c") == (2 if opened else 1)
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_stale_idle_candidate_does_not_consume_the_retry():
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_openrouter_recovery import (
        CapacityWorker,
    )

    class Worker(CapacityWorker):
        def close_session(self, sid):
            if sid not in self.sessions:
                raise WorkerError("missing session", code="session_not_found")
            return super().close_session(sid)

    async def scenario():
        worker = Worker(False)
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime, ChatTemplate(allow_fallback=True), "test", evict_idle_sessions=True
        )

        def request(sid):
            return ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], session_id=sid
            )

        try:
            for sid in ("a", "b"):
                await serving.create(request(sid))
            worker.sessions.remove("a")
            await serving.create(request("d"))
            await serving.create(request("c"))
            assert worker.sessions == {"c", "d"}
            assert worker.attempts.count("c") == 2
            assert not serving._cleanup_sessions
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("close", [False, True])
def test_cancellation_during_eviction_does_not_create_replacement(close):
    from executorch.examples.llm_server.python.serving_chat import (
        _AdmissionRetryStream,
        _AdmissionState,
    )

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        replacements = []

        class Serving:
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
            Serving(),
            "s",
            Rejected(),
            lambda: replacements.append(True),
            _AdmissionState(pending=True),
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


def test_cache_control_metadata_does_not_invalidate_replay():
    template = ChatTemplate(allow_fallback=True)
    state = OpenAITranscriptState(template)
    before = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ],
        )
    ]
    state.record_assistant_turn(
        session_id="s",
        request_messages=before,
        content="OK",
        tool_calls=None,
        generated_token_ids=[123],
        prior_turns=0,
    )
    after = [
        ChatMessage(role="user", content=[{"type": "text", "text": "hi"}]),
        ChatMessage(role="assistant", content="OK"),
    ]
    result = state.build_prompt_input(
        session_id="s",
        messages=after,
        rendered_prompt=template.render(after),
        tools=None,
        template_kwargs=None,
    )
    assert any(segment.get("ids") == [123] for segment in result.segments)


@pytest.mark.parametrize("streaming", [False, True])
def test_task_cancellation_during_native_eviction_never_restarts(streaming):
    import threading

    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_openrouter_recovery import (
        CapacityWorker,
    )

    class Worker(CapacityWorker):
        def __init__(self):
            super().__init__(True)
            self.closing = threading.Event()
            self.release_close = threading.Event()

        def close_session(self, sid):
            if sid not in self.sessions:
                raise WorkerError("missing", code="session_not_found")
            if sid == "a":
                self.closing.set()
                assert self.release_close.wait(5)
            return super().close_session(sid)

    async def scenario():
        worker = Worker()
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            evict_idle_sessions=True,
            streaming_preflight_timeout=0.001,
        )

        def request(sid, stream=False):
            return ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                session_id=sid,
                stream=stream,
            )

        async def consume():
            response = await serving.create(request("c", streaming))
            if streaming:
                async with aclosing(response):
                    async for _ in response:
                        pass

        try:
            for sid in ("a", "b"):
                await serving.create(request(sid))
            task = asyncio.create_task(consume())
            worker.admission.set()
            assert await asyncio.to_thread(worker.closing.wait, 2)
            task.cancel()
            worker.release_close.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert worker.attempts.count("c") == 1
            assert "c" not in worker.sessions
            await serving.create(request("d"))
            await serving.create(request("e"))
            assert worker.sessions == {"d", "e"}
        finally:
            worker.release_close.set()
            worker.admission.set()
            await runtime.aclose_worker()

    asyncio.run(scenario())
