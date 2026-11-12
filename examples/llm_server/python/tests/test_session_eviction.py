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
@pytest.mark.parametrize("code", ["invalid_argument", "capacity_exhausted"])
def test_unconfirmed_native_admission_cleanup_is_retried(opened, code):
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
                raise WorkerError("generation failed before output", code=code)
            return super().generate(
                prompt, config, token_callback, stats_callback, request_id
            )

        def close_session(self, sid):
            if self.fail_close:
                self.fail_close = False
                raise WorkerError("close queue full", code="capacity_exhausted")
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

    async def scenario():
        worker = CapacityWorker(False)
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
        ServingChat,
    )

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        replacements = []

        class Serving:
            _retry_admission = ServingChat._retry_admission

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
        await asyncio.wait_for(entered.wait(), 2)
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
    unchanged = [u1, answer, u2, answer, ChatMessage(role="user", content="continue")]
    warm = state.build_prompt_input(
        session_id="s",
        messages=unchanged,
        rendered_prompt=template.render(unchanged),
        tools=None,
        template_kwargs=None,
    )
    assert [segment["ids"] for segment in warm.segments if "ids" in segment] == [
        [100],
        [101],
    ]
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
    unchanged = [*history, ChatMessage(role="assistant", content="OK")]
    warm = state.build_prompt_input(
        session_id="s",
        messages=unchanged,
        rendered_prompt=template.render(unchanged),
        tools=None,
        template_kwargs=None,
    )
    assert any(segment.get("ids") == [123] for segment in warm.segments)
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


@pytest.mark.parametrize("runtime_lease", [False, True])
def test_idle_eviction_protects_active_transactions(make_client, runtime_lease):
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
        locks = (
            serving._runtime._session_locks if runtime_lease else serving._transactions
        )
        async with locks.hold("a"):
            await serving.create(request("c"))
            assert worker.open_named == {"a", "c"}
            async with locks.hold("c"):
                assert not await serving._evict_idle_session()

    asyncio.run(scenario())


@pytest.mark.parametrize("before_string", [False, True])
@pytest.mark.parametrize("after_string", [False, True])
def test_cache_control_metadata_does_not_invalidate_replay(before_string, after_string):
    template = ChatTemplate(allow_fallback=True)
    state = OpenAITranscriptState(template)
    before = [
        ChatMessage(
            role="user",
            content=(
                "hi"
                if before_string
                else [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
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
        ChatMessage(
            role="user",
            content="hi" if after_string else [{"type": "text", "text": "hi"}],
        ),
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


@pytest.mark.parametrize("streaming", [False, True])
def test_task_cancellation_during_refused_generation_close(monkeypatch, streaming):
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import (
        Generation,
        SessionRuntime,
    )
    from executorch.examples.llm_server.python.tests.test_openrouter_recovery import (
        CapacityWorker,
    )

    async def scenario():
        worker = CapacityWorker(True)
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test",
            evict_idle_sessions=True,
            streaming_preflight_timeout=0,
        )
        closing, release = asyncio.Event(), asyncio.Event()
        original_close = Generation._close

        async def gated_close(generation):
            await original_close(generation)
            if worker.attempts == ["a", "b", "c"]:
                closing.set()
                await release.wait()

        monkeypatch.setattr(Generation, "_close", gated_close)

        def request(sid):
            return ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                session_id=sid,
                stream=streaming and sid == "c",
            )

        async def consume():
            response = await serving.create(request("c"))
            if streaming:
                async with aclosing(response):
                    async for _ in response:
                        pass

        try:
            for sid in ("a", "b"):
                await serving.create(request(sid))
            task = asyncio.create_task(consume())
            worker.admission.set()
            await asyncio.wait_for(closing.wait(), 2)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert worker.attempts == ["a", "b", "c"]
            assert worker.closed_sessions == []
            assert worker.sessions == {"a", "b"}
            assert not serving._cleanup_sessions
        finally:
            release.set()
            worker.admission.set()
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_local_preadmission_overload_does_not_accumulate_cleanup():
    from executorch.examples.llm_server.python.errors import SessionCapacity
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
        _NativeWorker,
        _request,
    )

    async def scenario():
        worker = _NativeWorker(tokens=())
        runtime = SessionRuntime(worker, max_concurrent_requests=1)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test-model",
            evict_idle_sessions=True,
        )
        try:
            async with runtime._operation("busy"):
                for i in range(20):
                    with pytest.raises(SessionCapacity):
                        await serving.create(_request(f"refused-{i}"))
                    assert not serving._cleanup_sessions
                assert worker.calls.empty()
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_pending_cleanup_does_not_wait_for_unsettled_native_prefill():
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
        _call,
        _NativeWorker,
        _request,
    )

    async def scenario():
        worker = _NativeWorker(tokens=(), cooperative=False)
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
        serving = ServingChat(
            runtime,
            ChatTemplate(allow_fallback=True),
            "test-model",
            evict_idle_sessions=True,
            streaming_preflight_timeout=0,
        )
        closing = None
        request_id = None
        try:
            stream = await serving.create(_request("s", stream=True))
            request_id = (await _call(worker))[1]
            closing = asyncio.create_task(stream.aclose())
            done, _ = await asyncio.wait([closing], timeout=0.5)
            assert closing in done, "teardown waited for unsettled native prefill"
            await closing
            assert runtime.session_in_use("s")
            assert list(serving._cleanup_sessions) == ["s"]
            assert not serving._transactions.in_use("s")
            assert worker.calls.empty()
            assert not await serving._evict_idle_session()
            worker.finish(request_id)
            request_id = None
            await asyncio.gather(*runtime._settlements)
            assert await serving._evict_idle_session()
            assert await _call(worker) == ("close", "s")
            assert not serving._cleanup_sessions
        finally:
            if request_id is not None:
                worker.finish(request_id)
            if closing is not None:
                await closing
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_partial_decode_error_preserves_admitted_session(stream):
    from executorch.examples.llm_server.python.errors import APIError
    from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
    from executorch.examples.llm_server.python.serving_chat import ServingChat
    from executorch.examples.llm_server.python.session_runtime import SessionRuntime
    from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
        _NativeWorker,
    )

    async def scenario():
        worker = _NativeWorker(
            tokens=("partial",), errors={"generate": ("decode failed", "internal")}
        )
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime, ChatTemplate(allow_fallback=True), "test", evict_idle_sessions=True
        )
        try:
            req = ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                session_id="new",
                stream=stream,
            )
            if stream:
                result = await serving.create(req)
                output = "".join([chunk async for chunk in result])
                assert '"error"' in output
            else:
                with pytest.raises(APIError):
                    await serving.create(req)
            assert list(serving._idle_sessions) == ["new"]
            assert not serving._cleanup_sessions
            calls = []
            while not worker.calls.empty():
                calls.append(worker.calls.get_nowait()[0])
            assert calls == ["generate"]  # Neither response mode closes the session.
        finally:
            await runtime.aclose_worker()

    asyncio.run(asyncio.wait_for(scenario(), 5))
