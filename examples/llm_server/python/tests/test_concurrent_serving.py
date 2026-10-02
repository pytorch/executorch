# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concurrent session/runtime contracts without models or accelerator state."""

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    _GenerationBridge,
    GenerationOptions,
    GenStats,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.worker_client import WorkerError, WorkerStats


_OPTIONS = GenerationOptions(max_new_tokens=8)


class _Worker:
    supports_multiplexing = True
    max_inflight_requests = 4
    healthy = True

    def __init__(self, cooperative=True, pair=False):
        self.calls = queue.Queue()
        self.cancelled = []
        self._lock = threading.Lock()
        self._next_id = 1
        self._gates = {}
        self._active = set()
        self._cooperative = cooperative
        self._pair = pair
        self.lifecycle_gate = None

    def reserve_request(self):
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._gates[request_id] = threading.Event()
            return request_id

    def release_request(self, request_id):
        with self._lock:
            return self._gates.pop(request_id, None) is not None

    def generate(self, prompt, config, token_callback, stats_callback, request_id):
        with self._lock:
            gate = self._gates[request_id]
            self._active.add(request_id)
            if self._pair and len(self._active) == 2:
                for active_id in self._active:
                    self._gates[active_id].set()
        try:
            token_callback("reply")
            self.calls.put(("generate", request_id, config))
            assert gate.wait(5), "generation never settled"
            stats_callback(
                WorkerStats(
                    num_prompt_tokens=3,
                    num_generated_tokens=1,
                    generated_token_ids=[request_id],
                    cancelled=request_id in self.cancelled,
                    finish_reason="stop",
                )
            )
        finally:
            with self._lock:
                self._active.remove(request_id)
                del self._gates[request_id]

    def finish(self, request_id):
        with self._lock:
            self._gates[request_id].set()

    def cancel(self, request_id):
        with self._lock:
            gate = self._gates.get(request_id)
            if gate is None:
                return False
            if request_id not in self.cancelled:
                self.cancelled.append(request_id)
            if self._cooperative:
                gate.set()
            return True

    def stop(self):
        raise AssertionError("multiplexed requests must never use global stop")

    def _op(self, op, session_id):
        self.calls.put((op, session_id))
        if self.lifecycle_gate is not None:
            assert self.lifecycle_gate.wait(5)

    def open_session(self, session_id):
        self._op("open", session_id)

    def reset_session(self, session_id):
        self._op("reset", session_id)

    def close_session(self, session_id):
        self._op("close", session_id)

    def close(self):
        self.healthy = False
        with self._lock:
            for gate in self._gates.values():
                gate.set()
        if self.lifecycle_gate is not None:
            self.lifecycle_gate.set()


async def _call(worker):
    return await asyncio.to_thread(worker.calls.get, True, 2)


async def _collect(runtime, session_id, stats=None):
    async with runtime.generate_stream(
        session_id, PromptInput(text="hi"), _OPTIONS, stats
    ) as generation:
        return [token async for token in generation]


def _chat(runtime):
    return ServingChat(
        runtime, ChatTemplate(hf_tokenizer_path=None, allow_fallback=True), "test-model"
    )


def _request(session_id, *, stream=False, continuation=False):
    messages = [{"role": "user", "content": "hi"}]
    if continuation:
        messages += [
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "more"},
        ]
    return ChatCompletionRequest(
        model="test-model", session_id=session_id, messages=messages, stream=stream
    )


@pytest.mark.parametrize("session_ids", [("a", "b"), (None, None)])
def test_worker_requires_b_admission_before_a_can_finish(session_ids):
    async def scenario():
        worker = _Worker(pair=True)
        runtime = SessionRuntime(worker)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(_collect(runtime, sid) for sid in session_ids)), 2
            )
            assert results == [["reply"], ["reply"]]
            calls = [await _call(worker), await _call(worker)]
            assert calls[0][1] != calls[1][1]
            assert sorted((call[2].session_id for call in calls), key=str) == sorted(
                session_ids, key=str
            )
            assert not runtime._session_locks._entries
            assert runtime._admitted == 0
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_targeted_cancellation_keeps_other_generation_running():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        a = runtime.generate_stream("a", PromptInput(text="a"), _OPTIONS)
        b = runtime.generate_stream("b", PromptInput(text="b"), _OPTIONS)
        try:
            assert await anext(a) == "reply"
            assert await anext(b) == "reply"
            await _call(worker)
            await _call(worker)
            a_id, b_id = a.request_id, b.request_id
            await a.aclose()
            assert worker.cancelled == [a_id]
            assert runtime.healthy
            pending_b = asyncio.create_task(b.result())
            await asyncio.sleep(0)
            assert not pending_b.done()
            worker.finish(b_id)
            assert (await pending_b).generated_token_ids == [b_id]
        finally:
            await a.aclose()
            await b.aclose()
            runtime.close_worker()

    asyncio.run(scenario())


def test_cancel_before_iteration_is_latched():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        generation = runtime.generate_stream(None, PromptInput(text="hi"), _OPTIONS)
        try:
            assert generation.cancel()
            stats = await generation.result()
            assert stats.cancelled
            assert worker.cancelled == [generation.request_id]
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_same_session_generation_and_lifecycle_are_serialized():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        try:
            a = asyncio.create_task(_collect(runtime, "s"))
            first = await _call(worker)
            reset = asyncio.create_task(runtime.reset("s"))
            await asyncio.sleep(0)
            next_turn = asyncio.create_task(_collect(runtime, "s"))
            other = asyncio.create_task(_collect(runtime, "other"))
            unrelated = await _call(worker)
            assert unrelated[2].session_id == "other"
            assert worker.calls.empty()
            worker.finish(unrelated[1])
            await other
            worker.finish(first[1])
            await a
            assert (await _call(worker)) == ("reset", "s")
            await reset
            second = await _call(worker)
            assert second[2].session_id == "s"
            worker.finish(second[1])
            await next_turn
            for i in range(20):
                await runtime.close(str(i))
                await _call(worker)
            assert not runtime._session_locks._entries
            assert runtime._admitted == 0
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_cancel_timeout_retains_bounded_admission_until_real_completion():
    async def scenario():
        worker = _Worker(cooperative=False)
        runtime = SessionRuntime(
            worker, max_concurrent_requests=2, cancel_grace_seconds=0.01
        )
        a = runtime.generate_stream("s", PromptInput(text="a"), _OPTIONS)
        try:
            await anext(a)
            first = await _call(worker)
            await a.aclose()
            assert runtime.healthy and runtime._admitted == 1
            assert len(runtime._settlements) == 1
            successor = asyncio.create_task(_collect(runtime, "s"))
            await asyncio.sleep(0)
            with pytest.raises(WorkerError, match="capacity"):
                await _collect(runtime, "other")
            assert worker.calls.empty()
            assert len(runtime._executor._threads) <= 2
            worker.finish(first[1])
            second = await _call(worker)
            worker.finish(second[1])
            await successor
            await asyncio.sleep(0)
            assert not runtime._settlements
            assert not runtime._session_locks._entries
            assert runtime._admitted == 0
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_cancelled_lifecycle_keeps_same_session_exclusion():
    async def scenario():
        worker = _Worker()
        worker.lifecycle_gate = threading.Event()
        runtime = SessionRuntime(worker)
        try:
            reset = asyncio.create_task(runtime.reset("s"))
            assert await _call(worker) == ("reset", "s")
            reset.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reset
            generation = asyncio.create_task(_collect(runtime, "s"))
            await asyncio.sleep(0)
            assert worker.calls.empty()
            worker.lifecycle_gate.set()
            call = await _call(worker)
            worker.finish(call[1])
            await generation
            assert not runtime._session_locks._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("overflow", ["count", "characters"])
def test_cross_thread_mailbox_bounds_tokens_and_scheduled_wakeups(
    monkeypatch, overflow
):
    async def scenario():
        worker = _Worker()
        request_id = worker.reserve_request()
        bridge = _GenerationBridge(
            worker,
            "hi",
            SimpleNamespace(),
            GenStats(),
            request_id,
            mailbox_capacity=2,
            max_buffered_chars=16,
        )
        loop = asyncio.get_running_loop()
        original = loop.call_soon_threadsafe
        scheduled = []

        def schedule(callback, *args, **kwargs):
            if callback == bridge._wake:
                scheduled.append(callback)
            return original(callback, *args, **kwargs)

        monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
        tokens = ["x"] * 1000 if overflow == "count" else ["x" * 17]

        def produce():
            for token in tokens:
                try:
                    bridge.token_cb(token)
                except WorkerError:
                    pass

        thread = threading.Thread(target=produce)
        thread.start()
        thread.join(2)  # Deliberately park the loop while the producer runs.
        assert not thread.is_alive()
        assert len(scheduled) == 1
        assert len(bridge._tokens) <= 2 and bridge._buffered_chars <= 16
        bridge.finish()
        with pytest.raises(WorkerError, match="mailbox overflow") as error:
            async for _ in bridge.items():
                pass
        assert error.value.code == "slow_consumer"
        assert worker.cancelled == [request_id]
        worker.release_request(request_id)

    asyncio.run(scenario())


def test_terminal_has_reserved_capacity():
    async def scenario():
        bridge = _GenerationBridge(
            _Worker(), "hi", SimpleNamespace(), GenStats(), 1, mailbox_capacity=2
        )
        bridge.token_cb("a")
        bridge.token_cb("b")
        bridge.finish()
        assert [token async for token in bridge.items()] == ["a", "b"]

    asyncio.run(scenario())


def test_shutdown_settles_every_active_stream():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        operations = [
            asyncio.create_task(_collect(runtime, sid)) for sid in ("a", "b", None)
        ]
        for _ in operations:
            await _call(worker)
        runtime.close_worker()
        results = await asyncio.gather(*operations, return_exceptions=True)
        assert all(isinstance(result, WorkerError) for result in results)
        assert not runtime.healthy
        assert runtime._admitted == 0
        assert not runtime._session_locks._entries

    asyncio.run(scenario())


def test_chat_transaction_covers_prepare_generation_commit_and_reset(monkeypatch):
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        events = []
        prepare = serving._transcript.build_prompt_input
        commit = serving._transcript.record_assistant_turn

        def record_prepare(**kwargs):
            events.append(("prepare", kwargs["session_id"]))
            return prepare(**kwargs)

        def record_commit(**kwargs):
            events.append(("commit", kwargs["session_id"]))
            return commit(**kwargs)

        monkeypatch.setattr(serving._transcript, "build_prompt_input", record_prepare)
        monkeypatch.setattr(serving._transcript, "record_assistant_turn", record_commit)
        try:
            first = asyncio.create_task(serving.create(_request("s")))
            call1 = await _call(worker)
            second = asyncio.create_task(
                serving.create(_request("s", continuation=True))
            )
            await asyncio.sleep(0)
            other = asyncio.create_task(serving.create(_request("other")))
            other_call = await _call(worker)
            assert other_call[0] == "generate"  # No implicit explicit open.
            assert events == [("prepare", "s"), ("prepare", "other")]
            worker.finish(other_call[1])
            await other
            reset = asyncio.create_task(serving.reset_session("s"))
            await asyncio.sleep(0)
            worker.finish(call1[1])
            await first
            call2 = await _call(worker)
            assert any(seg.get("ids") == [call1[1]] for seg in call2[2].prompt_segments)
            assert events.index(("commit", "s")) < len(events) - 1
            assert events[-1] == ("prepare", "s")
            worker.finish(call2[1])
            await second
            assert await _call(worker) == ("reset", "s")
            await reset
            assert "s" not in serving._transcript._turns
            assert not serving._transactions._entries
            assert not runtime._session_locks._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_unstarted_chat_stream_close_releases_transaction():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            stream = await serving.create(_request("s", stream=True))
            first = await _call(worker)
            close = asyncio.create_task(serving.close_session("s"))
            await asyncio.sleep(0)
            assert worker.calls.empty()
            await stream.aclose()
            assert worker.cancelled == [first[1]]
            assert await _call(worker) == ("close", "s")
            await close
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_slow_runtime_mailbox_cancels_only_its_owner():
    class BurstWorker(_Worker):
        def __init__(self):
            super().__init__()
            self.burst = threading.Event()

        def generate(self, prompt, config, token_callback, stats_callback, request_id):
            def emit(token):
                token_callback(token)
                if config.session_id == "slow":
                    assert self.burst.wait(5)
                    for _ in range(1000):
                        token_callback("spam")

            super().generate(prompt, config, emit, stats_callback, request_id)

    async def scenario():
        worker = BurstWorker()
        runtime = SessionRuntime(worker, mailbox_capacity=2)
        slow = runtime.generate_stream("slow", PromptInput(text="hi"), _OPTIONS)
        try:
            assert await anext(slow) == "reply"
            fast = asyncio.create_task(_collect(runtime, "fast"))
            peer = await _call(worker)
            assert peer[2].session_id == "fast"
            worker.burst.set()
            worker.finish(peer[1])
            assert await fast == ["reply"]
            with pytest.raises(WorkerError, match="mailbox overflow"):
                await slow.result()
            assert worker.cancelled == [slow.request_id]
            assert runtime.healthy
        finally:
            worker.burst.set()
            runtime.close_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["reasoning", "tools", "filter"])
def test_buffered_chat_disconnect_cancels_own_generation(path):
    from executorch.examples.llm_server.python.tool_parsers import HermesDetector

    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        kwargs = {}
        if path == "reasoning":
            kwargs["reasoning_extractor"] = lambda text: (None, text)
        elif path == "tools":
            kwargs["tool_detector_cls"] = HermesDetector
        else:
            kwargs["content_filter"] = lambda text: text
        serving = ServingChat(
            runtime,
            ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
            "test-model",
            **kwargs,
        )
        request = _request("s", stream=True)
        if path == "tools":
            request.tools = [
                {
                    "type": "function",
                    "function": {"name": "tool", "parameters": {"type": "object"}},
                }
            ]
        try:
            stream = await serving.create(request)
            assert '"role":"assistant"' in await anext(stream)
            pending = asyncio.create_task(anext(stream))
            generation = await _call(worker)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert worker.cancelled == [generation[1]]
            assert not serving._transactions._entries
            assert runtime.healthy
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code,status",
    [
        ("capacity_exhausted", 429),
        ("unsupported_session", 400),
        ("session_not_found", 404),
        ("session_busy", 409),
        ("not_ready", 503),
        ("invalid_argument", 400),
        ("internal", 500),
        (None, 500),
    ],
)
@pytest.mark.parametrize(
    "operation", ["complete", "stream", "stream_preflight", "reset", "close"]
)
def test_native_error_codes_survive_chat_adapter(code, status, operation):
    import json

    from executorch.examples.llm_server.python.errors import APIError

    class RejectedWorker(_Worker):
        def generate(self, *args, **kwargs):
            if operation == "stream":
                args[2]("prefix")
            raise WorkerError("rejected", code=code)

        def reset_session(self, session_id):
            raise WorkerError("rejected", code=code)

        def close_session(self, session_id):
            raise WorkerError("rejected", code=code)

    async def scenario():
        runtime = SessionRuntime(RejectedWorker())
        serving = _chat(runtime)
        try:
            if operation == "stream":
                stream = await serving.create(_request("s", stream=True))
                chunks = [chunk async for chunk in stream]
                assert chunks[-1] == "data: [DONE]\n\n"
                body = json.loads(chunks[-2].removeprefix("data: "))
                assert body["error"]["code"] == code
            elif operation == "reset" and code == "session_not_found":
                await serving.reset_session("s")
            else:
                with pytest.raises(APIError) as error:
                    if operation in ("complete", "stream_preflight"):
                        await serving.create(
                            _request("s", stream=operation == "stream_preflight")
                        )
                    else:
                        await getattr(serving, operation + "_session")("s")
                assert error.value.status == status
                assert error.value.code == code
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_failed_multiplexed_reset_invalidates_old_transcript():
    from executorch.examples.llm_server.python.errors import GenerationError

    class FailedReset(_Worker):
        def reset_session(self, session_id):
            raise WorkerError("replacement failed")

    async def scenario():
        worker = FailedReset()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        serving._transcript.record_assistant_turn(
            session_id="s",
            content="reply",
            tool_calls=None,
            generated_token_ids=[1],
            prior_turns=0,
        )
        try:
            with pytest.raises(GenerationError, match="replacement failed"):
                await serving.reset_session("s")
            assert "s" not in serving._transcript._turns
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("multiplexed", [True, False])
@pytest.mark.parametrize(
    "code,status",
    [("session_not_found", 200), ("internal", 500), ("session_busy", 409)],
)
def test_http_reset_missing_session_is_idempotent_but_preserves_real_errors(
    multiplexed, code, status
):
    import httpx
    from executorch.examples.llm_server.python.server import build_app

    class ResetWorker(_Worker):
        supports_multiplexing = multiplexed

        def reset_session(self, session_id):
            self.calls.put(("reset", session_id))
            raise WorkerError("reset failed", code=code)

    async def scenario():
        worker = ResetWorker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        serving._transcript.record_assistant_turn(
            session_id="s",
            content="reply",
            tool_calls=None,
            generated_token_ids=[1],
            prior_turns=0,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",  # @lint-ignore: in-process ASGI test URL
            ) as client:
                for _ in range(2):
                    response = await client.post("/v1/sessions/s/reset")
                    assert response.status_code == status
                    if code == "session_not_found":
                        assert response.json() == {"reset": True, "session_id": "s"}
                    else:
                        assert response.json()["error"]["code"] == code
                    assert worker.calls.get_nowait() == ("reset", "s")
                    assert ("s" not in serving._transcript._turns) == (
                        multiplexed or code == "session_not_found"
                    )
                    assert not serving._transactions._entries
                    assert not runtime._session_locks._entries
                    assert runtime._admitted == 0
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [True, False])
def test_completed_response_racing_disconnect_only_closes_stream(stream):
    from executorch.examples.llm_server.python.server import _create_with_disconnect
    from starlette.requests import ClientDisconnect, Request

    async def scenario():
        completed = asyncio.Event()
        closed = []

        class Stream:
            async def aclose(self):
                closed.append(True)

        async def create(_):
            completed.set()
            return Stream() if stream else object()

        async def receive():
            await completed.wait()
            return {"type": "http.disconnect"}

        with pytest.raises(ClientDisconnect):
            await _create_with_disconnect(
                Request({"type": "http"}, receive),
                SimpleNamespace(create=create),
                _request("s", stream=stream),
            )
        assert closed == ([True] if stream else [])

    asyncio.run(scenario())


def test_streaming_admission_failure_is_http_429_without_explicit_open():
    import httpx
    from executorch.examples.llm_server.python.server import build_app

    class FullWorker(_Worker):
        def generate(self, *args, **kwargs):
            raise WorkerError("full", code="capacity_exhausted")

    async def scenario():
        worker = FullWorker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=build_app(serving, "test-model")),
                base_url="http://test",  # @lint-ignore: in-process ASGI test URL
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json=_request("s", stream=True).model_dump(exclude_none=True),
                )
            assert response.status_code == 429
            assert response.json()["error"]["code"] == "capacity_exhausted"
            assert worker.calls.empty()
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_cancelling_stream_preflight_cancels_own_worker_request():
    class SilentWorker(_Worker):
        def generate(self, prompt, config, token_callback, stats_callback, request_id):
            super().generate(prompt, config, lambda _: None, stats_callback, request_id)

    async def scenario():
        worker = SilentWorker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            creating = asyncio.create_task(serving.create(_request("s", stream=True)))
            first = await _call(worker)
            assert not creating.done()
            creating.cancel()
            with pytest.raises(asyncio.CancelledError):
                await creating
            assert worker.cancelled == [first[1]]
            assert not serving._transactions._entries
            assert runtime._admitted == 0
            assert runtime.healthy
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_stream_preflight_keeps_first_raw_token_for_consumption():
    import json

    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            stream = await serving.create(_request("s", stream=True))
            first = await _call(worker)
            worker.finish(first[1])
            chunks = [
                json.loads(chunk.removeprefix("data: "))
                async for chunk in stream
                if "[DONE]" not in chunk
            ]
            assert (
                "".join(
                    chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
                )
                == "reply"
            )
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_generation_close_joins_active_read_without_cancelling_peer():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        generation = runtime.generate_stream("s", PromptInput(text="hi"), _OPTIONS)
        try:
            await anext(generation)
            first = await _call(worker)
            peer = asyncio.create_task(_collect(runtime, "other"))
            other = await _call(worker)
            reader = asyncio.create_task(anext(generation))
            await asyncio.sleep(0)
            assert generation._reader is reader
            await asyncio.wait_for(
                asyncio.gather(generation.aclose(), generation.aclose()), 2
            )
            with pytest.raises(asyncio.CancelledError):
                await reader
            assert worker.cancelled == [first[1]]
            worker.finish(other[1])
            assert await peer == ["reply"]
            assert runtime.healthy
            assert runtime._admitted == 0
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_chat_close_joins_active_read_before_releasing_transaction():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            stream = await serving.create(_request("s", stream=True))
            first = await _call(worker)
            await anext(stream)
            reader = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert stream._reader is reader
            lifecycle = asyncio.create_task(serving.close_session("s"))
            await asyncio.sleep(0)
            assert worker.calls.empty()
            await asyncio.wait_for(asyncio.gather(stream.aclose(), stream.aclose()), 2)
            with pytest.raises(asyncio.CancelledError):
                await reader
            assert worker.cancelled == [first[1]]
            assert await _call(worker) == ("close", "s")
            await lifecycle
            assert not serving._transactions._entries
            assert "s" not in serving._transcript._turns
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_sse_header_send_failure_closes_unstarted_transaction():
    from executorch.examples.llm_server.python.server import _ClosingStreamingResponse
    from starlette.requests import ClientDisconnect

    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        stream = await serving.create(_request("s", stream=True))
        first = await _call(worker)
        response = _ClosingStreamingResponse(stream, media_type="text/event-stream")

        async def send(_):
            raise OSError("socket disconnected before response headers")

        async def receive():
            await asyncio.Event().wait()

        try:
            with pytest.raises(ClientDisconnect):
                await response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                )
            assert worker.calls.empty()
            assert worker.cancelled == [first[1]]
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())


def test_chat_cancelled_waiter_does_not_delete_an_owned_lock():
    async def scenario():
        worker = _Worker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            stream = await serving.create(_request("s", stream=True))
            blocked = asyncio.create_task(serving.create(_request("s")))
            await asyncio.sleep(0)
            entry = serving._transactions._entries["s"]
            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            assert serving._transactions._entries["s"] is entry
            assert entry.users == 1
            await stream.aclose()
            assert not serving._transactions._entries
        finally:
            runtime.close_worker()

    asyncio.run(scenario())
