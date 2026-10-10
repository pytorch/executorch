# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concurrent session/runtime contracts without models or accelerator state."""

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest
from executorch.examples.llm_server.python import session_runtime as runtime_module
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
)
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    _GenerationBridge,
    GenerationOptions,
    GenStats,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.tests.test_multiplexed_worker_client import (
    _AsyncProc,
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


class _NativeWorker(MultiplexedWorkerClient):
    """Real async transport with a deterministic JSONL worker on the pipe side."""

    def __init__(
        self,
        cooperative=True,
        pair=False,
        tokens=("reply",),
        errors=None,
        ack_cancel=True,
        **limits,
    ):
        proc = _AsyncProc()
        super().__init__(
            proc, max_inflight_requests=limits.pop("max_inflight_requests", 4), **limits
        )
        self.calls = asyncio.Queue()
        self.cancelled = []
        self.cancel_frames = asyncio.Queue()
        self.ack_cancel = ack_cancel
        self._active = set()
        self._cooperative = cooperative
        self._pair = pair
        self.tokens = tokens
        self.errors = errors or {}
        self.hold_lifecycle = False
        self._lifecycle = []
        self._write_pipe = proc.stdin.write
        proc.stdin.write = self._respond

    def _respond(self, payload):
        self._write_pipe(payload)
        frame = json.loads(payload)
        request_id, op = frame["request_id"], frame["op"]
        if op == "cancel":
            target = frame["target_request_id"]
            self.cancelled.append(target)
            self.cancel_frames.put_nowait(frame)
            if self.ack_cancel:
                self._proc.send(request_id, cancelled=True)
            if self._cooperative:
                self.finish(target)
            return
        if op == "generate":
            config = SimpleNamespace(**{"session_id": None, **frame})
            self.calls.put_nowait((op, request_id, config))
            self._active.add(request_id)
            for token in self.tokens:
                self._proc.send(request_id, token=token)
        else:
            self.calls.put_nowait((op, frame["session_id"]))
        if op in self.errors:
            message, code = self.errors[op]
            self._proc.send(
                request_id,
                error=message,
                **({"code": code} if code is not None else {}),
            )
            self._active.discard(request_id)
        elif op == "generate":
            if self._pair and len(self._active) == 2:
                for active_id in tuple(self._active):
                    self.finish(active_id)
        elif self.hold_lifecycle:
            self._lifecycle.append(frame)
        else:
            self._ack_lifecycle(frame)

    def _ack_lifecycle(self, frame):
        ack = {"open": "opened", "reset": "reset", "close": "closed"}[frame["op"]]
        self._proc.send(frame["request_id"], **{ack: True})

    def release_lifecycle(self):
        self.hold_lifecycle = False
        for frame in self._lifecycle:
            self._ack_lifecycle(frame)
        self._lifecycle.clear()

    def finish(self, request_id, **metadata):
        self._active.remove(request_id)
        fields = {
            "done": True,
            "prompt_tokens": 3,
            "completion_tokens": len(self.tokens),
            "generated_token_ids": [request_id],
            "cancelled": request_id in self.cancelled,
            "finish_reason": "stop",
        }
        fields.update(metadata)
        self._proc.send(request_id, **fields)


async def _call(worker):
    if isinstance(worker, MultiplexedWorkerClient):
        return await asyncio.wait_for(worker.calls.get(), 2)
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
        worker = _NativeWorker(pair=True)
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_targeted_cancellation_keeps_other_generation_running():
    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_cancel_before_iteration_is_latched():
    async def scenario():
        worker = _NativeWorker()
        runtime = SessionRuntime(worker)
        generation = runtime.generate_stream(None, PromptInput(text="hi"), _OPTIONS)
        try:
            assert generation.cancel()
            stats = await generation.result()
            assert stats.cancelled
            assert generation.request_id is not None
            assert worker.calls.empty() and worker.cancel_frames.empty()
            assert not worker._requests
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_same_session_generation_and_lifecycle_are_serialized():
    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_cancel_timeout_retains_bounded_admission_until_real_completion():
    async def scenario():
        worker = _NativeWorker(cooperative=False)
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
            assert not worker._controls
            assert a.request_id in worker._requests
            peer = asyncio.create_task(_collect(runtime, "peer"))
            peer_call = await _call(worker)
            worker.finish(peer_call[1])
            assert await peer == ["reply"]
            assert runtime.healthy and runtime._admitted == 1
            successor = asyncio.create_task(_collect(runtime, "s"))
            await asyncio.sleep(0)
            with pytest.raises(WorkerError, match="capacity"):
                await _collect(runtime, "other")
            assert worker.calls.empty()
            assert runtime._executor is None
            worker.finish(first[1])
            second = await _call(worker)
            worker.finish(second[1])
            await successor
            await asyncio.sleep(0)
            assert not runtime._settlements
            assert not runtime._session_locks._entries
            assert runtime._admitted == 0
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_cancelled_lifecycle_keeps_same_session_exclusion():
    async def scenario():
        worker = _NativeWorker()
        worker.hold_lifecycle = True
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
            assert [state.op for state in worker._requests.values()] == ["reset"]
            worker.release_lifecycle()
            call = await _call(worker)
            worker.finish(call[1])
            await generation
            assert not runtime._session_locks._entries
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("overflow", ["count", "characters"])
def test_legacy_cross_thread_mailbox_bounds_tokens_and_scheduled_wakeups(
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


def test_legacy_terminal_has_reserved_capacity():
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
        worker = _NativeWorker()
        runtime = SessionRuntime(worker)
        operations = [
            asyncio.create_task(_collect(runtime, sid)) for sid in ("a", "b", None)
        ]
        for _ in operations:
            await _call(worker)
        await runtime.aclose_worker()
        results = await asyncio.gather(*operations, return_exceptions=True)
        assert all(isinstance(result, WorkerError) for result in results)
        assert not runtime.healthy
        assert runtime._admitted == 0
        assert not runtime._session_locks._entries

    asyncio.run(scenario())


def test_chat_transaction_covers_prepare_generation_commit_and_reset(monkeypatch):
    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_unstarted_chat_stream_close_releases_transaction():
    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("overflow", ["count", "characters"])
def test_native_mailbox_overflow_holds_lease_until_terminal_and_isolates_peer(overflow):
    async def scenario():
        limits = (
            {"mailbox_capacity": 2}
            if overflow == "count"
            else {"max_buffered_chars": 8}
        )
        worker = _NativeWorker(cooperative=False, **limits)
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
        slow = runtime.generate_stream("slow", PromptInput(text="hi"), _OPTIONS)
        try:
            assert await anext(slow) == "reply"
            first = await _call(worker)
            fast = asyncio.create_task(_collect(runtime, "fast"))
            peer = await _call(worker)
            assert peer[2].session_id == "fast"
            for token in ["x"] * 3 if overflow == "count" else ["x" * 9]:
                worker._proc.send(first[1], token=token)
            cancel = await asyncio.wait_for(worker.cancel_frames.get(), 2)
            assert cancel["target_request_id"] == first[1]
            worker.finish(peer[1])
            assert await fast == ["reply"]
            with pytest.raises(WorkerError, match="mailbox overflow") as error:
                await slow.result()
            assert error.value.code == "slow_consumer"
            assert worker.cancelled == [slow.request_id]
            assert not worker._controls  # ACK has arrived; generation has not settled.
            assert runtime._admitted == 1
            assert slow.request_id in worker._requests
            successor = asyncio.create_task(_collect(runtime, "slow"))
            await asyncio.sleep(0)
            assert worker.calls.empty() and not successor.done()
            assert list(worker._requests) == [first[1]]
            worker.finish(first[1])
            second = await _call(worker)
            worker.finish(second[1])
            assert await successor == ["reply"]
            assert not worker._requests
            assert not runtime._session_locks._entries
            assert runtime._admitted == 0 and runtime.healthy
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_native_streaming_parser_emits_before_completion_and_handles_split_stop():
    from executorch.examples.llm_server.python.protocol import DeltaMessage

    class Parser:
        def __init__(self):
            self.reasoning = True

        def feed(self, text):
            if self.reasoning:
                text, separator, content = text.partition("|")
                yield DeltaMessage(reasoning_content=text)
                self.reasoning = not separator
                text = content
            if text:
                yield DeltaMessage(content=text)

        def finish(self):
            yield DeltaMessage(content="!")

    async def scenario():
        worker = _NativeWorker(cooperative=False, tokens=("r" * 40 + "|" + "c" * 40,))
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
            "test-model",
            streaming_parser_factory=Parser,
            reasoning_extractor=lambda text: (None, "buffered fallback"),
        )
        try:
            for return_reasoning in (True, False):
                request = _request("s", stream=True)
                request.stop = ["[STOP]"]
                request.chat_template_kwargs = {"return_reasoning": return_reasoning}
                stream = await serving.create(request)
                request_id = (await _call(worker))[1]
                assert '"role":"assistant"' in await anext(stream)
                chunks = []
                fields = (
                    ("reasoning_content", "content")
                    if return_reasoning
                    else ("content",)
                )
                for field in fields:
                    chunks.append(await asyncio.wait_for(anext(stream), 2))
                    choice = json.loads(chunks[-1].removeprefix("data: "))["choices"][0]
                    assert choice["delta"][field]
                assert not worker._requests[request_id].completion.done()
                worker._proc.send(request_id, token="[ST")
                worker._proc.send(request_id, token="OP]must not leak")

                async def drain(stream=stream, chunks=chunks):
                    chunks.extend([chunk async for chunk in stream])

                draining = asyncio.create_task(drain())
                cancel = await asyncio.wait_for(worker.cancel_frames.get(), 2)
                assert cancel["target_request_id"] == request_id
                worker.finish(request_id, cancelled=False, finish_reason="length")
                await asyncio.wait_for(draining, 2)
                assert chunks[-1] == "data: [DONE]\n\n"
                choices = [
                    json.loads(chunk.removeprefix("data: "))["choices"][0]
                    for chunk in chunks[:-1]
                ]
                content = "".join(c["delta"].get("content", "") for c in choices)
                reasoning = "".join(
                    c["delta"].get("reasoning_content", "") for c in choices
                )
                assert content == "c" * 40 + "!"
                assert reasoning == ("r" * 40 if return_reasoning else "")
                assert choices[-1]["finish_reason"] == "stop"
                transcript = serving._transcript
                record = transcript._turns["s"][0]
                assert record["fp"] == transcript._assistant_fingerprint(content, None)
                assert record["reasoning_fp"] == transcript._reasoning_fingerprint(
                    reasoning or None
                )
                assert worker.cancelled.count(request_id) == 1
                assert not worker._requests and not worker._controls
                assert not serving._transactions._entries
                assert not runtime._session_locks._entries and runtime._admitted == 0
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("path", ["reasoning", "tools", "filter", "stop"])
def test_buffered_chat_disconnect_cancels_own_generation(path):
    from executorch.examples.llm_server.python.tool_parsers import HermesDetector

    async def scenario():
        worker = _NativeWorker()
        runtime = SessionRuntime(worker)
        kwargs = {}
        if path == "reasoning":
            kwargs["reasoning_extractor"] = lambda text: (None, text)
        elif path == "tools":
            kwargs["tool_detector_cls"] = HermesDetector
            kwargs["streaming_parser_factory"] = lambda: pytest.fail(
                "explicit tools must bypass the streaming parser"
            )
        elif path == "filter":
            kwargs["content_filter"] = lambda text: text
        serving = ServingChat(
            runtime,
            ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
            "test-model",
            **kwargs,
        )
        request = _request("s", stream=True)
        if path == "stop":
            request.stop = ["replytail"]
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
            generation = await _call(worker)
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert stream._reader is pending and not pending.done()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert worker.cancelled == [generation[1]]
            assert not serving._transactions._entries
            assert runtime.healthy
        finally:
            await runtime.aclose_worker()

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

    async def scenario():
        worker = _NativeWorker(
            tokens=("prefix",) if operation == "stream" else (),
            errors={op: ("rejected", code) for op in ("generate", "reset", "close")},
        )
        runtime = SessionRuntime(worker)
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_failed_multiplexed_reset_invalidates_old_transcript():
    from executorch.examples.llm_server.python.errors import GenerationError

    async def scenario():
        worker = _NativeWorker(errors={"reset": ("replacement failed", None)})
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
            await runtime.aclose_worker()

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
        supports_multiplexing = False

        def reset_session(self, session_id):
            self.calls.put(("reset", session_id))
            raise WorkerError("reset failed", code=code)

    async def scenario():
        worker = (
            _NativeWorker(errors={"reset": ("reset failed", code)})
            if multiplexed
            else ResetWorker()
        )
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
                    assert await _call(worker) == ("reset", "s")
                    assert ("s" not in serving._transcript._turns) == (
                        multiplexed or code == "session_not_found"
                    )
                    assert not serving._transactions._entries
                    assert not runtime._session_locks._entries
                    assert runtime._admitted == 0
        finally:
            if multiplexed:
                await runtime.aclose_worker()
            else:
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

    async def scenario():
        worker = _NativeWorker(
            tokens=(), errors={"generate": ("full", "capacity_exhausted")}
        )
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
            assert (await _call(worker))[0] == "generate"
            assert worker.calls.empty()
            assert not serving._transactions._entries
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_cancelling_stream_preflight_cancels_own_worker_request():
    async def scenario():
        worker = _NativeWorker(tokens=())
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_stream_preflight_keeps_first_raw_token_for_consumption():
    import json

    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_generation_close_joins_active_read_without_cancelling_peer():
    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_chat_close_joins_active_read_before_releasing_transaction():
    async def scenario():
        worker = _NativeWorker()
        runtime = SessionRuntime(worker)
        serving = _chat(runtime)
        try:
            stream = await serving.create(_request("s", stream=True))
            first = await _call(worker)
            await anext(stream)  # Role chunk; stop filtering retains the short reply.
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_sse_header_send_failure_closes_unstarted_transaction():
    from executorch.examples.llm_server.python.server import _ClosingStreamingResponse
    from starlette.requests import ClientDisconnect

    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("ack_first", [True, False])
def test_cancel_ack_order_does_not_control_same_session_lease(ack_first):
    async def scenario():
        worker = _NativeWorker(cooperative=False, ack_cancel=False)
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
        generation = runtime.generate_stream("s", PromptInput(text="hi"), _OPTIONS)
        try:
            assert await anext(generation) == "reply"
            first = await _call(worker)
            await generation.aclose()
            cancel = await asyncio.wait_for(worker.cancel_frames.get(), 2)
            if ack_first:
                worker._proc.send(cancel["request_id"], cancelled=True)
            successor = asyncio.create_task(_collect(runtime, "s"))
            # A lifecycle round trip flushes prior frames without settling s.
            await worker.open_session("barrier")
            assert await _call(worker) == ("open", "barrier")
            assert worker.calls.empty() and not successor.done()
            assert runtime._admitted == 2
            worker.finish(first[1])
            second = await _call(worker)
            worker.finish(second[1])
            assert await successor == ["reply"]
            assert runtime._admitted == 0
            if not ack_first:
                assert cancel["request_id"] in worker._controls
                worker._proc.send(cancel["request_id"], cancelled=True)
            await worker.open_session("barrier")
            assert not worker._controls and not worker._requests
            assert runtime.healthy
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_native_generation_and_lifecycle_never_construct_bridge_or_executor(
    monkeypatch,
):
    async def scenario():
        def forbidden(*args, **kwargs):
            raise AssertionError("native serving must remain on the owner loop")

        monkeypatch.setattr(runtime_module, "_GenerationBridge", forbidden)
        monkeypatch.setattr(runtime_module, "ThreadPoolExecutor", forbidden)
        monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", forbidden)
        monkeypatch.setattr(threading.Thread, "start", forbidden)
        worker = _NativeWorker(pair=True)
        runtime = SessionRuntime(worker)
        try:
            assert runtime._executor is None
            assert await asyncio.wait_for(
                asyncio.gather(_collect(runtime, "a"), _collect(runtime, "b")), 2
            ) == [["reply"], ["reply"]]
            generation = runtime.generate_stream("c", PromptInput(text="hi"), _OPTIONS)
            assert await anext(generation) == "reply"
            await generation.aclose()
            assert worker.cancelled == [generation.request_id]
            await runtime.open("a")
            await runtime.reset("a")
            await runtime.close("a")
            assert runtime.healthy
        finally:
            await runtime.aclose_worker()
        assert worker._proc.reaped
        assert worker._reader.done() and worker._writer.done()

    asyncio.run(scenario())


@pytest.mark.parametrize("prefetched", [True, False])
def test_terminal_while_consumer_paused_then_close_releases_native_capacity(prefetched):
    async def scenario():
        worker = _NativeWorker(max_inflight_requests=1)
        runtime = SessionRuntime(worker)
        generation = runtime.generate_stream("s", PromptInput(text="hi"), _OPTIONS)
        try:
            if prefetched:
                await generation.wait_ready()
            else:
                assert await anext(generation) == "reply"
            first = await _call(worker)
            worker._proc.send(first[1], token="unread")
            worker.finish(first[1])
            await worker.wait_for_request(first[1])
            with pytest.raises(WorkerError) as full:
                worker.reserve_request()
            assert full.value.code == "capacity_exhausted"
            await generation.aclose()
            assert not worker._requests
            assert not worker.cancelled
            assert not runtime._session_locks._entries
            assert runtime._admitted == 0
            replacement = asyncio.create_task(_collect(runtime, "s"))
            second = await _call(worker)
            worker.finish(second[1])
            assert await replacement == ["reply"]
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("first_token", ["", "first"])
@pytest.mark.parametrize("generated_ids", [None, [], [7, 8]])
def test_native_prefetch_preserves_tokens_and_final_metadata(
    first_token, generated_ids
):
    async def scenario():
        worker = _NativeWorker(tokens=(first_token,), mailbox_capacity=4)
        # Runtime bridge limits must not constrain the native client's mailbox.
        runtime = SessionRuntime(worker, mailbox_capacity=1, max_buffered_chars=1)
        stats = GenStats()
        generation = runtime.generate_stream(
            "s",
            PromptInput(segments=[{"text": "hi"}, {"ids": [1, 2]}]),
            GenerationOptions(max_new_tokens=8, top_p=0.75, top_k=24, seed=456),
            stats,
        )
        try:
            assert generation.request_id is None and worker.calls.empty()
            await generation.wait_ready()
            await generation.wait_ready()
            call = await _call(worker)
            assert call[2].prompt_segments == [{"text": "hi"}, {"ids": [1, 2]}]
            assert (call[2].top_p, call[2].top_k, call[2].seed) == (0.75, 24, 456)
            worker._proc.send(call[1], token="second")
            worker._proc.send(call[1], token="third")
            metadata = {
                "done": True,
                "prompt_tokens": 3,
                "completion_tokens": 3,
                "finish_reason": "length",
                "reused_prompt_tokens": 1,
                "prefilled_prompt_tokens": 2,
                "session_reset_reason": "exact_prefix",
                "prefill_ms": 4.0,
                "decode_ms": 5.0,
                "total_ms": 10.0,
                "prefill_tok_s": 750.0,
                "decode_tok_s": 400.0,
                "vision_encoder_ms": 123.5,
            }
            if generated_ids is not None:
                metadata["generated_token_ids"] = generated_ids
            worker._proc.send(call[1], **metadata)
            await worker.wait_for_request(call[1])
            assert [token async for token in generation] == [
                first_token,
                "second",
                "third",
            ]
            assert await generation.result() is stats
            assert stats == GenStats(
                prompt_tokens=3,
                completion_tokens=3,
                finish_reason="length",
                reused_prompt_tokens=1,
                prefilled_prompt_tokens=2,
                session_reset_reason="exact_prefix",
                prefill_ms=4.0,
                decode_ms=5.0,
                total_ms=10.0,
                prefill_tok_s=750.0,
                decode_tok_s=400.0,
                vision_encoder_ms=123.5,
                generated_token_ids=generated_ids,
            )
            assert not worker._requests and runtime._admitted == 0
        finally:
            await generation.aclose()
            await runtime.aclose_worker()

    asyncio.run(scenario())


def test_chat_cancelled_waiter_does_not_delete_an_owned_lock():
    async def scenario():
        worker = _NativeWorker()
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
            await runtime.aclose_worker()

    asyncio.run(scenario())
