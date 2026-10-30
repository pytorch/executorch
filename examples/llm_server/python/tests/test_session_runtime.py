# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session routing, native streams, legacy bridging, cancellation, and shutdown.

Synchronous workers and real async clients with fake pipes need no model or GPU.
asyncio.run keeps the test bodies sync.
"""

import asyncio
import logging
import threading

import pytest

from executorch.examples.llm_server.python import session_runtime as session_runtime_mod
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
)
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    GenStats,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.tests.test_multiplexed_worker_client import (
    _AsyncProc,
    _barrier,
    _fake_client,
    _run,
)
from executorch.examples.llm_server.python.worker_client import WorkerError

_OPTS = GenerationOptions(max_new_tokens=8)


def _text(s="hi") -> PromptInput:
    return PromptInput(text=s)


class _Worker:
    """Records session ops + process close; emits nothing on generate."""

    def __init__(self):
        self.opened, self.reset_ids, self.closed_ids = [], [], []
        self.proc_closed = False
        self.healthy = True

    def open_session(self, sid):
        self.opened.append(sid)

    def reset_session(self, sid):
        self.reset_ids.append(sid)

    def close_session(self, sid):
        self.closed_ids.append(sid)

    def close(self):
        self.proc_closed = True

    def stop(self):
        pass

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        pass


def test_session_ops_route_to_worker():
    async def scenario():
        w = _Worker()
        rt = SessionRuntime(w)
        await rt.open("a")
        await rt.reset("a")
        await rt.close("a")
        return w

    w = asyncio.run(scenario())
    assert w.opened == ["a"] and w.reset_ids == ["a"] and w.closed_ids == ["a"]


def test_session_ops_noop_when_worker_lacks_support():
    # A minimal worker without session ops: the runtime silently no-ops.
    class _Bare:
        def stop(self):
            pass

        def generate(self, *a, **k):
            pass

    async def scenario():
        rt = SessionRuntime(_Bare())
        await rt.open("a")
        await rt.reset("a")
        await rt.close("a")

    asyncio.run(scenario())  # must not raise


def test_generate_stream_yields_and_fills_stats():
    class _Echo:
        def stop(self):
            pass

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            token_callback("Hello")
            token_callback(" world")

            class S:
                num_prompt_tokens = 3
                num_generated_tokens = 2
                finish_reason = "stop"
                prefill_ms = 4.0
                decode_ms = 5.0
                total_ms = 10.0
                prefill_tok_s = 750.0
                decode_tok_s = 400.0
                vision_encoder_ms = 123.5
                generated_token_ids = [10, 11]

            stats_callback(S())

    async def scenario():
        rt = SessionRuntime(_Echo())
        stats = GenStats()
        out = [t async for t in rt.generate_stream("a", _text(), _OPTS, stats)]
        return out, stats

    out, stats = asyncio.run(scenario())
    assert "".join(out) == "Hello world"
    assert stats.completion_tokens == 2
    assert stats.finish_reason == "stop"
    assert stats.prefill_ms == 4.0
    assert stats.decode_ms == 5.0
    assert stats.total_ms == 10.0
    assert stats.prefill_tok_s == 750.0
    assert stats.decode_tok_s == 400.0
    assert stats.vision_encoder_ms == 123.5
    assert stats.generated_token_ids == [10, 11]


def test_generate_stream_defaults_missing_vision_encoder_metric_to_none():
    class _Echo:
        def stop(self):
            pass

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            class S:
                num_prompt_tokens = 1
                num_generated_tokens = 0

            stats_callback(S())

    async def scenario():
        runtime = SessionRuntime(_Echo())
        stats = GenStats()
        async for _ in runtime.generate_stream("a", _text(), _OPTS, stats):
            pass
        return stats

    assert asyncio.run(scenario()).vision_encoder_ms is None


def test_generation_stats_log_includes_only_reported_vision_metric(caplog):
    caplog.set_level(logging.INFO)
    stats = GenStats(prompt_tokens=3, completion_tokens=2)
    ServingChat._log_generation_stats(None, stats, "stop")
    assert "vision_encoder_ms" not in caplog.messages[-1]

    stats.vision_encoder_ms = 123.5
    ServingChat._log_generation_stats(None, stats, "stop")
    assert "vision_encoder_ms=123.5" in caplog.messages[-1]


def test_generate_stream_forwards_session_and_segments_to_worker():
    captured = {}

    class _Cap:
        def stop(self):
            pass

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            captured["session_id"] = config.session_id
            captured["segments"] = config.prompt_segments
            captured["prompt"] = prompt
            captured["top_p"] = config.top_p
            captured["top_k"] = config.top_k
            captured["seed"] = config.seed

    async def scenario():
        rt = SessionRuntime(_Cap())
        seg = PromptInput(segments=[{"text": "a"}, {"ids": [1, 2]}])
        options = GenerationOptions(max_new_tokens=8, top_p=0.75, top_k=24, seed=456)
        async for _ in rt.generate_stream("sess", seg, options, GenStats()):
            pass

    asyncio.run(scenario())
    assert captured["session_id"] == "sess"
    assert captured["segments"] == [{"text": "a"}, {"ids": [1, 2]}]
    assert captured["top_p"] == 0.75
    assert captured["top_k"] == 24
    assert captured["seed"] == 456


def test_cancellation_calls_worker_stop():
    class _Blocking:
        def __init__(self):
            self._gate = threading.Event()
            self.stopped = False

        def stop(self):
            self.stopped = True
            self._gate.set()

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            token_callback("TOKEN")
            self._gate.wait(timeout=5)

    async def scenario():
        w = _Blocking()
        rt = SessionRuntime(w)
        agen = rt.generate_stream("a", _text(), _OPTS).__aiter__()
        assert await agen.__anext__() == "TOKEN"  # worker now blocking
        nxt = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)
        nxt.cancel()
        try:
            await nxt
        except asyncio.CancelledError:
            pass
        for _ in range(100):  # let the worker observe stop()
            if w.stopped:
                break
            await asyncio.sleep(0.02)
        await agen.aclose()
        return w

    w = asyncio.run(scenario())
    assert w.stopped


def test_cancellation_drops_late_worker_tokens(monkeypatch):
    class _CountingQueue(asyncio.Queue):
        put_count = 0

        def put_nowait(self, item):
            type(self).put_count += 1
            return super().put_nowait(item)

    monkeypatch.setattr(session_runtime_mod.asyncio, "Queue", _CountingQueue)

    class _SpamAfterStop:
        def __init__(self):
            self._gate = threading.Event()
            self.stopped = False

        def stop(self):
            self.stopped = True
            self._gate.set()

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            token_callback("TOKEN")
            self._gate.wait(timeout=5)
            for _ in range(1000):
                token_callback("DROP")

    async def scenario():
        _CountingQueue.put_count = 0
        w = _SpamAfterStop()
        rt = SessionRuntime(w)
        agen = rt.generate_stream("a", _text(), _OPTS).__aiter__()
        assert await agen.__anext__() == "TOKEN"
        nxt = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)
        nxt.cancel()
        try:
            await nxt
        except asyncio.CancelledError:
            pass
        await agen.aclose()
        return w.stopped, _CountingQueue.put_count

    stopped, put_count = asyncio.run(scenario())
    assert stopped
    assert put_count < 10


def test_reserves_request_before_executor_submission():
    class _Reserved(_Worker):
        def __init__(self):
            super().__init__()
            self.reserved = False
            self.request_id = None

        def reserve_request(self):
            self.reserved = True
            return 17

        def release_request(self, request_id):
            self.reserved = False
            return request_id == 17

        def generate(
            self,
            prompt,
            config,
            token_callback=None,
            stats_callback=None,
            request_id=None,
        ):
            assert self.reserved
            self.request_id = request_id

    async def scenario():
        worker = _Reserved()
        runtime = SessionRuntime(worker)
        async for _ in runtime.generate_stream(None, _text(), _OPTS):
            pass
        return worker

    worker = asyncio.run(scenario())
    assert worker.request_id == 17


def test_cooperative_cancellation_keeps_runtime_healthy():
    class _Cooperative(_Worker):
        def __init__(self):
            super().__init__()
            self._gate = threading.Event()
            self._next_id = 1
            self.abort_count = 0

        def reserve_request(self):
            request_id = self._next_id
            self._next_id += 1
            return request_id

        def release_request(self, request_id):
            return True

        def stop(self):
            self._gate.set()
            return True

        def abort(self):
            self.abort_count += 1
            self.healthy = False

        def generate(
            self,
            prompt,
            config,
            token_callback=None,
            stats_callback=None,
            request_id=None,
        ):
            token_callback("TOKEN")
            self._gate.wait(timeout=5)
            self._gate.clear()

    async def scenario():
        worker = _Cooperative()
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.5)
        generator = runtime.generate_stream(None, _text(), _OPTS)
        assert await generator.__anext__() == "TOKEN"
        await generator.aclose()
        assert runtime.healthy
        assert worker.abort_count == 0
        second = runtime.generate_stream(None, _text(), _OPTS)
        assert await second.__anext__() == "TOKEN"
        await second.aclose()
        return worker.abort_count

    assert asyncio.run(scenario()) == 0


def test_repeated_cancellation_does_not_interrupt_cleanup():
    class _SlowAbort(_Worker):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.finished = threading.Event()
            self.abort_started = threading.Event()
            self.abort_count = 0

        def reserve_request(self):
            return 1

        def release_request(self, request_id):
            return True

        def stop(self):
            return True

        def abort(self):
            self.abort_count += 1
            self.abort_started.set()
            time.sleep(0.03)
            self.healthy = False
            self.finished.set()

        def generate(
            self,
            prompt,
            config,
            token_callback=None,
            stats_callback=None,
            request_id=None,
        ):
            self.started.set()
            self.finished.wait(timeout=5)

    async def scenario():
        worker = _SlowAbort()
        runtime = SessionRuntime(
            worker, cancel_grace_seconds=0.01, abort_timeout_seconds=0.5
        )
        generator = runtime.generate_stream(None, _text(), _OPTS)
        pending = asyncio.create_task(generator.__anext__())
        await asyncio.to_thread(worker.started.wait, 1)
        pending.cancel()
        await asyncio.to_thread(worker.abort_started.wait, 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        return runtime.healthy, worker.abort_count, worker.finished.is_set()

    import time

    import pytest

    assert asyncio.run(scenario()) == (False, 1, True)


def test_stop_false_completion_race_keeps_runtime_healthy():
    class _AlreadyCompleted(_Worker):
        def __init__(self):
            super().__init__()
            self.finished = threading.Event()
            self.abort_count = 0

        def reserve_request(self):
            return 1

        def release_request(self, request_id):
            return True

        def stop(self):
            self.finished.set()
            return False

        def abort(self):
            self.abort_count += 1
            self.healthy = False

        def generate(
            self,
            prompt,
            config,
            token_callback=None,
            stats_callback=None,
            request_id=None,
        ):
            token_callback("TOKEN")
            self.finished.wait(timeout=5)

    async def scenario():
        worker = _AlreadyCompleted()
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.5)
        generator = runtime.generate_stream(None, _text(), _OPTS)
        assert await generator.__anext__() == "TOKEN"
        await generator.aclose()
        return runtime.healthy, worker.abort_count

    assert asyncio.run(scenario()) == (True, 0)


def test_noncooperative_cancellation_aborts_and_fails_fast():
    class _Noncooperative(_Worker):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.finished = threading.Event()
            self.abort_count = 0
            self._next_id = 1

        def reserve_request(self):
            request_id = self._next_id
            self._next_id += 1
            return request_id

        def release_request(self, request_id):
            return True

        def stop(self):
            return True

        def abort(self):
            self.abort_count += 1
            self.healthy = False
            self.finished.set()

        def generate(
            self,
            prompt,
            config,
            token_callback=None,
            stats_callback=None,
            request_id=None,
        ):
            self.started.set()
            self.finished.wait(timeout=5)

    async def scenario():
        worker = _Noncooperative()
        runtime = SessionRuntime(
            worker, cancel_grace_seconds=0.02, abort_timeout_seconds=0.5
        )
        generator = runtime.generate_stream(None, _text(), _OPTS)
        pending = asyncio.create_task(generator.__anext__())
        await asyncio.to_thread(worker.started.wait, 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert worker.abort_count == 1
        assert not runtime.healthy
        with pytest.raises(WorkerError, match="restart the server"):
            await anext(runtime.generate_stream(None, _text(), _OPTS))

    import pytest
    from executorch.examples.llm_server.python.worker_client import WorkerError

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["generate", "settle"])
def test_thread_interruption_reaches_consumer_and_finalizes_bridge(phase):
    from executorch.examples.llm_server.python.worker_client import WorkerError

    class InterruptedWorker(_Worker):
        supports_multiplexing = True
        max_inflight_requests = 1

        def reserve_request(self):
            return 17

        def generate(self, *args, **kwargs):
            if phase == "generate":
                raise KeyboardInterrupt(phase)

        def wait_for_request(self, request_id):
            self.waited = request_id
            if phase == "settle":
                raise KeyboardInterrupt(phase)

        def cancel(self, request_id):
            return True

    async def scenario():
        worker = InterruptedWorker()
        runtime = SessionRuntime(worker)
        generation = runtime.generate_stream("s", _text(), _OPTS)
        try:
            with pytest.raises(WorkerError, match=phase):
                await asyncio.wait_for(anext(generation), 2)
            assert worker.waited == 17
            assert generation._bridge.worker_done.is_set()
            assert generation._future.done()
            assert runtime._admitted == 0
            assert not runtime._session_locks._entries
            assert runtime.healthy
        finally:
            await generation.aclose()
            runtime.close_worker()

    asyncio.run(scenario())


def test_close_worker_shuts_down_worker():
    w = _Worker()
    SessionRuntime(w).close_worker()
    assert w.proc_closed


def test_prompt_input_requires_exactly_one():
    import pytest

    with pytest.raises(ValueError):
        PromptInput()
    with pytest.raises(ValueError):
        PromptInput(text="x", segments=[{"text": "y"}])


async def _prefetched_native(runtime, proc, session_id="s"):
    generation = runtime.generate_stream(session_id, _text(), _OPTS)
    ready = asyncio.create_task(generation.wait_ready())
    request = await proc.stdin.frames.get()
    proc.send(request["request_id"], token="first")
    await ready
    return generation


def test_native_direct_stream_preserves_prefetch_and_final_stats(monkeypatch):
    async def scenario():
        async with _fake_client() as (client, proc):

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "native runtime must not construct a bridge/executor"
                )

            monkeypatch.setattr(session_runtime_mod, "_GenerationBridge", forbidden)
            monkeypatch.setattr(session_runtime_mod, "ThreadPoolExecutor", forbidden)
            runtime = SessionRuntime(client, mailbox_capacity=1, max_buffered_chars=1)
            stats = GenStats()
            generation = runtime.generate_stream(
                "s", PromptInput(segments=[{"ids": [1, 2]}]), _OPTS, stats
            )
            assert generation.request_id is None and not client._requests
            ready = asyncio.create_task(generation.wait_ready())
            request = await proc.stdin.frames.get()
            assert request["prompt_segments"] == [{"ids": [1, 2]}]
            assert request["session_id"] == "s"
            assert runtime._executor is None and generation._bridge is None
            assert generation._stream.request_id == generation.request_id
            proc.send(generation.request_id, token="")
            await ready
            proc.send(generation.request_id, token="buffered")
            proc.send(
                generation.request_id,
                done=True,
                prompt_tokens=2,
                completion_tokens=2,
                generated_token_ids=[7, 8],
                finish_reason="length",
                reused_prompt_tokens=1,
                prefilled_prompt_tokens=1,
                session_reset_reason="exact_prefix",
                prefill_ms=12.5,
                decode_ms=25,
                total_ms=40,
                prefill_tok_s=80,
                decode_tok_s=40,
                vision_encoder_ms=123.5,
            )
            await generation._future
            assert generation in runtime._generations
            assert generation.request_id in client._requests
            assert [t async for t in generation] == ["", "buffered"]
            assert generation.stats is stats
            assert stats == GenStats(
                prompt_tokens=2,
                completion_tokens=2,
                generated_token_ids=[7, 8],
                finish_reason="length",
                reused_prompt_tokens=1,
                prefilled_prompt_tokens=1,
                session_reset_reason="exact_prefix",
                prefill_ms=12.5,
                decode_ms=25,
                total_ms=40,
                prefill_tok_s=80,
                decode_tok_s=40,
                vision_encoder_ms=123.5,
            )
            assert not client._requests and not runtime._generations
            await runtime.aclose_worker()

    _run(scenario())


def test_native_exhaustion_waits_for_stats_owner(monkeypatch):
    async def scenario():
        async with _fake_client() as (client, proc):
            runtime = SessionRuntime(client)
            terminal = runtime._native_terminal
            allowed = asyncio.Event()
            entered = asyncio.Event()

            async def delayed_terminal(stream, stats):
                entered.set()
                await allowed.wait()
                await terminal(stream, stats)

            monkeypatch.setattr(runtime, "_native_terminal", delayed_terminal)
            generation = await _prefetched_native(runtime, proc)
            await entered.wait()
            assert await anext(generation) == "first"
            proc.send(generation.request_id, done=True, completion_tokens=1)
            finished = asyncio.create_task(generation.result())
            await _barrier(client, proc)
            assert not finished.done() and runtime._admitted == 1
            allowed.set()
            stats = await finished
            assert stats is generation.stats and stats.completion_tokens == 1
            assert runtime._admitted == 0
            await runtime.aclose_worker()

    _run(scenario())


def test_native_precancel_and_unused_reservation_cleanup(monkeypatch):
    async def scenario():
        async with _fake_client(max_inflight_requests=1) as (client, proc):
            runtime = SessionRuntime(client)
            generation = runtime.generate_stream("s", _text(), _OPTS)
            assert generation.cancel()
            assert [t async for t in generation] == []
            assert generation.stats.cancelled
            assert generation.request_id is not None
            assert not proc.stdin.written and not client._requests
            failure = WorkerError("submission failed", code="invalid_argument")

            def fail(*args, **kwargs):
                raise failure

            monkeypatch.setattr(client, "generate", fail)
            invalid = runtime.generate_stream("s", _text(), _OPTS)
            with pytest.raises(WorkerError) as caught:
                await invalid.wait_ready()
            assert caught.value is failure
            assert not client._requests and runtime._admitted == 0
            assert not runtime._generations
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("wire_error", [False, True])
def test_native_closing_completed_unconsumed_stream_retires_capacity(wire_error):
    async def scenario():
        async with _fake_client(max_inflight_requests=1) as (client, proc):
            runtime = SessionRuntime(client)
            generation = await _prefetched_native(runtime, proc)
            proc.send(generation.request_id, token="unconsumed")
            if wire_error:
                proc.send(generation.request_id, error="terminal failure", code="bad")
                with pytest.raises(WorkerError):
                    await generation._future
            else:
                proc.send(generation.request_id, done=True)
                await generation._future
            assert generation.request_id in client._requests
            await generation.aclose()
            assert not client._requests and not runtime._generations
            assert runtime._admitted == 0 and not runtime._session_locks._entries
            replacement = client.reserve_request()
            assert client.release_request(replacement)
            await runtime.aclose_worker()

    _run(scenario())


def test_native_cancel_ack_and_cancelled_settlement_keep_session_lease():
    async def scenario():
        async with _fake_client() as (client, proc):
            runtime = SessionRuntime(client, cancel_grace_seconds=0)
            generation = await _prefetched_native(runtime, proc)
            await generation.aclose()
            cancel = await proc.stdin.frames.get()
            assert cancel["target_request_id"] == generation.request_id
            proc.send(cancel["request_id"], cancelled=True)
            await _barrier(client, proc)
            assert not generation._future.done()
            assert runtime._admitted == 1
            settlement = next(iter(runtime._settlements))
            settlement.cancel()
            reset = asyncio.create_task(runtime.reset("s"))
            await _barrier(client, proc)
            assert not reset.done() and proc.stdin.frames.empty()
            assert not settlement.done()
            assert not generation._future.cancelled()
            proc.send(
                generation.request_id, done=True, finish_reason="stop", cancelled=True
            )
            request = await proc.stdin.frames.get()
            assert request["op"] == "reset"
            proc.send(request["request_id"], reset=True)
            await reset
            assert generation.stats.cancelled
            assert not runtime._session_locks._entries and runtime._admitted == 0
            assert not proc.signals and client.healthy
            await runtime.aclose_worker()

    _run(scenario())


@pytest.mark.parametrize("terminal_first", [False, True])
def test_native_local_error_preserved_while_wire_owns_completion(terminal_first):
    async def scenario():
        async with _fake_client(mailbox_capacity=1) as (client, proc):
            runtime = SessionRuntime(client, cancel_grace_seconds=0)
            generation = await _prefetched_native(runtime, proc)
            assert await anext(generation) == "first"
            proc.send(generation.request_id, token="one")
            proc.send(generation.request_id, token="overflow")
            cancel = await proc.stdin.frames.get()
            proc.send(cancel["request_id"], cancelled=True)
            if terminal_first:
                proc.send(
                    generation.request_id, error="secondary wire failure", code="wire"
                )
                with pytest.raises(WorkerError):
                    await generation._future
            with pytest.raises(WorkerError) as caught:
                await anext(generation)
            assert caught.value.code == "slow_consumer"
            if not terminal_first:
                assert not generation._future.done() and runtime._admitted == 1
                proc.send(
                    generation.request_id, error="secondary wire failure", code="wire"
                )
                with pytest.raises(WorkerError):
                    await generation._future
            await runtime.aclose_worker()
            assert not client._requests and not runtime._settlements

    _run(scenario())


@pytest.mark.parametrize(
    "method,ack", [("open", "opened"), ("reset", "reset"), ("close", "closed")]
)
def test_native_lifecycle_caller_cancellation_does_not_cancel_owner(method, ack):
    async def scenario():
        async with _fake_client() as (client, proc):
            runtime = SessionRuntime(client)
            caller = asyncio.create_task(getattr(runtime, method)("s"))
            request = await proc.stdin.frames.get()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            following = asyncio.create_task(runtime.reset("s"))
            await _barrier(client, proc)
            assert not following.done() and runtime._admitted == 2
            proc.send(request["request_id"], **{ack: True})
            next_request = await proc.stdin.frames.get()
            assert next_request["op"] == "reset"
            proc.send(next_request["request_id"], reset=True)
            await following
            await runtime.aclose_worker()
            assert not runtime._native_tasks and not runtime._settlements
            assert runtime._admitted == 0

    _run(scenario())


@pytest.mark.parametrize(
    "phase",
    ["lazy", "prefetch_read", "active_read", "prefetched", "suspended", "wire_done"],
)
def test_native_shutdown_joins_every_generation_phase(phase):
    async def scenario():
        proc = _AsyncProc()
        client = MultiplexedWorkerClient(proc)
        runtime = SessionRuntime(client)
        generation = runtime.generate_stream("s", _text(), _OPTS)
        reader = None
        if phase != "lazy":
            reader = asyncio.create_task(
                anext(generation) if phase == "active_read" else generation.wait_ready()
            )
            request = await proc.stdin.frames.get()
            if phase not in ("prefetch_read", "active_read"):
                proc.send(request["request_id"], token="first")
                await reader
                if phase == "suspended":
                    assert await anext(generation) == "first"
                if phase == "wire_done":
                    proc.send(request["request_id"], token="unread")
                    proc.send(request["request_id"], done=True)
                    await generation._future
                    assert generation in runtime._generations
        await runtime.aclose_worker()
        shutdown = runtime._shutdown_task
        await runtime.aclose_worker()
        assert runtime._shutdown_task is shutdown
        if reader is not None:
            await asyncio.gather(reader, return_exceptions=True)
        assert proc.reaped and proc.wait_count == 1
        assert not runtime._generations and not runtime._native_tasks
        assert not runtime._settlements and runtime._admitted == 0
        assert not runtime._session_locks._entries
        assert not client._requests and not runtime.healthy
        if phase == "lazy":
            assert not proc.stdin.written
            with pytest.raises(WorkerError, match="closed"):
                await anext(generation)
        else:
            assert generation._closed
            assert generation._stream._state.consumed
            if generation._prefetch is not None:
                assert generation._prefetch.done()
        await generation.aclose()

    _run(scenario())


def test_native_shutdown_joins_queued_generations_and_lifecycle_settlement():
    async def scenario():
        proc = _AsyncProc()
        client = MultiplexedWorkerClient(proc)
        runtime = SessionRuntime(client, cancel_grace_seconds=0)
        active = await _prefetched_native(runtime, proc)
        queued = runtime.generate_stream("s", _text(), _OPTS)
        ready = asyncio.create_task(queued.wait_ready())
        lifecycle = asyncio.create_task(runtime.open("other"))
        request = await proc.stdin.frames.get()
        assert request["op"] == "open"
        lifecycle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await lifecycle
        assert runtime._admitted == 3
        await runtime.aclose_worker()
        await asyncio.gather(ready, return_exceptions=True)
        assert active._closed and queued._closed
        assert not runtime._generations and not runtime._native_tasks
        assert not runtime._settlements and runtime._admitted == 0
        assert not runtime._session_locks._entries and proc.reaped

    _run(scenario())


def test_native_shutdown_waits_for_all_queued_lifecycle_admissions():
    async def scenario():
        async with _fake_client() as (client, proc):
            runtime = SessionRuntime(client)
            await _prefetched_native(runtime, proc)
            waiters = [
                asyncio.create_task(operation("s"))
                for operation in (runtime.open, runtime.reset, runtime.close) * 4
            ]
            await _barrier(client, proc)
            assert runtime._admitted == 13
            await runtime.aclose_worker()
            assert not runtime._operations and runtime._admitted == 0
            assert not runtime._session_locks._entries
            assert all(waiter.done() for waiter in waiters)
            errors = await asyncio.gather(*waiters, return_exceptions=True)
            assert all(isinstance(error, WorkerError) for error in errors)

    _run(scenario())


@pytest.mark.parametrize("code", ["capacity_exhausted", "invalid_argument"])
def test_native_wait_ready_preserves_worker_error_code(code):
    async def scenario():
        async with _fake_client() as (client, proc):
            runtime = SessionRuntime(client)
            generation = runtime.generate_stream("s", _text(), _OPTS)
            ready = asyncio.create_task(generation.wait_ready())
            request = await proc.stdin.frames.get()
            proc.send(request["request_id"], error="preheader error", code=code)
            with pytest.raises(WorkerError) as caught:
                await ready
            assert caught.value.code == code
            assert not client._requests and not runtime._generations
            assert runtime._admitted == 0
            await runtime.aclose_worker()

    _run(scenario())


def test_native_shutdown_repeated_cancellation_waits_for_reap_and_is_shared():
    async def scenario():
        proc = _AsyncProc()
        client = MultiplexedWorkerClient(proc)
        runtime = SessionRuntime(client)
        await _prefetched_native(runtime, proc)
        proc.reap_allowed.clear()
        caller = asyncio.create_task(runtime.aclose_worker())
        await proc.wait_started.wait()
        owner = runtime._shutdown_task
        peer = asyncio.create_task(runtime.aclose_worker())
        for _ in range(3):
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done() and not owner.cancelled()
            assert not proc.reaped
        proc.reap_allowed.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await peer
        assert runtime._shutdown_task is owner
        assert proc.reaped and proc.wait_count == 1
        assert not runtime._generations and not runtime._native_tasks

    _run(scenario())


def test_native_shutdown_reap_failure_surfaces_and_retries():
    async def scenario():
        proc = _AsyncProc()
        client = MultiplexedWorkerClient(proc)
        runtime = SessionRuntime(client)
        await _prefetched_native(runtime, proc)
        proc.wait_failures = 2
        with pytest.raises(WorkerError, match="could not be reaped"):
            await runtime.aclose_worker()
        failed_owner = runtime._shutdown_task
        assert not proc.reaped and not runtime._generations
        assert not runtime._native_tasks and runtime._admitted == 0
        await runtime.aclose_worker()
        assert runtime._shutdown_task is not failed_owner
        assert proc.reaped and proc.wait_count == 3
        await runtime.aclose_worker()
        assert proc.wait_count == 3

    _run(scenario())


def test_shutdown_api_rejects_native_sync_and_supports_legacy_async():
    async def scenario():
        async with _fake_client() as (client, proc):
            runtime = SessionRuntime(client)
            with pytest.raises(WorkerError, match="await runtime.aclose_worker"):
                runtime.close_worker()
            assert runtime.healthy
            await runtime.aclose_worker()
        legacy = _Worker()
        runtime = SessionRuntime(legacy)
        await runtime.aclose_worker()
        assert legacy.proc_closed

    _run(scenario())
