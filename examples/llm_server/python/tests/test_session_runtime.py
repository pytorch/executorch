# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SessionRuntime tests: session-op routing, the blocking->async stream bridge,
cancellation, and worker shutdown. A fake worker stands in for the WorkerClient
(no model, GPU, or subprocess). asyncio.run keeps the test bodies sync."""

import asyncio
import logging
import threading

from executorch.examples.llm_server.python import session_runtime as session_runtime_mod
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    GenStats,
    PromptInput,
    SessionRuntime,
)

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
