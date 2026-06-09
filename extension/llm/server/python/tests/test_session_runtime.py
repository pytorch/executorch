# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SessionRuntime tests: session-op routing, the blocking->async stream bridge,
cancellation, and worker shutdown. A fake worker stands in for the WorkerClient
(no model, GPU, or subprocess). asyncio.run keeps the test bodies sync."""

import asyncio
import threading

from executorch.extension.llm.server.python.session_runtime import (
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
    assert stats.generated_token_ids == [10, 11]


def test_generate_stream_forwards_session_and_segments_to_worker():
    captured = {}

    class _Cap:
        def stop(self):
            pass

        def generate(self, prompt, config, token_callback=None, stats_callback=None):
            captured["session_id"] = config.session_id
            captured["segments"] = config.prompt_segments
            captured["prompt"] = prompt

    async def scenario():
        rt = SessionRuntime(_Cap())
        seg = PromptInput(segments=[{"text": "a"}, {"ids": [1, 2]}])
        async for _ in rt.generate_stream("sess", seg, _OPTS, GenStats()):
            pass

    asyncio.run(scenario())
    assert captured["session_id"] == "sess"
    assert captured["segments"] == [{"text": "a"}, {"ids": [1, 2]}]


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
