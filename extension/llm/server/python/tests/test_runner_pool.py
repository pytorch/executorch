# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool tests: idle-worker scheduling, cancellation, and concurrency isolation.

The pool drives worker handles (a WorkerClient over a subprocess in production);
here we inject fakes — no model, GPU, or subprocess. Written with asyncio.run
(sync test bodies) to avoid depending on an async pytest plugin.
"""

import asyncio
import threading

import pytest

from executorch.extension.llm.server.python.runner_pool import RunnerPool


class _BlockingRunner:
    """Emits one token, then blocks until stop() is called."""

    def __init__(self):
        self._gate = threading.Event()
        self.stopped = False

    def reset(self):
        pass

    def stop(self):
        self.stopped = True
        self._gate.set()

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        if token_callback:
            token_callback("TOKEN")
        self._gate.wait(timeout=5)


class _EchoRunner:
    """Emits the prompt back as a single token; used to detect cross-talk."""

    def reset(self):
        pass

    def stop(self):
        pass

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        if token_callback:
            token_callback(prompt)


def test_pool_requires_at_least_one_worker():
    with pytest.raises(ValueError):
        RunnerPool([])


# Client disconnect / cancellation invokes the worker's stop() HOOK — the pool's
# contract. Whether that actually halts generation is up to the worker: a
# production WorkerClient.stop() is a no-op (see worker_client.py), so early
# termination comes from worker-side stop strings / EOS, not this hook. This test
# asserts only that the pool calls the hook.
def test_cancellation_calls_stop_hook():
    async def scenario():
        worker = _BlockingRunner()
        pool = RunnerPool([worker])
        async with pool.acquire() as r:
            agen = pool.generate_stream(r, "p", None).__aiter__()
            assert await agen.__anext__() == "TOKEN"  # worker now blocking
            nxt = asyncio.ensure_future(agen.__anext__())
            await asyncio.sleep(0.05)
            nxt.cancel()
            try:
                await nxt
            except asyncio.CancelledError:
                pass
            for _ in range(100):  # let the worker observe stop()
                if worker.stopped:
                    break
                await asyncio.sleep(0.02)
        assert worker.stopped

    asyncio.run(scenario())


# Concurrent requests across workers don't interleave / corrupt each other, and
# requests beyond the worker count queue for an idle worker rather than failing.
def test_concurrent_requests_isolated_and_queued():
    async def scenario():
        pool = RunnerPool([_EchoRunner(), _EchoRunner()])  # two workers

        async def one(prompt):
            async with pool.acquire() as r:
                return "".join([t async for t in pool.generate_stream(r, prompt, None)])

        # Three requests, two workers: the third queues; all echo correctly.
        out = await asyncio.gather(one("AAA"), one("BBB"), one("CCC"))
        assert sorted(out) == ["AAA", "BBB", "CCC"]

    asyncio.run(scenario())


def test_close_shuts_down_workers():
    class _Closable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    workers = [_Closable(), _Closable()]
    RunnerPool(workers).close()
    assert all(w.closed for w in workers)
