# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pool of model-execution workers + the streaming bridge.

The Python server is HTTP/control plane only: it never loads a model, links a
backend, or imports a runtime pybind. Each pooled worker is a separate process
(a C++ worker binary in production; a fake in tests) that owns one model session
and is driven over JSONL by a WorkerClient. The pool hands an idle worker to a
request and bridges the worker's blocking generate() into an async token stream.

One worker == one session; a request holds a worker exclusively for its
duration, so requests beyond the worker count queue. The number of workers is
the serving capacity, chosen by the launcher: each worker is its own process
with its own weights, so N workers cost N x the weight memory — an operator
decision, not something the pool infers.

There is no prefix cache and no prefix-affinity routing here: caching, if any,
lives inside the worker/session, not the control plane.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Sequence

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class GenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Worker-reported stop reason ("stop" | "length"), or None if not reported.
    finish_reason: Optional[str] = None


class RunnerPool:
    """A fixed pool of model-execution workers.

    `workers` is a non-empty sequence of worker handles, each exposing
    ``generate(prompt, config, token_callback, stats_callback)`` / ``stop()``
    (a WorkerClient in production; a fake in tests). The pool owns scheduling
    (one in-flight request per worker) and the blocking->async stream bridge; the
    workers own all model execution.
    """

    def __init__(self, workers: Sequence[object]):
        self._workers = list(workers)
        if not self._workers:
            raise ValueError("RunnerPool requires at least one worker")
        n = len(self._workers)
        self._busy = [False] * n
        self._cond = asyncio.Condition()
        # One executor thread per worker: generate() blocks on worker I/O, and a
        # worker is never driven by two threads at once (the busy flags enforce
        # exclusive acquisition).
        self._executor = ThreadPoolExecutor(max_workers=n)

    @asynccontextmanager
    async def acquire(self, prompt: str = ""):
        # `prompt` is accepted for call-site compatibility but unused: with no
        # prefix cache there is no affinity routing — any idle worker will do.
        del prompt
        async with self._cond:
            while all(self._busy):
                await self._cond.wait()
            idx = next(i for i, busy in enumerate(self._busy) if not busy)
            self._busy[idx] = True
        try:
            yield self._workers[idx]
        finally:
            async with self._cond:
                self._busy[idx] = False
                self._cond.notify()

    async def generate_stream(
        self,
        runner,
        prompt: str,
        config,
        stats: Optional[GenStats] = None,
    ) -> AsyncIterator[str]:
        """Yield generated text pieces. If `stats` is given it's filled in place
        with token counts (per-request, so concurrent streams don't race)."""
        out_stats = stats if stats is not None else GenStats()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def token_cb(token: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, token)

        def stats_cb(s) -> None:
            out_stats.prompt_tokens = s.num_prompt_tokens
            out_stats.completion_tokens = s.num_generated_tokens
            out_stats.finish_reason = getattr(s, "finish_reason", None)

        def run() -> None:
            try:
                runner.generate(prompt, config, token_cb, stats_cb)
            except Exception as e:  # noqa: BLE001 - surface to the stream consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        fut = loop.run_in_executor(self._executor, run)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except asyncio.CancelledError:
            runner.stop()
            raise
        finally:
            await fut

    def close(self) -> None:
        """Shut down all workers (called at server shutdown)."""
        for w in self._workers:
            close = getattr(w, "close", None)
            if close is not None:
                close()
