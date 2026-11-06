# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from dataclasses import replace

import pytest
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.tests.conftest import FakeRunner
from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
    _call,
    _NativeWorker,
    _Worker,
)
from executorch.examples.llm_server.python.worker_client import WorkerError


class _Observer:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.first_output = asyncio.Event()
        self.terminal = asyncio.Event()
        self.arrivals = []
        self.stats = []

    def output(self, arrived):
        assert asyncio.get_running_loop() is self.loop
        self.arrivals.append(arrived)
        self.first_output.set()

    def record_stats(self, stats):
        assert asyncio.get_running_loop() is self.loop
        assert stats.terminal_received
        self.stats.append(replace(stats))
        self.terminal.set()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_observer_receives_terminal_stats_once_without_consumption(native, cancel):
    async def scenario():
        worker = (
            _NativeWorker(cooperative=False) if native else _Worker(cooperative=False)
        )
        runtime = SessionRuntime(worker, cancel_grace_seconds=0.01)
        observer = _Observer()
        generation = runtime.generate_stream(
            None,
            PromptInput(text="hi"),
            GenerationOptions(max_new_tokens=8),
            observer=observer,
        )
        try:
            await asyncio.wait_for(generation.wait_ready(), 2)
            _, request_id, _ = await _call(worker)
            await asyncio.wait_for(observer.first_output.wait(), 2)
            assert len(observer.arrivals) == 1
            assert not observer.stats
            if cancel:
                await asyncio.wait_for(generation.aclose(), 2)
                assert not observer.stats
            worker.finish(request_id)
            await asyncio.wait_for(observer.terminal.wait(), 2)
            assert len(observer.stats) == 1
            assert observer.stats[0].completion_tokens == 1
            assert observer.stats[0].cancelled is cancel
            await generation.result()
            await generation.aclose()
            assert len(observer.stats) == 1
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())


@pytest.mark.parametrize("native", [False, True])
def test_observer_does_not_replace_worker_errors_or_report_missing_stats(native):
    async def scenario():
        worker = (
            _NativeWorker(tokens=(), errors={"generate": ("denied", "unsupported")})
            if native
            else FakeRunner([], fail=True)
        )
        runtime = SessionRuntime(worker)
        observer = _Observer()
        try:
            async with runtime.generate_stream(
                None,
                PromptInput(text="hi"),
                GenerationOptions(max_new_tokens=8),
                observer=observer,
            ) as generation:
                with pytest.raises(
                    WorkerError if native else RuntimeError,
                    match="denied" if native else "Generation failed",
                ):
                    await generation.result()
            assert not observer.arrivals
            assert not observer.stats
        finally:
            await runtime.aclose_worker()

    asyncio.run(scenario())
