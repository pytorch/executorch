# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import os
from types import SimpleNamespace

import pytest
from executorch.examples.llm_server.python.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.worker_client import WorkerError

pytestmark = pytest.mark.native_worker


@pytest.fixture
def worker_path():
    path = os.environ.get("EXECUTORCH_BATCHING_TEST_WORKER")
    if not path:
        pytest.skip(
            "set EXECUTORCH_BATCHING_TEST_WORKER to the native batching fixture"
        )
    return path


async def generate(worker, session, prompt):
    stream = worker.generate(
        prompt, SimpleNamespace(session_id=session, max_new_tokens=3)
    )
    text = "".join([piece async for piece in stream])
    stats = await stream.wait()
    return text, stats


def test_restart_partial_restore_and_source_preservation(worker_path, tmp_path):
    async def run():
        worker = await spawn_multiplexed_worker([worker_path])
        try:
            await generate(worker, "seed", "abcdef")
            checkpoint = await worker.export_checkpoint_to_cpu(
                "seed", position=6, identity="test"
            )
            assert checkpoint.tokens == tuple(b"abcdef")
            save_checkpoint(checkpoint, tmp_path / "prefix.etcp")
            unchanged = await worker.export_checkpoint_to_cpu(
                "seed", position=6, identity="test"
            )
            assert unchanged == checkpoint
        finally:
            await worker.close()
        checkpoint = load_checkpoint(tmp_path / "prefix.etcp", expected_identity="test")
        worker = await spawn_multiplexed_worker([worker_path])
        try:
            await worker.restore_checkpoint_from_cpu(
                checkpoint, "restored", position=6, expected_identity="test"
            )
            actual, stats = await generate(worker, "restored", "abcdefg")
            assert (
                stats.reused_prompt_tokens == 6 and stats.prefilled_prompt_tokens == 1
            )
            await worker.restore_checkpoint_from_cpu(
                checkpoint, "partial", position=3, expected_identity="test"
            )
            _, stats = await generate(worker, "partial", "abcXYZ")
            assert (
                stats.reused_prompt_tokens == 3 and stats.prefilled_prompt_tokens == 3
            )
        finally:
            await worker.close()

    asyncio.run(asyncio.wait_for(run(), 20))


def test_failed_restore_preserves_existing_session_and_worker(worker_path):
    async def run():
        worker = await spawn_multiplexed_worker([worker_path])
        try:
            await generate(worker, "seed", "abcdef")
            checkpoint = await worker.export_checkpoint_to_cpu(
                "seed", position=6, identity="test"
            )
            with pytest.raises(WorkerError, match="unused session key"):
                await worker.restore_checkpoint_from_cpu(
                    checkpoint, "seed", position=6, expected_identity="test"
                )
            invalid = copy.deepcopy(checkpoint)
            invalid.manifest["format"] = "incompatible"
            with pytest.raises(WorkerError, match="incompatible"):
                await worker.restore_checkpoint_from_cpu(
                    invalid, "failed", position=6, expected_identity="test"
                )
            await worker.restore_checkpoint_from_cpu(
                checkpoint, "failed", position=6, expected_identity="test"
            )
            unchanged = await worker.export_checkpoint_to_cpu(
                "seed", position=6, identity="test"
            )
            assert unchanged == checkpoint and worker.healthy
        finally:
            await worker.close()

    asyncio.run(asyncio.wait_for(run(), 20))
