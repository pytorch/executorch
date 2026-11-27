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
from executorch.examples.llm_server.python.session_runtime import (
    GenerationOptions,
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.session_store import SessionStore
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
            await worker.restore_checkpoint_from_cpu(
                checkpoint, "retry", position=5, expected_identity="test"
            )
            _, stats = await generate(worker, "retry", "abcdef")
            assert (
                stats.reused_prompt_tokens == 5 and stats.prefilled_prompt_tokens == 1
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


def test_save_close_restart_and_resume_without_checkpoint_details(
    worker_path, tmp_path
):
    async def run():
        store = await SessionStore.create(
            tmp_path / "sessions",
            model_files=[worker_path],
            configuration={"backend": "test"},
        )
        worker = await spawn_multiplexed_worker([worker_path], session_store=store)
        runtime = SessionRuntime(worker)
        try:
            stream = runtime.generate_stream(
                "chat", PromptInput(text="abcdef"), GenerationOptions(max_new_tokens=3)
            )
            async for _ in stream:
                pass
            await runtime.save_session("chat")
            saved = store.load("chat")
            assert saved.tokens == tuple(b"abcdef") + tuple(
                stream.stats.generated_token_ids[:-1]
            )
            await runtime.close("chat")
        finally:
            await runtime.aclose_worker()
        worker = await spawn_multiplexed_worker([worker_path], session_store=store)
        runtime = SessionRuntime(worker)
        try:
            session = await runtime.resume_session("chat")
            with pytest.raises(WorkerError, match="unused session key"):
                await runtime.resume_session("chat")
            tokens = list(b"abcdef") + stream.stats.generated_token_ids + list(b"next")
            prompt = PromptInput(segments=[{"ids": tokens}])
            restored = runtime.generate_stream(
                session, prompt, GenerationOptions(max_new_tokens=3)
            )
            cold = runtime.generate_stream(
                "cold", prompt, GenerationOptions(max_new_tokens=3)
            )
            async for _ in restored:
                pass
            async for _ in cold:
                pass
            assert cold.stats.reused_prompt_tokens == 0
            assert cold.stats.prefilled_prompt_tokens == len(tokens)
            assert restored.stats.reused_prompt_tokens == len(saved.tokens)
            assert restored.stats.prefilled_prompt_tokens == 5
            await runtime.save_session("chat")
            assert len(store.load("chat").tokens) > len(saved.tokens)
            with pytest.raises(FileNotFoundError):
                await runtime.resume_session("missing")
            assert runtime.healthy
        finally:
            await runtime.aclose_worker()

    asyncio.run(asyncio.wait_for(run(), 20))


@pytest.mark.parametrize("outcome", ["reset", "stop_string", "cancel"])
def test_unsaveable_turn_preserves_saved_state_until_clean_turn(
    worker_path, tmp_path, outcome
):
    async def run():
        store = SessionStore(tmp_path, "test")
        worker = await spawn_multiplexed_worker([worker_path], session_store=store)
        try:
            await generate(worker, "chat", "abcdef")
            await worker.save_session("chat")
            before = store.load("chat")
            if outcome == "reset":
                await worker.reset_session("chat")
            else:
                stream = worker.generate(
                    "next turn",
                    SimpleNamespace(
                        session_id="chat",
                        max_new_tokens=512,
                        stop=(
                            list("abcdefghijklmnopqrstuvwxyz")
                            if outcome == "stop_string"
                            else []
                        ),
                    ),
                )
                async for _ in stream:
                    if outcome == "cancel":
                        assert stream.cancel()
                        break
                await stream.wait()
            with pytest.raises(WorkerError, match="committed history"):
                await worker.save_session("chat")
            assert store.load("chat") == before
            await generate(worker, "chat", "clean replacement turn")
            await worker.save_session("chat")
            assert store.load("chat") != before
        finally:
            await worker.close()

    asyncio.run(asyncio.wait_for(run(), 20))
