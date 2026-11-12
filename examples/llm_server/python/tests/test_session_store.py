# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import threading
from pathlib import Path

import pytest
from executorch.examples.llm_server.python.checkpoint import CPUCheckpoint
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.session_store import SessionStore
from executorch.examples.llm_server.python.tests.test_checkpoint import sample
from executorch.examples.llm_server.python.tests.test_multiplexed_worker_client import (
    _barrier,
    _fake_client,
)
from executorch.examples.llm_server.python.worker_client import WorkerError


def test_store_binds_artifacts_and_settings_and_contains_session_paths(tmp_path):
    async def run():
        model = tmp_path / "model"
        model.write_bytes(b"model")
        store = await SessionStore.create(
            tmp_path / "sessions", model_files=[model], configuration={"backend": "a"}
        )
        source = sample()
        checkpoint = CPUCheckpoint(store.identity, source.manifest, source.data)
        key = "../../outside/会話"
        store.save(key, checkpoint)
        assert store.load(key) == checkpoint
        assert len(list(store.directory.iterdir())) == 1
        assert not (tmp_path / "outside").exists()
        with pytest.raises(FileNotFoundError):
            store.load("missing")
        changed = await SessionStore.create(
            store.directory, model_files=[model], configuration={"backend": "b"}
        )
        with pytest.raises(ValueError, match="identity"):
            changed.load(key)
        model.write_bytes(b"new model")
        changed = await SessionStore.create(
            store.directory, model_files=[model], configuration={"backend": "a"}
        )
        with pytest.raises(ValueError, match="identity"):
            changed.load(key)

    asyncio.run(run())


def test_save_cancellation_waits_for_atomic_publication(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    original = SessionStore.save

    def blocked_save(store, session_id, checkpoint):
        started.set()
        assert release.wait(10)
        original(store, session_id, checkpoint)

    monkeypatch.setattr(SessionStore, "save", blocked_save)

    async def run():
        store = SessionStore(tmp_path, "model-a")
        async with _fake_client(session_store=store) as (client, proc):
            task = asyncio.create_task(client.save_session("chat"))
            try:
                frame = await proc.stdin.frames.get()
                assert "position" not in frame
                checkpoint = sample()
                Path(frame["path"]).write_bytes(checkpoint.data)
                proc.send(
                    frame["request_id"], exported=True, manifest=checkpoint.manifest
                )
                assert await asyncio.to_thread(started.wait, 5)
                newer = asyncio.create_task(client.save_session("chat"))
                for _ in range(2):
                    task.cancel()
                    await _barrier(client, proc)
                    assert not task.done()
                    assert proc.stdin.frames.empty()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert store.load("chat") == checkpoint
                frame = await proc.stdin.frames.get()
                checkpoint = sample((11, 12, 13, 14))
                Path(frame["path"]).write_bytes(checkpoint.data)
                proc.send(
                    frame["request_id"], exported=True, manifest=checkpoint.manifest
                )
                await newer
                assert store.load("chat") == checkpoint
                resumed = asyncio.create_task(client.resume_session("chat"))
                frame = await proc.stdin.frames.get()
                assert frame["position"] == 4
                proc.send(frame["request_id"], restored=True)
                assert await resumed == "chat"
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(asyncio.wait_for(run(), 15))


def test_failed_save_preserves_the_previous_saved_session(tmp_path):
    async def run():
        store = SessionStore(tmp_path, "model-a")
        previous = sample()
        store.save("chat", previous)
        async with _fake_client(session_store=store) as (client, proc):
            task = asyncio.create_task(client.save_session("chat"))
            frame = await proc.stdin.frames.get()
            proc.send(
                frame["request_id"], error="session busy", code="checkpoint_error"
            )
            with pytest.raises(WorkerError, match="busy"):
                await task
            assert store.load("chat") == previous

    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize("through_runtime", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_cancelled_queued_persistence_never_reaches_worker(
    tmp_path, through_runtime, resume
):
    async def run():
        store = SessionStore(tmp_path, "model-a")
        checkpoint = sample()
        store.save("other", checkpoint)
        async with _fake_client(session_store=store) as (client, proc):
            owner = SessionRuntime(client) if through_runtime else client
            active = asyncio.create_task(owner.save_session("active"))
            frame = await proc.stdin.frames.get()
            operation = owner.resume_session if resume else owner.save_session
            queued = asyncio.create_task(operation("other"))
            try:
                await _barrier(client, proc)
                queued.cancel()
                done, _ = await asyncio.wait({queued}, timeout=1)
                assert queued in done
                with pytest.raises(asyncio.CancelledError):
                    await queued
                assert proc.stdin.frames.empty()
                if through_runtime:
                    await _barrier(client, proc)
                    assert not owner.session_in_use("other")
            finally:
                Path(frame["path"]).write_bytes(checkpoint.data)
                proc.send(
                    frame["request_id"], exported=True, manifest=checkpoint.manifest
                )
                await active
            await _barrier(client, proc)
            assert proc.stdin.frames.empty()
            assert store.load("other") == checkpoint
            if through_runtime:
                await owner.aclose_worker()

    asyncio.run(asyncio.wait_for(run(), 10))


def test_runtime_retains_session_lease_after_save_waiter_cancels(tmp_path):
    async def run():
        store = SessionStore(tmp_path, "model-a")
        async with _fake_client(session_store=store) as (client, proc):
            runtime = SessionRuntime(client)
            saving = asyncio.create_task(runtime.save_session("chat"))
            frame = await proc.stdin.frames.get()
            closing = asyncio.create_task(runtime.close("chat"))
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await saving
            await _barrier(client, proc)
            assert runtime.session_in_use("chat")
            assert proc.stdin.frames.empty()
            checkpoint = sample()
            Path(frame["path"]).write_bytes(checkpoint.data)
            proc.send(frame["request_id"], exported=True, manifest=checkpoint.manifest)
            close = await proc.stdin.frames.get()
            assert close["op"] == "close"
            assert store.load("chat") == checkpoint
            proc.send(close["request_id"], closed=True, existed=True)
            assert await closing
            await runtime.aclose_worker()

    asyncio.run(asyncio.wait_for(run(), 10))


@pytest.mark.parametrize("method", ["save_session", "resume_session"])
def test_persistence_requires_explicit_storage(method):
    async def run():
        async with _fake_client() as (client, proc):
            operation = (
                client.save_session
                if method == "save_session"
                else client.resume_session
            )
            with pytest.raises(WorkerError, match="not configured"):
                await operation("chat")
            assert proc.stdin.frames.empty()

    asyncio.run(run())
