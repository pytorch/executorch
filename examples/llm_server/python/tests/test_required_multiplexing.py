# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batching-required startup must never silently select a serial worker."""

import asyncio
import json
import sys

import pytest

from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.worker_client import (
    spawn_worker,
    WorkerError,
)


def _command(capabilities):
    readiness = {"ready": True, "max_named_sessions": 8, **capabilities}
    program = (
        "import sys; "
        f"print({json.dumps(readiness)!r}, flush=True); "
        "sys.stdin.read()"
    )
    return [sys.executable, "-u", "-c", program]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def readiness_worker(monkeypatch):
    children = []
    create_subprocess = asyncio.create_subprocess_exec

    async def record_process(*args, **kwargs):
        proc = await create_subprocess(*args, **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_process)

    async def launch(capabilities):
        return await asyncio.wait_for(
            spawn_multiplexed_worker(_command(capabilities)), 5
        )

    try:
        yield launch, children
    finally:
        for proc in children:
            if proc.returncode is None:
                proc.kill()
            await asyncio.wait_for(proc.wait(), 5)
            if proc.stdin is not None:
                proc.stdin.close()
                await asyncio.wait_for(proc.stdin.wait_closed(), 5)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "capabilities", [{}, {"multiplexed": False}, {"multiplexed": 1}]
)
async def test_required_multiplexing_rejects_and_reaps_legacy_worker(
    readiness_worker, capabilities
):
    launch, children = readiness_worker
    with pytest.raises(WorkerError, match="multiplexing") as caught:
        await launch(capabilities)
    assert caught.value.code == "unsupported_multiplexing"
    assert len(children) == 1
    proc = children[0]
    assert proc.returncode is not None
    assert proc.stdin.is_closing()
    assert proc.stdout.at_eof()


@pytest.mark.anyio
async def test_required_multiplexing_accepts_explicit_capability(readiness_worker):
    launch, children = readiness_worker
    client = await launch({"multiplexed": True})
    try:
        assert client.supports_multiplexing
    finally:
        await client.close()
    assert len(children) == 1
    assert children[0].returncode is not None


def test_explicit_legacy_factory_preserves_sequential_worker():
    client = spawn_worker(_command({}))
    try:
        assert not client.supports_multiplexing
    finally:
        client.close()
