# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Request-local validation against the C++ worker and its fake executor.

Enable EXECUTORCH_LLM_SERVER_PYTHON_TESTS in the native CMake test build, or
set EXECUTORCH_TEST_MULTIPLEXED_WORKER to the built test executable for pytest.
"""

import asyncio
import json
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
)
from executorch.examples.llm_server.python.worker_client import WorkerError


@pytest.fixture(scope="module")
def native_worker():
    """Require an explicitly selected, freshly built native test executable."""
    executable = os.environ.get("EXECUTORCH_TEST_MULTIPLEXED_WORKER")
    if not executable:
        pytest.skip("set EXECUTORCH_TEST_MULTIPLEXED_WORKER to the native test target")
    assert Path(executable).is_file(), executable
    return executable


@asynccontextmanager
async def _active_peer(executable):
    loop = asyncio.get_running_loop()
    parent, child = socket.socketpair()
    parent.setblocking(False)
    proc = None
    client = None
    try:
        proc = await asyncio.create_subprocess_exec(
            executable,
            "--validation-worker",
            str(child.fileno()),
            pass_fds=(child.fileno(),),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )
        child.close()
        ready = json.loads(await proc.stdout.readline())
        assert ready["ready"] and ready["multiplexed"]
        client = MultiplexedWorkerClient(
            proc,
            max_named_sessions=ready["max_named_sessions"],
            max_inflight_requests=2,
        )
        peer = client.generate("active peer", SimpleNamespace(max_new_tokens=3))
        # Native execute() is blocked on this socket until explicitly released.
        assert await loop.sock_recv(parent, 1) == b"A"
        assert not peer._state.completion.done()
        yield client, peer, parent
    finally:
        parent.close()
        child.close()
        try:
            if client is not None:
                await client.close()
        finally:
            if proc is not None:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()


async def _finish_peer_and_reuse(client, peer, gate):
    assert not peer._state.completion.done()
    assert client.healthy and not client.failed
    await asyncio.get_running_loop().sock_sendall(gate, b"R")
    assert "".join([token async for token in peer]) == 'x\n"' * 3
    stats = await peer.wait()
    assert stats.num_generated_tokens == 3
    assert stats.finish_reason == "length" and not stats.cancelled
    # Valid scalar Unicode is escaped as a surrogate pair on the JSON wire.
    later = client.generate(
        "valid \U0001f600",
        SimpleNamespace(max_new_tokens=2, session_id="session-\U0001f600"),
    )
    assert "".join([token async for token in later]) == 'x\n"' * 2
    stats = await later.wait()
    assert stats.num_prompt_tokens == len("valid \U0001f600".encode("utf-8"))
    assert stats.num_generated_tokens == 2
    assert stats.finish_reason == "length" and not stats.cancelled
    assert client.healthy and not client.failed
    assert client._proc.returncode is None
    assert not client._requests and not client._writes and not client._controls


@pytest.mark.parametrize("reserved", [False, True])
@pytest.mark.parametrize(
    "prompt,config",
    [
        pytest.param("\ud800", {}, id="prompt-high-surrogate"),
        pytest.param("\udfff", {}, id="prompt-low-surrogate"),
        pytest.param("hi", {"max_new_tokens": 10**400}, id="huge-token-budget"),
        pytest.param("hi", {"top_k": 2**31}, id="top-k-int32-overflow"),
        pytest.param("hi", {"seed": 2**64}, id="seed-uint64-overflow"),
        pytest.param("hi", {"stop": ["\ud800"]}, id="stop-surrogate"),
        pytest.param("hi", {"session_id": "\ud800"}, id="session-surrogate"),
        pytest.param(
            "hi", {"prompt_segments": [{"text": "\ud800"}]}, id="nested-surrogate"
        ),
        pytest.param(
            "hi",
            {"prompt_segments": [{"text": "ok", "\ud800": "ignored"}]},
            id="nested-key-surrogate",
        ),
        pytest.param(
            "hi", {"prompt_segments": [{"ids": [10**400]}]}, id="huge-token-id"
        ),
        pytest.param(
            "hi",
            {"prompt_segments": [{"text": "ok", "extra": [-(2**63) - 1]}]},
            id="nested-int64-underflow",
        ),
    ],
)
def test_invalid_request_is_local_with_native_peer(
    native_worker, reserved, prompt, config
):
    """Reject parser-unsafe values before writing, without failing active work."""

    async def scenario():
        async with _active_peer(native_worker) as (client, peer, gate):
            request_id = client.reserve_request() if reserved else None
            # Observe actual stdin writes; do not replace the transport or parser.
            with patch.object(
                client._proc.stdin, "write", wraps=client._proc.stdin.write
            ) as write:
                with pytest.raises(WorkerError) as error:
                    client.generate(prompt, SimpleNamespace(**config), request_id)
                assert error.value.code == "invalid_argument"
                if reserved:
                    await client.wait_for_request(request_id)
                assert list(client._requests) == [peer.request_id]
                assert not client._writes and not client._controls
                write.assert_not_called()
            await _finish_peer_and_reuse(client, peer, gate)

    async def bounded():
        await asyncio.wait_for(scenario(), timeout=15)

    asyncio.run(bounded())


@pytest.mark.parametrize(
    "operation", ["open_session", "reset_session", "close_session"]
)
def test_invalid_lifecycle_is_local_with_native_peer(native_worker, operation):
    """Apply the same local encoding checks to session control operations."""

    async def scenario():
        async with _active_peer(native_worker) as (client, peer, gate):
            with patch.object(
                client._proc.stdin, "write", wraps=client._proc.stdin.write
            ) as write:
                with pytest.raises(WorkerError) as error:
                    await getattr(client, operation)("\ud800")
                assert error.value.code == "invalid_argument"
                assert list(client._requests) == [peer.request_id]
                assert not client._writes and not client._controls
                write.assert_not_called()
            await _finish_peer_and_reuse(client, peer, gate)

    async def bounded():
        await asyncio.wait_for(scenario(), timeout=15)

    asyncio.run(bounded())


def test_valid_non_bmp_nested_values_and_integer_boundaries(native_worker):
    """Keep native-representable Unicode and integer endpoints usable."""

    async def scenario():
        async with _active_peer(native_worker) as (client, peer, gate):
            await _finish_peer_and_reuse(client, peer, gate)
            stream = client.generate(
                "unused",
                SimpleNamespace(
                    max_new_tokens=-1,
                    top_k=2**31 - 1,
                    seed=2**64 - 1,
                    stop=["\U0001f600", "x"],
                    prompt_segments=[
                        {"text": "\U0001f600"},
                        {"ids": [0, 2**64 - 1]},
                    ],
                ),
            )
            assert [token async for token in stream] == []
            stats = await stream.wait()
            assert stats.num_prompt_tokens == 6
            assert stats.finish_reason == "stop" and not stats.cancelled
            assert client.healthy and not client.failed
            assert not client._requests

    async def bounded():
        await asyncio.wait_for(scenario(), timeout=15)

    asyncio.run(bounded())
