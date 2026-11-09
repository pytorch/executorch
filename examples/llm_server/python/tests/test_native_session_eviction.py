# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session-slot recovery through the real C++ worker and JSONL client."""

import asyncio
import json
from unittest.mock import patch

import pytest
from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime


def request(sid=None, stream=False):
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        session_id=sid,
        max_tokens=1,
        stream=stream,
    )


@pytest.mark.parametrize("stream,timeout", [(False, None), (True, None), (True, 0)])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("sid", [None, "new"])
def test_native_slot_refusal_evicts_and_retries(
    active_peer, stream, timeout, stale, sid
):
    async def scenario():
        async with active_peer() as (worker, peer, gate):
            await asyncio.get_running_loop().sock_sendall(gate, b"R")
            async for _ in peer:
                pass
            await peer.wait()
            runtime = SessionRuntime(worker)
            serving = ServingChat(
                runtime,
                ChatTemplate(allow_fallback=True),
                "test",
                evict_idle_sessions=True,
                streaming_preflight_timeout=timeout,
            )
            assert runtime._native
            for existing in ("a", "b", "c", "d"):
                await serving.create(request(existing))
            if stale:
                assert await worker.close_session("a") is True
                await worker.open_session("outside")
            with patch.object(
                worker._proc.stdin, "write", wraps=worker._proc.stdin.write
            ) as write:
                result = await serving.create(request(sid, stream))
                if stream:
                    output = "".join([chunk async for chunk in result])
                    assert '"error"' not in output
                    chunks = [
                        json.loads(line[6:])
                        for line in output.splitlines()
                        if line.startswith("data: {")
                    ]
                    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
                else:
                    assert result.choices[0].message.content == 'x\n"'
                    assert result.choices[0].finish_reason == "length"
            frames = [json.loads(call.args[0]) for call in write.call_args_list]
            assert [frame["op"] for frame in frames] == (
                ["generate", "close", "close", "generate"]
                if stale
                else ["generate", "close", "generate"]
            )
            assert [
                frame["session_id"] for frame in frames if frame["op"] == "close"
            ] == (["a", "b"] if stale else ["a"])
            expected = (["c", "d"] if stale else ["b", "c", "d"]) + (
                [sid] if sid else []
            )
            assert list(serving._idle_sessions) == expected
            assert not serving._cleanup_sessions
            assert not runtime._admitted
            assert await worker.close_session("a") is False
            if stale:
                assert await worker.close_session("b") is False
            if sid is None:
                # Anonymous generation released its temporary slot. Filling it
                # directly succeeds without evicting another named session.
                await worker.open_session("temporary-slot-was-released")
                assert list(serving._idle_sessions) == expected
            assert worker.healthy

    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_local_close_overload_preserves_native_lru(active_peer):
    async def scenario():
        async with active_peer() as (worker, peer, gate):
            await asyncio.get_running_loop().sock_sendall(gate, b"R")
            async for _ in peer:
                pass
            await peer.wait()
            runtime = SessionRuntime(worker, max_concurrent_requests=1)
            serving = ServingChat(
                runtime,
                ChatTemplate(allow_fallback=True),
                "test",
                evict_idle_sessions=True,
            )
            for sid in ("a", "b"):
                await serving.create(request(sid))
            with patch.object(
                worker._proc.stdin, "write", wraps=worker._proc.stdin.write
            ) as write:
                # Hold a real runtime admission lease; the close fails locally.
                async with runtime._operation("busy"):
                    assert await serving._evict_idle_session() is False
                write.assert_not_called()
            assert list(serving._idle_sessions) == ["a", "b"]
            assert await serving._evict_idle_session() is True
            assert list(serving._idle_sessions) == ["b"]
            assert await worker.close_session("a") is False
            assert await worker.close_session("b") is True

    asyncio.run(asyncio.wait_for(scenario(), 15))
