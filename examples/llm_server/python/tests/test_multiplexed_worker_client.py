# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async multiplexed transport tests with deterministic pipes and real children."""

import asyncio
import json
import os
import sys
import textwrap
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.tests.test_worker_client import _FakeProc
from executorch.examples.llm_server.python.worker_client import (
    _UINT64_MAX,
    spawn_worker,
    WorkerClient,
    WorkerError,
    WorkerStats,
)


_PREAMBLE = """
import json, os, sys

def recv():
    return json.loads(sys.stdin.readline())

def send(request, **fields):
    print(json.dumps(dict(request_id=request['request_id'], **fields)), flush=True)
"""
_LIMIT = 1024 * 1024


def _run(scenario):
    async def bounded():
        await asyncio.wait_for(scenario, timeout=10)

    asyncio.run(bounded())


class _Writer:
    def __init__(self):
        self.frames = asyncio.Queue()
        self.written = []
        self.drain_started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.closed = False
        self.failure = None

    def write(self, payload):
        self.written.append(payload)
        self.frames.put_nowait(json.loads(payload))
        if self.failure is not None:
            raise self.failure

    async def drain(self):
        self.drain_started.set()
        await self.release.wait()

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return

    def is_closing(self):
        return self.closed


class _AsyncProc:
    def __init__(self):
        self.stdin = _Writer()
        self.stdout = asyncio.StreamReader(limit=2 * _LIMIT)
        self.returncode = None
        self.exited = asyncio.Event()
        self.wait_started = asyncio.Event()
        self.reap_allowed = asyncio.Event()
        self.reap_allowed.set()
        self.signals = []
        self.wait_count = 0
        self.wait_failures = 0
        self.reaped = False

    def send(self, request_id, **fields):
        self.stdout.feed_data(
            (json.dumps(dict(request_id=request_id, **fields)) + "\n").encode()
        )

    def terminate(self):
        self.signals.append("terminate")
        self.returncode = -15
        self.exited.set()
        self.stdout.feed_eof()

    def kill(self):
        self.signals.append("kill")
        self.returncode = -9
        self.exited.set()
        self.stdout.feed_eof()

    async def wait(self):
        self.wait_count += 1
        self.wait_started.set()
        if self.wait_failures:
            self.wait_failures -= 1
            raise asyncio.TimeoutError
        await self.exited.wait()
        await self.reap_allowed.wait()
        self.reaped = True
        return self.returncode


@asynccontextmanager
async def _fake_client(**kwargs):
    proc = _AsyncProc()
    client = MultiplexedWorkerClient(proc, **kwargs)
    try:
        yield client, proc
    finally:
        proc.stdin.release.set()
        proc.reap_allowed.set()
        await client.close()


@asynccontextmanager
async def _real_client(script, *, negotiate=False, **kwargs):
    cmd = [
        sys.executable,
        "-u",
        "-c",
        _PREAMBLE + textwrap.dedent(script) + "\nfor _ in sys.stdin: pass\n",
    ]
    if negotiate:
        client = await spawn_multiplexed_worker(cmd, **kwargs)
        proc = client._proc
    else:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        client = MultiplexedWorkerClient(proc, **kwargs)
    try:
        yield client
    finally:
        try:
            await client.close()
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()


async def _collect(stream):
    return [token async for token in stream]


async def _barrier(client, proc):
    operation = asyncio.create_task(client.open_session("barrier"))
    request = await proc.stdin.frames.get()
    assert request["op"] == "open"
    proc.send(request["request_id"], opened=True)
    await operation


def test_interleaved_generation_lifecycle_and_out_of_order_completion():
    async def scenario():
        async with _real_client(
            """
            a = recv()
            send(a, token='a1')
            requests = [recv(), recv()]
            b = next(r for r in requests if r['op'] == 'generate')
            lifecycle = next(r for r in requests if r['op'] == 'open')
            assert a['request_id'] < b['request_id'] < lifecycle['request_id']
            send(lifecycle, opened=True)
            send(b, token='b1')
            send(b, done=True, completion_tokens=1)
            send(a, token='a2')
            send(a, done=True, completion_tokens=2)
            """
        ) as client:
            a_id, b_id = client.reserve_request(), client.reserve_request()
            a = client.generate("a", SimpleNamespace(), request_id=a_id)
            assert a.request_id == a_id
            assert await anext(a) == "a1"
            b = client.generate("b", SimpleNamespace(), request_id=b_id)
            lifecycle = asyncio.create_task(client.open_session("s"))
            tokens_a, tokens_b, _ = await asyncio.gather(
                _collect(a), _collect(b), lifecycle
            )
            assert tokens_a == ["a2"] and tokens_b == ["b1"]
            assert (await a.wait()).num_generated_tokens == 2
            assert (await b.wait()).num_generated_tokens == 1
            assert client.healthy and not client._requests

    _run(scenario())


@pytest.mark.parametrize("ack_first", [True, False])
def test_targeted_cancel_ack_and_generation_terminal_are_independent(ack_first):
    async def scenario():
        async with _real_client(
            f"""
            a, b = sorted([recv(), recv()], key=lambda r: r['prompt'])
            send(a, token='started')
            c = recv()
            assert c['op'] == 'cancel' and c['target_request_id'] == a['request_id']
            assert c['request_id'] > b['request_id']
            send(b, token='unaffected')
            send(b, done=True)
            if {ack_first!r}:
                send(c, cancelled=True)
            send(a, done=True, cancelled=True, finish_reason='stop')
            if not {ack_first!r}:
                send(c, cancelled=True)
            barrier = recv()
            send(barrier, opened=True)
            """
        ) as client:
            a = client.generate("a", SimpleNamespace())
            b = client.generate("b", SimpleNamespace())
            assert await anext(a) == "started"
            assert not client.stop()
            assert a.cancel() and client.cancel(a.request_id)
            tokens_a, tokens_b = await asyncio.gather(_collect(a), _collect(b))
            assert tokens_a == [] and tokens_b == ["unaffected"]
            stats = await a.wait()
            assert stats.cancelled and stats.finish_reason == "stop"
            assert not client.cancel(a.request_id)
            await client.open_session("barrier")
            assert not client._requests and not client._controls

    _run(scenario())


def test_cancel_before_submission_is_latched_without_sending():
    async def scenario():
        async with _real_client(
            """
            b = recv()
            assert b['op'] == 'generate' and b['request_id'] == 2
            send(b, done=True)
            """
        ) as client:
            request_id = client.reserve_request()
            assert client.cancel(request_id)
            stream = client.generate("cancelled", SimpleNamespace(), request_id)
            assert await _collect(stream) == []
            assert (await stream.wait()).cancelled
            peer = client.generate("b", SimpleNamespace())
            assert await _collect(peer) == []
            assert not client._requests

    _run(scenario())


def test_cancel_queued_generation_bounds_writer_backlog():
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            proc.stdin.release.clear()
            active = client.generate("a", SimpleNamespace())
            await proc.stdin.drain_started.wait()
            assert (await proc.stdin.frames.get())["prompt"] == "a"
            for _ in range(20):
                queued = client.generate("cancelled", SimpleNamespace())
                assert queued.cancel()
                assert await _collect(queued) == []
                assert (await queued.wait()).cancelled
                assert len(client._writes) <= 2 * client.max_inflight_requests
            assert len(proc.stdin.written) == 1
            proc.send(active.request_id, done=True)
            assert await _collect(active) == []
            proc.stdin.release.set()
            await _barrier(client, proc)
            assert not client._requests and not client._controls

    _run(scenario())


@pytest.mark.parametrize("overflow", ["tokens", "characters"])
def test_local_overflow_holds_reservation_until_wire_terminal_and_isolates_peer(
    overflow,
):
    async def scenario():
        limits = (
            {"mailbox_capacity": 2}
            if overflow == "tokens"
            else {"max_buffered_chars": 3}
        )
        async with _fake_client(max_inflight_requests=2, **limits) as (client, proc):
            slow = client.generate("slow", SimpleNamespace())
            fast = client.generate("fast", SimpleNamespace())
            await proc.stdin.frames.get()
            await proc.stdin.frames.get()
            proc.stdin.release.clear()
            proc.stdin.drain_started.clear()
            for token in ["a", "b", "c"] if overflow == "tokens" else ["ab", "cd"]:
                proc.send(slow.request_id, token=token)
            cancel = await proc.stdin.frames.get()
            assert cancel["op"] == "cancel"
            assert cancel["target_request_id"] == slow.request_id
            await proc.stdin.drain_started.wait()
            with pytest.raises(WorkerError) as error:
                await _collect(slow)
            assert error.value.code == "slow_consumer"
            wire = asyncio.create_task(slow.wait())
            settlement = asyncio.create_task(client.wait_for_request(slow.request_id))
            proc.send(fast.request_id, token="ok")
            proc.send(fast.request_id, done=True)
            assert await _collect(fast) == ["ok"]
            assert not wire.done() and not settlement.done()
            replacement = client.reserve_request()
            with pytest.raises(WorkerError) as full:
                client.reserve_request()
            assert full.value.code == "capacity_exhausted"
            assert slow.request_id in client._requests
            proc.send(cancel["request_id"], cancelled=True)
            proc.send(slow.request_id, done=True, cancelled=True, finish_reason="stop")
            assert (await wire).cancelled
            await settlement
            assert slow.request_id not in client._requests
            assert client.release_request(replacement)
            assert client.healthy

    _run(scenario())


def test_cancel_ack_error_is_local_until_native_terminal():
    async def scenario():
        async with _fake_client(max_inflight_requests=1) as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            assert stream.cancel()
            cancel = await proc.stdin.frames.get()
            proc.send(cancel["request_id"], error="cannot cancel", code="busy")
            with pytest.raises(WorkerError, match="cannot cancel") as error:
                await anext(stream)
            assert error.value.code == "busy"
            waiter = asyncio.create_task(stream.wait())
            await asyncio.sleep(0)
            assert not waiter.done()
            with pytest.raises(WorkerError) as full:
                client.reserve_request()
            assert full.value.code == "capacity_exhausted"
            proc.send(
                stream.request_id, done=True, cancelled=True, finish_reason="stop"
            )
            assert (await waiter).cancelled
            assert not client._requests and client.healthy

    _run(scenario())


@pytest.mark.parametrize("terminal_error", [False, True])
def test_wire_terminal_requires_consumption_and_drains_queued_tokens(terminal_error):
    async def scenario():
        async with _fake_client(max_inflight_requests=1) as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            proc.send(stream.request_id, token="first")
            proc.send(stream.request_id, token="second")
            if terminal_error:
                proc.send(stream.request_id, error="native failure", code="bad_request")
            else:
                proc.send(stream.request_id, done=True, completion_tokens=2)
            await client.wait_for_request(stream.request_id)
            with pytest.raises(WorkerError) as full:
                client.reserve_request()
            assert full.value.code == "capacity_exhausted"
            assert await anext(stream) == "first"
            assert await anext(stream) == "second"
            if terminal_error:
                with pytest.raises(WorkerError, match="native failure") as consumed:
                    await anext(stream)
                with pytest.raises(WorkerError) as waited:
                    await stream.wait()
                assert consumed.value is waited.value
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
                assert (await stream.wait()).num_generated_tokens == 2
            assert not client._requests
            await client.wait_for_request(stream.request_id)
            replacement = client.reserve_request()
            assert client.release_request(replacement)

    _run(scenario())


@pytest.mark.parametrize("abandonment", ["break", "body_error", "cancel_anext"])
def test_abandoned_consumer_does_not_poison_peers_or_wait_for_wire(abandonment):
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            if abandonment == "cancel_anext":
                consumer = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                consumer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await consumer
            else:
                proc.send(stream.request_id, token="first")
                proc.send(stream.request_id, token="discard")

                async def consume():
                    try:
                        async for _ in stream:
                            if abandonment == "body_error":
                                raise ValueError("consumer failed")
                            break
                    finally:
                        await stream.aclose()

                if abandonment == "body_error":
                    with pytest.raises(ValueError, match="consumer failed"):
                        await consume()
                else:
                    await consume()
            await stream.aclose()
            cancel = await proc.stdin.frames.get()
            assert cancel["op"] == "cancel"
            assert cancel["target_request_id"] == stream.request_id
            assert stream.request_id in client._requests
            assert not client._requests[stream.request_id].tokens
            waiter = asyncio.create_task(stream.wait())
            await _barrier(client, proc)
            assert not waiter.done()
            proc.send(cancel["request_id"], cancelled=True)
            proc.send(
                stream.request_id, done=True, cancelled=True, finish_reason="stop"
            )
            assert (await waiter).cancelled
            assert not client._requests and client.healthy

    _run(scenario())


def test_aclose_returns_while_writer_is_blocked_and_discards_buffer():
    async def scenario():
        async with _fake_client() as (client, proc):
            proc.stdin.release.clear()
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.drain_started.wait()
            await proc.stdin.frames.get()
            proc.send(stream.request_id, token="first")
            proc.send(stream.request_id, token="discard")
            assert await anext(stream) == "first"
            await stream.aclose()
            assert not client._requests[stream.request_id].tokens
            assert stream.request_id in client._requests
            assert len(proc.stdin.written) == 1
            proc.stdin.release.set()
            cancel = await proc.stdin.frames.get()
            assert cancel["op"] == "cancel"
            proc.send(cancel["request_id"], cancelled=True)
            proc.send(
                stream.request_id, done=True, cancelled=True, finish_reason="stop"
            )
            assert (await stream.wait()).cancelled
            assert not client._requests

    _run(scenario())


def test_waiter_cancellation_is_shielded_and_does_not_steal_token_wakeup():
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            cancelled = asyncio.create_task(stream.wait())
            survivor = asyncio.create_task(stream.wait())
            settlement = asyncio.create_task(client.wait_for_request(stream.request_id))
            cancelled_settlement = asyncio.create_task(
                client.wait_for_request(stream.request_id)
            )
            consumer = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            cancelled.cancel()
            cancelled_settlement.cancel()
            for waiter in (cancelled, cancelled_settlement):
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            proc.send(stream.request_id, token="awake")
            assert await consumer == "awake"
            assert not survivor.done() and not settlement.done()
            assert not client._controls
            proc.send(stream.request_id, done=True, completion_tokens=1)
            assert (await survivor).num_generated_tokens == 1
            await settlement
            assert await _collect(stream) == []
            assert not client._requests

    _run(scenario())


def test_concurrent_anext_is_rejected_without_abandoning_first_consumer():
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            first = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            with pytest.raises(
                (RuntimeError, WorkerError),
                match="concurrent|consumer|active token reader",
            ):
                await anext(stream)
            proc.send(stream.request_id, token="first")
            assert await first == "first"
            proc.send(stream.request_id, done=True)
            assert await _collect(stream) == []
            assert not client._controls and client.healthy

    _run(scenario())


_BAD_FRAMES = [
    b"not JSON\n",
    b"[]\n",
    b'{"request_id":1,"token":"\xff"}\n',
    b'{"request_id":1',
    b'{"request_id":1,"done":true}',
    b'{"request_id":1,"done":true,"total_ms":NaN}\n',
    b'{"request_id":1,"done":true,"total_ms":Infinity}\n',
] + [
    (json.dumps(message) + "\n").encode()
    for message in [
        {"request_id": 1, "token": 123},
        {"request_id": 1, "done": True, "token": "conflict"},
        {"request_id": 1, "done": False},
        {"request_id": 1, "done": True, "cancelled": True, "finish_reason": "length"},
        {"request_id": 1, "error": 4},
        {"request_id": 1, "error": "bad", "code": False},
        {"request_id": 1, "opened": True},
        {"request_id": 1, "cancelled": True},
        {"request_id": True, "done": True},
        {"request_id": 0, "done": True},
        {"request_id": 2**64, "done": True},
        {"request_id": 99999, "done": True},
        {"done": True},
    ]
]


@pytest.mark.parametrize("frame", [b"", *_BAD_FRAMES])
def test_eof_or_protocol_failure_settles_all_operations_and_reaps(frame):
    async def scenario():
        async with _fake_client() as (client, proc):
            a = client.generate("a", SimpleNamespace())
            b = client.generate("b", SimpleNamespace())
            lifecycle = asyncio.create_task(client.open_session("s"))
            for _ in range(3):
                await proc.stdin.frames.get()
            proc.stdout.feed_data(frame)
            if not frame.endswith(b"\n"):
                proc.stdout.feed_eof()
            results = await asyncio.gather(
                a.wait(), b.wait(), lifecycle, return_exceptions=True
            )
            assert isinstance(results[0], WorkerError)
            assert results[0] is results[1] is results[2]
            assert client.failed and not client.healthy
            with pytest.raises(WorkerError) as later:
                client.reserve_request()
            assert later.value is results[0]
            await client.wait_for_request(a.request_id)
            await proc.wait_started.wait()
            assert client._cleanup_task is not None
            cleanup = client._cleanup_task
            await asyncio.shield(cleanup)
            assert proc.returncode is not None
            assert client._reader.done() and client._writer.done()
            await client.abort()
            assert client._cleanup_task is cleanup
            assert proc.wait_count == 1

    _run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "reused_prompt_tokens",
            "prefilled_prompt_tokens",
            "prompt_positions",
            "reused_prompt_positions",
            "prefilled_prompt_positions",
        )
        for value in (True, -1, 1.5)
    ]
    + [
        (field, value)
        for field in (
            "prefill_ms",
            "decode_ms",
            "total_ms",
            "prefill_tok_s",
            "decode_tok_s",
            "vision_encoder_ms",
        )
        for value in (True, -1, "slow")
    ]
    + [
        ("finish_reason", "unknown"),
        ("cancelled", 1),
        ("cancelled", True),
        ("session_reset_reason", False),
        ("generated_token_ids", {}),
        ("generated_token_ids", [True]),
        ("generated_token_ids", [-1]),
    ],
)
def test_invalid_generation_statistics_are_fatal(field, value):
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            proc.send(stream.request_id, done=True, **{field: value})
            with pytest.raises(WorkerError):
                await stream.wait()
            assert client.failed

    _run(scenario())


def test_duplicate_terminal_is_fatal_without_changing_completed_stats():
    async def scenario():
        async with _fake_client() as (client, proc):
            a = client.generate("a", SimpleNamespace())
            b = client.generate("b", SimpleNamespace())
            await proc.stdin.frames.get()
            await proc.stdin.frames.get()
            proc.send(a.request_id, done=True, completion_tokens=2)
            assert await _collect(a) == []
            stats = await a.wait()
            proc.send(a.request_id, done=True, completion_tokens=99)
            with pytest.raises(WorkerError, match="inactive request"):
                await b.wait()
            assert await a.wait() is stats
            assert stats.num_generated_tokens == 2 and client.failed

    _run(scenario())


@pytest.mark.parametrize("operation", ["generation", "cancel"])
def test_response_before_frame_is_sent_is_fatal(operation):
    async def scenario():
        async with _fake_client() as (client, proc):
            proc.stdin.release.clear()
            a = client.generate("a", SimpleNamespace())
            await proc.stdin.drain_started.wait()
            await proc.stdin.frames.get()
            if operation == "generation":
                queued = client.generate("queued", SimpleNamespace())
                proc.send(queued.request_id, done=True)
            else:
                assert a.cancel()
                cancel_id = next(iter(client._controls))
                proc.send(cancel_id, cancelled=True)
            with pytest.raises(WorkerError):
                await a.wait()
            assert client.failed and len(proc.stdin.written) == 1

    _run(scenario())


def test_write_failure_settles_every_request_with_same_error():
    async def scenario():
        async with _fake_client() as (client, proc):
            proc.stdin.failure = BrokenPipeError("partial write")
            a = client.generate("a", SimpleNamespace())
            b = client.generate("b", SimpleNamespace())
            errors = await asyncio.gather(a.wait(), b.wait(), return_exceptions=True)
            assert isinstance(errors[0], WorkerError)
            assert errors[0] is errors[1]
            assert "write failed" in str(errors[0])
            assert client.failed and len(proc.stdin.written) == 1

    _run(scenario())


@pytest.mark.parametrize("failure", ["abandon", "overflow", "cancel", "stop"])
def test_cancel_retries_when_acks_for_retired_targets_free_control_budget(failure):
    async def scenario():
        async with _fake_client(max_inflight_requests=2, mailbox_capacity=1) as (
            client,
            proc,
        ):
            delayed = []
            for _ in range(2):
                seed = client.generate("seed", SimpleNamespace())
                await proc.stdin.frames.get()
                await seed.aclose()
                cancel = await proc.stdin.frames.get()
                delayed.append(cancel)
                proc.send(
                    seed.request_id, done=True, cancelled=True, finish_reason="stop"
                )
                assert (await seed.wait()).cancelled
                assert seed.request_id not in client._requests
            assert len(client._controls) == 2
            target = client.generate("target", SimpleNamespace())
            await proc.stdin.frames.get()
            if failure == "abandon":
                await target.aclose()
            elif failure == "overflow":
                proc.send(target.request_id, token="one")
                proc.send(target.request_id, token="two")
                await _barrier(client, proc)
                with pytest.raises(WorkerError) as error:
                    await _collect(target)
                assert error.value.code == "slow_consumer"
            else:
                for _ in range(20):
                    assert target.cancel() if failure == "cancel" else client.stop()
            assert len(client._controls) == 2
            assert proc.stdin.frames.empty()
            assert len(client._writes) <= 4
            proc.send(delayed[0]["request_id"], cancelled=True)
            retried = await proc.stdin.frames.get()
            assert retried["op"] == "cancel"
            assert retried["target_request_id"] == target.request_id
            assert len(client._controls) == 2
            proc.send(delayed[1]["request_id"], cancelled=True)
            proc.send(retried["request_id"], cancelled=True)
            proc.send(
                target.request_id, done=True, cancelled=True, finish_reason="stop"
            )
            if failure in ("cancel", "stop"):
                assert await _collect(target) == []
            assert (await target.wait()).cancelled
            await _barrier(client, proc)
            assert not client._requests and not client._controls
            assert proc.stdin.frames.empty() and client.healthy

    _run(scenario())


def test_capacity_release_reset_and_uint64_exhaustion():
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            client.reset()
            assert not client.stop()
            a_id, b_id = client.reserve_request(), client.reserve_request()
            with pytest.raises(WorkerError) as full:
                client.reserve_request()
            assert full.value.code == "capacity_exhausted"
            assert not client.release_request(999)
            assert client.release_request(b_id)
            client._next_request_id = _UINT64_MAX
            last = client.reserve_request()
            assert last == _UINT64_MAX and client.release_request(last)
            a = client.generate("a", SimpleNamespace(), a_id)
            assert not client.release_request(a_id)
            with pytest.raises(WorkerError, match="ids exhausted") as exhausted:
                client.reserve_request()
            with pytest.raises(WorkerError) as failed:
                await a.wait()
            assert failed.value is exhausted.value and client.failed
            with pytest.raises(WorkerError):
                client.reset()

    _run(scenario())


def test_generation_shape_stats_and_lifecycle_errors():
    async def scenario():
        async with _real_client(
            """
            g = recv()
            assert g == dict(op='generate', request_id=1, prompt_segments=[dict(ids=[1,2])], session_id='s', max_new_tokens=8, temperature=.5, top_p=.9, top_k=3, seed=9, stop=['end'])
            send(g, done=True, generated_token_ids=[7], prompt_tokens=2,
                 completion_tokens=1, reused_prompt_tokens=1, prefilled_prompt_tokens=1,
                 session_reset_reason='exact_prefix', prefill_ms=12.5, decode_ms=25,
                 total_ms=40, prefill_tok_s=80, decode_tok_s=40,
                 vision_encoder_ms=123.5, finish_reason='length')
            r = recv()
            assert r['op'] == 'reset' and r['request_id'] == 2
            send(r, reset=True)
            c = recv()
            assert c['op'] == 'close' and c['request_id'] == 3
            send(c, closed=True)
            o = recv()
            assert o['op'] == 'open'
            send(o, error='full', code='capacity_exhausted')
            """
        ) as client:
            config = SimpleNamespace(
                prompt_segments=[{"ids": [1, 2]}],
                session_id="s",
                max_new_tokens=8,
                temperature=0.5,
                top_p=0.9,
                top_k=3,
                seed=9,
                stop=["end"],
            )
            stream = client.generate("ignored", config)
            assert await _collect(stream) == []
            stats = await stream.wait()
            assert isinstance(stats, WorkerStats)
            assert stats.generated_token_ids == [7]
            assert stats.num_prompt_tokens == 2 and stats.num_generated_tokens == 1
            assert (
                stats.reused_prompt_tokens == 1 and stats.prefilled_prompt_tokens == 1
            )
            assert stats.session_reset_reason == "exact_prefix"
            assert (stats.prefill_ms, stats.decode_ms, stats.total_ms) == (12.5, 25, 40)
            assert (stats.prefill_tok_s, stats.decode_tok_s) == (80, 40)
            assert stats.vision_encoder_ms == 123.5 and stats.finish_reason == "length"
            await client.reset_session("s")
            await client.close_session("s")
            with pytest.raises(WorkerError) as error:
                await client.open_session("s")
            assert error.value.code == "capacity_exhausted"
            assert client.healthy and not client._requests

    _run(scenario())


@pytest.mark.parametrize("generated_ids", [None, [], [7, 8]])
def test_generated_token_ids_presence_survives_async_transport(generated_ids):
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("hi", SimpleNamespace())
            await proc.stdin.frames.get()
            metadata = (
                {} if generated_ids is None else {"generated_token_ids": generated_ids}
            )
            proc.send(stream.request_id, done=True, **metadata)
            assert await _collect(stream) == []
            assert (await stream.wait()).generated_token_ids == generated_ids

    _run(scenario())


def test_async_factory_negotiates_limits_and_forwards_env_cwd(tmp_path):
    async def scenario():
        async with _real_client(
            f"""
            assert os.path.realpath(os.getcwd()) == os.path.realpath({str(tmp_path)!r})
            assert os.environ['ASYNC_WORKER_TEST'] == 'present'
            print(json.dumps(dict(ready=True, multiplexed=True, max_named_sessions=3, max_inflight_requests=5)), flush=True)
            g = recv()
            assert g['op'] == 'generate' and 'cancel_request_id' not in g
            send(g, done=True)
            """,
            negotiate=True,
            env={**os.environ, "ASYNC_WORKER_TEST": "present"},
            cwd=str(tmp_path),
        ) as client:
            assert client.supports_multiplexing and client.supports_cancel
            assert client.max_named_sessions == 3 and client.max_inflight_requests == 5
            assert await _collect(client.generate("hi", SimpleNamespace())) == []

    _run(scenario())


@pytest.mark.parametrize("capability", [{}, {"multiplexed": False}])
@pytest.mark.parametrize("generated_ids", [None, [], [7, 8]])
def test_sync_factory_keeps_legacy_transport_and_generated_ids(
    capability, generated_ids
):
    metadata = {} if generated_ids is None else {"generated_token_ids": generated_ids}
    proc = _FakeProc(
        [
            json.dumps(dict(ready=True, **capability)) + "\n",
            json.dumps(dict(done=True, **metadata)) + "\n",
        ]
    )
    client = spawn_worker([sys.executable], popen=lambda *args, **kwargs: proc)
    try:
        assert isinstance(client, WorkerClient) and not client.supports_multiplexing
        stats = []
        client.generate("hi", SimpleNamespace(), stats_callback=stats.append)
        request = json.loads(proc.stdin.written[0])
        assert "op" not in request and "request_id" not in request
        assert len(stats) == 1 and stats[0].generated_token_ids == generated_ids
    finally:
        client.close()


def test_sync_factory_rejects_multiplexing_with_async_guidance_and_reaps():
    proc = _FakeProc([json.dumps({"ready": True, "multiplexed": True}) + "\n"])
    stdin, stdout = proc.stdin, proc.stdout
    with pytest.raises(WorkerError, match="await spawn_multiplexed_worker"):
        spawn_worker([sys.executable], popen=lambda *args, **kwargs: proc)
    assert proc.terminated and proc.waited
    assert stdin.closed and stdout.closed


@pytest.mark.parametrize(
    "readiness",
    [
        {},
        {"ready": False, "multiplexed": True},
        {"ready": 1, "multiplexed": True},
        {"ready": True},
        {"ready": True, "multiplexed": False},
        {"ready": True, "multiplexed": 1},
        {"ready": True, "multiplexed": True, "max_inflight_requests": 0},
        {"ready": True, "multiplexed": True, "max_inflight_requests": True},
        {"ready": True, "multiplexed": True, "max_named_sessions": -1},
    ],
)
def test_async_readiness_failure_reaps_process(readiness):
    async def scenario():
        proc = _AsyncProc()
        proc.stdout.feed_data((json.dumps(readiness) + "\n").encode())

        async def spawn(*args, **kwargs):
            return proc

        with patch("asyncio.create_subprocess_exec", spawn):
            with pytest.raises(WorkerError):
                await spawn_multiplexed_worker([sys.executable])
        assert proc.returncode is not None and proc.wait_count == 1
        assert proc.stdin.closed

    _run(scenario())


def test_cancellation_during_readiness_reaps_child():
    async def scenario():
        proc = _AsyncProc()
        spawned = asyncio.Event()

        async def spawn(*args, **kwargs):
            spawned.set()
            return proc

        with patch("asyncio.create_subprocess_exec", spawn):
            operation = asyncio.create_task(spawn_multiplexed_worker([sys.executable]))
            await spawned.wait()
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        assert proc.returncode is not None and proc.wait_count == 1
        assert proc.stdin.closed

    _run(scenario())


def test_repeated_startup_cancellation_waits_for_held_reap():
    async def scenario():
        proc = _AsyncProc()
        proc.reap_allowed.clear()
        spawned = asyncio.Event()
        reaped_at_cancellation = []

        async def spawn(*args, **kwargs):
            spawned.set()
            return proc

        async def startup():
            try:
                await spawn_multiplexed_worker([sys.executable])
            except asyncio.CancelledError:
                reaped_at_cancellation.append(proc.reaped)
                raise

        with patch("asyncio.create_subprocess_exec", spawn):
            operation = asyncio.create_task(startup())
            try:
                await spawned.wait()
                operation.cancel()
                await proc.wait_started.wait()
                for _ in range(3):
                    assert operation.cancel()
                    await asyncio.sleep(0)
                    assert not operation.done()
                    assert not proc.reaped
                assert reaped_at_cancellation == []
                proc.reap_allowed.set()
                with pytest.raises(asyncio.CancelledError):
                    await operation
                assert reaped_at_cancellation == [True]
                assert proc.reaped and proc.wait_count == 1
                assert proc.stdin.closed and proc.signals == ["terminate"]
            finally:
                proc.reap_allowed.set()
                await asyncio.gather(operation, return_exceptions=True)

    _run(scenario())


@pytest.mark.parametrize("method", ["close", "abort"])
def test_shutdown_reports_failed_reap_and_retries_on_next_call(method):
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            proc.wait_failures = 2
            operation = client.close if method == "close" else client.abort
            with pytest.raises(WorkerError, match="could not be reaped"):
                await operation()
            failed_cleanup = client._cleanup_task
            assert failed_cleanup.done() and failed_cleanup.result() is False
            assert not proc.reaped and proc.wait_count == 2
            assert client._reader.done() and client._writer.done()
            with pytest.raises(WorkerError):
                await stream.wait()
            await operation()
            successful_cleanup = client._cleanup_task
            assert successful_cleanup is not failed_cleanup
            assert successful_cleanup.done() and successful_cleanup.result() is True
            assert proc.reaped and proc.wait_count == 3
            await operation()
            assert client._cleanup_task is successful_cleanup and proc.wait_count == 3
            assert not client.healthy and not client._requests

    _run(scenario())


@pytest.mark.parametrize("method", ["close", "abort"])
def test_shutdown_is_shielded_idempotent_and_settles_waiters(method):
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            lifecycle = asyncio.create_task(client.open_session("s"))
            await proc.stdin.frames.get()
            await proc.stdin.frames.get()
            proc.reap_allowed.clear()
            operation = client.close if method == "close" else client.abort
            first = asyncio.create_task(operation())
            await proc.wait_started.wait()
            cleanup = client._cleanup_task
            second = asyncio.create_task(operation())
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not cleanup.done() and not second.done()
            failures = await asyncio.gather(
                stream.wait(), lifecycle, return_exceptions=True
            )
            assert isinstance(failures[0], WorkerError) and failures[0] is failures[1]
            await client.wait_for_request(stream.request_id)
            proc.reap_allowed.set()
            await second
            await operation()
            assert client._cleanup_task is cleanup and cleanup.done()
            assert proc.wait_count == 1 and proc.signals == ["terminate"]
            assert client._reader.done() and client._writer.done()
            assert not client.healthy
            assert client.failed == (method == "abort")
            if method == "close":
                assert client.closed

    _run(scenario())


def test_real_eof_and_normal_shutdown_reap_children():
    async def scenario():
        async with _real_client("recv(); sys.exit(0)") as client:
            stream = client.generate("a", SimpleNamespace())
            with pytest.raises(WorkerError):
                await stream.wait()
            await client._cleanup_task
            assert client._proc.returncode is not None and client.failed
        async with _real_client("a = recv(); send(a, token='started')") as client:
            stream = client.generate("a", SimpleNamespace())
            assert await anext(stream) == "started"
            await client.close()
            await client.close()
            with pytest.raises(WorkerError, match="closed"):
                await stream.wait()
            assert client.closed and not client.failed
            assert client._proc.returncode is not None

    _run(scenario())


@pytest.mark.parametrize("request_id", [True, 0, -1, 2**64, "1"])
def test_invalid_request_handles(request_id):
    async def scenario():
        async with _fake_client() as (client, proc):
            for operation in (client.cancel, client.release_request):
                with pytest.raises(WorkerError, match="request_id"):
                    operation(request_id)
            with pytest.raises(WorkerError, match="request_id"):
                client.generate("hi", SimpleNamespace(), request_id)
            with pytest.raises(WorkerError, match="request_id"):
                await client.wait_for_request(request_id)
            assert not client._requests and not proc.stdin.written

    _run(scenario())


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in (
            "max_inflight_requests",
            "mailbox_capacity",
            "max_buffered_chars",
            "max_message_bytes",
        )
        for value in (True, 0, -1, 1.5)
    ]
    + [("max_named_sessions", value) for value in (True, -1, 1.5)],
)
def test_invalid_constructor_budgets_are_rejected(name, value):
    async def scenario():
        proc = _AsyncProc()
        with pytest.raises(WorkerError, match="invalid"):
            MultiplexedWorkerClient(proc, **{name: value})
        assert not proc.stdin.written

    _run(scenario())


def test_constructor_requires_running_loop():
    with pytest.raises(RuntimeError, match="running event loop"):
        MultiplexedWorkerClient(_FakeProc([]))


@pytest.mark.parametrize("extra_bytes", [0, 1])
@pytest.mark.parametrize("escaped", ['"\\\n', "\u00e9\U0001f600"])
def test_outbound_frame_limit_counts_encoded_jsonl_and_max_uint64(extra_bytes, escaped):
    async def scenario():
        async with _real_client(
            f"""
            line = sys.stdin.readline()
            assert len(line.encode('utf-8')) == {_LIMIT}
            request = json.loads(line)
            assert line == json.dumps(request, ensure_ascii=True, allow_nan=False) + '\\n'
            assert request['request_id'] == {_UINT64_MAX}
            send(request, done=True)
            """,
            max_message_bytes=128,
        ) as client:
            client._next_request_id = _UINT64_MAX
            request = {
                "op": "generate",
                "max_new_tokens": -1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 0,
                "stop": [],
                "prompt": escaped,
                "request_id": _UINT64_MAX,
            }
            overhead = len((json.dumps(request, ensure_ascii=True) + "\n").encode())
            prompt = escaped + "x" * (_LIMIT - overhead + extra_bytes)
            if extra_bytes:
                with pytest.raises(WorkerError) as error:
                    client.generate(prompt, SimpleNamespace())
                assert error.value.code == "invalid_argument"
            else:
                stream = client.generate(prompt, SimpleNamespace())
                assert await _collect(stream) == []
                assert isinstance(await stream.wait(), WorkerStats)
            assert not client._requests and not client._writes and not client._controls
            assert client.healthy

    _run(scenario())


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_inbound_frame_limit_counts_utf8_bytes_including_newline(extra_bytes):
    async def scenario():
        async with _fake_client(max_message_bytes=128) as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            prefix = b'{"request_id":1,"token":"'
            suffix = b'"}\n'
            token = "\u00e9" * ((128 - len(prefix) - len(suffix)) // 2)
            payload = (
                prefix
                + token.encode()
                + b"x"
                * (128 - len(prefix) - len(suffix) - len(token.encode()) + extra_bytes)
                + suffix
            )
            proc.stdout.feed_data(payload)
            if extra_bytes:
                with pytest.raises(WorkerError):
                    await stream.wait()
                assert client.failed
            else:
                assert await anext(stream) == json.loads(payload)["token"]
                proc.send(stream.request_id, done=True)
                assert await _collect(stream) == [] and client.healthy

    _run(scenario())


@pytest.mark.parametrize("reserved", [False, True])
@pytest.mark.parametrize("field", ["prompt", "stop", "prompt_segments", "session_id"])
def test_oversized_payload_releases_only_its_reservation(reserved, field):
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            peer_id = client.reserve_request()
            request_id = client.reserve_request() if reserved else None
            waiter = None
            if reserved:
                waiter = asyncio.create_task(client.wait_for_request(request_id))
                await asyncio.sleep(0)
            oversized = "x" * _LIMIT
            configs = {
                "prompt": SimpleNamespace(),
                "stop": SimpleNamespace(stop=[oversized]),
                "prompt_segments": SimpleNamespace(
                    prompt_segments=[{"text": oversized}]
                ),
                "session_id": SimpleNamespace(session_id=oversized),
            }
            with pytest.raises(WorkerError) as error:
                client.generate(
                    oversized if field == "prompt" else "small",
                    configs[field],
                    request_id,
                )
            assert error.value.code == "invalid_argument"
            if waiter is not None:
                await waiter
                await client.wait_for_request(request_id)
            assert list(client._requests) == [peer_id]
            assert (
                not proc.stdin.written and not client._writes and not client._controls
            )
            replacement = client.reserve_request()
            assert client.release_request(replacement) and client.release_request(
                peer_id
            )
            assert client.healthy

    _run(scenario())


_INVALID_NATIVE_INPUTS = (
    [
        pytest.param("\ud800", {}, id="prompt-high-surrogate"),
        pytest.param("\udfff", {}, id="prompt-low-surrogate"),
        pytest.param("hi", {"stop": ["end\ud800"]}, id="stop-surrogate"),
        pytest.param("hi", {"session_id": "s\udfff"}, id="session-surrogate"),
        pytest.param(
            "hi", {"prompt_segments": [{"text": "\ud800"}]}, id="segment-surrogate"
        ),
        pytest.param(
            "hi", {"prompt_segments": [{"\ud800": "text"}]}, id="key-surrogate"
        ),
        pytest.param("hi", {"temperature": 10**400}, id="numeric-parser-overflow"),
        pytest.param("hi", {"temperature": -(10**400)}, id="negative-parser-overflow"),
    ]
    + [
        pytest.param("hi", {field: value}, id=f"{field}-{name}")
        for field, low, high in (
            ("max_new_tokens", -1, 2**31 - 1),
            ("top_k", 0, 2**31 - 1),
            ("seed", 0, _UINT64_MAX),
        )
        for name, value in (
            ("below-min", low - 1),
            ("above-max", high + 1),
            ("huge", 10**400),
            ("bool", True),
            ("float", 1.5),
        )
    ]
    + [
        pytest.param("hi", {"prompt_segments": [{"ids": [value]}]}, id=f"token-{name}")
        for name, value in (
            ("negative", -1),
            ("overflow", _UINT64_MAX + 1),
            ("huge", 10**400),
            ("bool", True),
            ("float", 1.5),
        )
    ]
)


@pytest.mark.parametrize("reserved", [False, True])
@pytest.mark.parametrize("prompt,config", _INVALID_NATIVE_INPUTS)
def test_invalid_native_input_never_reaches_peer_and_releases_reservation(
    reserved, prompt, config
):
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            peer = client.generate("peer", SimpleNamespace())
            await proc.stdin.frames.get()
            request_id = client.reserve_request() if reserved else None
            waiter = None
            if reserved:
                waiter = asyncio.create_task(client.wait_for_request(request_id))
                await asyncio.sleep(0)
            with pytest.raises(WorkerError) as error:
                client.generate(prompt, SimpleNamespace(**config), request_id)
            assert error.value.code == "invalid_argument"
            if waiter is not None:
                await waiter
            assert list(client._requests) == [peer.request_id]
            assert not client._writes and not client._controls
            assert len(proc.stdin.written) == 1
            replacement = client.generate("valid", SimpleNamespace())
            request = await proc.stdin.frames.get()
            assert request["prompt"] == "valid"
            proc.send(peer.request_id, token="peer survived")
            proc.send(peer.request_id, done=True, completion_tokens=1)
            proc.send(replacement.request_id, token="valid survived")
            proc.send(replacement.request_id, done=True, completion_tokens=1)
            assert await _collect(peer) == ["peer survived"]
            assert await _collect(replacement) == ["valid survived"]
            assert (await peer.wait()).num_generated_tokens == 1
            assert (await replacement.wait()).num_generated_tokens == 1
            assert client.healthy and not client._requests

    _run(scenario())


@pytest.mark.parametrize("session_id", ["\ud800", "\udfff"])
def test_invalid_unicode_lifecycle_is_request_local(session_id):
    async def scenario():
        async with _fake_client(max_inflight_requests=1) as (client, proc):
            for operation in (
                client.open_session,
                client.reset_session,
                client.close_session,
            ):
                with pytest.raises(WorkerError) as error:
                    await operation(session_id)
                assert error.value.code == "invalid_argument"
                assert not client._requests and not client._writes
                assert not proc.stdin.written
            await _barrier(client, proc)
            assert client.healthy

    _run(scenario())


@pytest.mark.parametrize("max_new_tokens", [-1, 0, 2**31 - 1])
def test_native_integer_boundaries_and_valid_unicode_are_preserved(max_new_tokens):
    request = {
        "op": "generate",
        "request_id": _UINT64_MAX,
        "session_id": "session\U0001f600",
        "prompt_segments": [
            {"text": "\u00e9\U0001f600\\ud800"},
            {"ids": [0, 2**63, _UINT64_MAX]},
        ],
        "max_new_tokens": max_new_tokens,
        "top_k": 2**31 - 1,
        "seed": _UINT64_MAX,
        "stop": ["\u7d42"],
    }
    encoded = MultiplexedWorkerClient._encode_request(request)
    assert json.loads(encoded) == request
    assert encoded.endswith(b"\n")


@pytest.mark.parametrize("payload", ["object", "nan", "infinity", "bad_stop"])
def test_unserializable_and_nonfinite_payloads_are_request_local(payload):
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            peer = client.reserve_request()
            request_id = client.reserve_request()
            prompt, config = {
                "object": (object(), SimpleNamespace()),
                "nan": ("hi", SimpleNamespace(temperature=float("nan"))),
                "infinity": ("hi", SimpleNamespace(top_p=float("inf"))),
                "bad_stop": ("hi", SimpleNamespace(stop=3)),
            }[payload]
            with pytest.raises((TypeError, ValueError, WorkerError)):
                client.generate(prompt, config, request_id)
            assert list(client._requests) == [peer]
            await client.wait_for_request(request_id)
            assert not proc.stdin.written and not client._writes
            assert client.release_request(peer)
            valid = client.generate("valid", SimpleNamespace())
            request = await proc.stdin.frames.get()
            assert request["prompt"] == "valid"
            proc.send(valid.request_id, done=True)
            assert await _collect(valid) == [] and client.healthy

    _run(scenario())


def test_oversized_lifecycle_never_reaches_active_peer():
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            stream = client.generate("active", SimpleNamespace())
            await proc.stdin.frames.get()
            for operation in (
                client.open_session,
                client.reset_session,
                client.close_session,
            ):
                with pytest.raises(WorkerError) as error:
                    await operation("x" * _LIMIT)
                assert error.value.code == "invalid_argument"
                assert list(client._requests) == [stream.request_id]
                assert proc.stdin.frames.empty()
            proc.send(stream.request_id, token="survived")
            proc.send(stream.request_id, done=True)
            assert await _collect(stream) == ["survived"] and client.healthy

    _run(scenario())


def test_waiting_consumer_receives_coalesced_tokens_before_character_overflow():
    async def scenario():
        async with _fake_client(mailbox_capacity=64, max_buffered_chars=8) as (
            client,
            proc,
        ):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            first = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert not first.done()
            proc.stdout.feed_data(
                b"".join(
                    (
                        json.dumps(dict(request_id=stream.request_id, **fields)) + "\n"
                    ).encode()
                    for fields in (
                        {"token": "hello"},
                        {"token": "world"},
                        {"done": True, "completion_tokens": 2},
                    )
                )
            )
            assert await first == "hello"
            assert await _collect(stream) == ["world"]
            assert (await stream.wait()).num_generated_tokens == 2
            assert not client._controls and not client._requests
            assert proc.stdin.frames.empty() and client.healthy

    _run(scenario())


def test_reader_yields_while_draining_already_buffered_frames():
    async def scenario():
        async with _fake_client(mailbox_capacity=128) as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            progress = []
            observed = asyncio.Event()
            for _ in range(100):
                proc.send(stream.request_id, token="x")
            proc.send(stream.request_id, done=True)

            def observe():
                progress.append(len(client._requests[stream.request_id].tokens))
                observed.set()

            asyncio.get_running_loop().call_soon(observe)
            await observed.wait()
            assert 0 < progress[0] <= 32
            assert await _collect(stream) == ["x"] * 100
            assert client.healthy

    _run(scenario())


def test_cancel_ack_does_not_settle_generation_or_release_reservation():
    async def scenario():
        async with _fake_client(max_inflight_requests=2) as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            assert stream.cancel()
            cancel = await proc.stdin.frames.get()
            waiter = asyncio.create_task(stream.wait())
            proc.send(cancel["request_id"], cancelled=True)
            await _barrier(client, proc)
            assert not waiter.done()
            assert stream.request_id in client._requests
            assert not client._controls
            proc.send(stream.request_id, token="late")
            proc.send(
                stream.request_id, done=True, cancelled=True, finish_reason="stop"
            )
            assert await _collect(stream) == ["late"]
            assert (await waiter).cancelled
            assert not client._requests

    _run(scenario())


def test_consumption_releases_character_budget_and_accepts_empty_tokens():
    async def scenario():
        async with _fake_client(max_buffered_chars=3) as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            proc.send(stream.request_id, token="\u00e9\u00e9\u00e9")
            assert await anext(stream) == "\u00e9\u00e9\u00e9"
            proc.send(stream.request_id, token="")
            proc.send(stream.request_id, token="abc")
            proc.send(stream.request_id, done=True)
            assert await _collect(stream) == ["", "abc"]
            assert client.healthy and not client._controls

    _run(scenario())


@pytest.mark.parametrize("settlement", ["eof", "protocol", "close"])
def test_abandoned_stream_waiters_settle_on_transport_shutdown(settlement):
    async def scenario():
        async with _fake_client() as (client, proc):
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.frames.get()
            await stream.aclose()
            cancel = await proc.stdin.frames.get()
            proc.send(cancel["request_id"], cancelled=True)
            waiter = asyncio.create_task(stream.wait())
            settlement_waiter = asyncio.create_task(
                client.wait_for_request(stream.request_id)
            )
            await _barrier(client, proc)
            assert not waiter.done() and not settlement_waiter.done()
            if settlement == "eof":
                proc.stdout.feed_eof()
            elif settlement == "protocol":
                proc.stdout.feed_data(b"invalid JSON\n")
            else:
                await client.close()
            with pytest.raises(WorkerError):
                await waiter
            await settlement_waiter
            await client.wait_for_request(stream.request_id)
            assert not client._requests and not client._controls

    _run(scenario())


def test_shutdown_cancels_blocked_writer_and_waiting_token_consumer():
    async def scenario():
        async with _fake_client() as (client, proc):
            proc.stdin.release.clear()
            stream = client.generate("a", SimpleNamespace())
            await proc.stdin.drain_started.wait()
            consumer = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            await client.close()
            with pytest.raises(WorkerError, match="closed"):
                await consumer
            assert client._reader.done() and client._writer.done()
            assert not proc.stdin.release.is_set()
            assert proc.returncode is not None and client.closed

    _run(scenario())


def test_client_and_generation_reject_cross_loop_use():
    owner = asyncio.new_event_loop()

    async def create():
        proc = _AsyncProc()
        client = MultiplexedWorkerClient(proc)
        stream = client.generate("a", SimpleNamespace())
        return client, stream

    client, stream = owner.run_until_complete(create())

    async def wrong_loop():
        for operation in (
            client.reserve_request,
            lambda: client.release_request(stream.request_id),
            lambda: client.cancel(stream.request_id),
            client.stop,
            client.reset,
            lambda: client.generate("b", SimpleNamespace()),
            stream.cancel,
        ):
            with pytest.raises((RuntimeError, WorkerError), match="loop"):
                operation()
        for operation in (
            lambda: client.wait_for_request(stream.request_id),
            lambda: client.open_session("s"),
            lambda: client.reset_session("s"),
            lambda: client.close_session("s"),
            stream.wait,
            stream.aclose,
            lambda: anext(stream),
            client.close,
            client.abort,
        ):
            with pytest.raises((RuntimeError, WorkerError), match="loop"):
                await operation()

    try:
        _run(wrong_loop())
    finally:
        owner.run_until_complete(client.close())
        owner.close()
