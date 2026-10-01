# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multiplexed transport tests using real subprocess stdin/stdout pipes."""

import json
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from executorch.examples.llm_server.python.multiplexed_worker_client import (
    MultiplexedWorkerClient,
)
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


@pytest.fixture
def harness():
    clients = []
    processes = []
    pool = ThreadPoolExecutor(max_workers=8)

    def start(script, *, negotiate=False, wrap_stdin=None, wrap_stdout=None, **kwargs):
        cmd = [
            sys.executable,
            "-u",
            "-c",
            _PREAMBLE + textwrap.dedent(script) + "\nfor _ in sys.stdin: pass\n",
        ]
        if negotiate:
            client = spawn_worker(cmd)
            processes.append(client._proc)
        else:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
            )
            processes.append(proc)
            if wrap_stdin:
                proc.stdin = wrap_stdin(proc.stdin)
            if wrap_stdout:
                proc.stdout = wrap_stdout(proc.stdout)
            client = MultiplexedWorkerClient(proc, **kwargs)
        clients.append(client)
        return client

    yield start, pool
    for client in clients:
        client.close()
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    pool.shutdown(wait=True)


def test_interleaved_generation_lifecycle_and_out_of_order_completion(harness):
    start, pool = harness
    client = start(
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
    )
    a_id, b_id = client.reserve_request(), client.reserve_request()
    seen_a, seen_b, stats = [], [], []
    a_started = threading.Event()

    def token_a(token):
        seen_a.append(token)
        a_started.set()
        assert threading.current_thread() is not client._reader

    a = pool.submit(
        client.generate, "a", SimpleNamespace(), token_a, stats.append, a_id
    )
    assert a_started.wait(5)
    lifecycle = pool.submit(client.open_session, "s")
    b = pool.submit(
        client.generate, "b", SimpleNamespace(), seen_b.append, stats.append, b_id
    )
    lifecycle.result(timeout=5)
    b.result(timeout=5)
    a.result(timeout=5)
    assert seen_a == ["a1", "a2"]
    assert seen_b == ["b1"]
    assert sorted(s.num_generated_tokens for s in stats) == [1, 2]
    assert client.healthy


@pytest.mark.parametrize("ack_first", [True, False])
def test_targeted_cancel_ack_and_generation_terminal_are_independent(
    harness, ack_first
):
    start, pool = harness
    client = start(
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
    )
    a_id = client.reserve_request()
    started = threading.Event()
    stats = []
    a = pool.submit(
        client.generate,
        "a",
        SimpleNamespace(),
        lambda _: started.set(),
        stats.append,
        a_id,
    )
    b = pool.submit(client.generate, "b", SimpleNamespace())
    assert started.wait(5)
    assert not client.stop()
    with client._lock:
        assert client.cancel(a_id)
        assert client.cancel(a_id)
    a.result(timeout=5)
    b.result(timeout=5)
    assert len(stats) == 1 and stats[0].cancelled
    assert stats[0].finish_reason == "stop"
    assert not client.cancel(a_id)
    client.open_session("barrier")
    with client._lock:
        assert not client._requests and not client._controls


def test_cancel_before_submission_is_latched_without_sending(harness):
    start, _ = harness
    client = start(
        """
        b = recv()
        assert b['op'] == 'generate' and b['request_id'] == 2
        send(b, done=True)
    """
    )
    request_id = client.reserve_request()
    assert client.cancel(request_id)
    stats = []
    client.generate(
        "cancelled",
        SimpleNamespace(),
        stats_callback=stats.append,
        request_id=request_id,
    )
    assert len(stats) == 1 and stats[0].cancelled
    client.generate("b", SimpleNamespace())


class _GatedStdin:
    def __init__(self, stream):
        self.stream = stream
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, payload):
        self.entered.set()
        assert self.release.wait(5)
        return self.stream.write(payload)

    def flush(self):
        self.stream.flush()

    def close(self):
        self.release.set()
        self.stream.close()


def test_cancel_queued_generation_and_bound_writer_backlog(harness):
    start, pool = harness
    client = start(
        """
        a = recv()
        assert a['prompt'] == 'a'
        send(a, done=True)
        b = recv()
        assert b['prompt'] == 'last'
        send(b, done=True)
    """,
        max_inflight_requests=2,
        wrap_stdin=_GatedStdin,
    )
    gate = client._proc.stdin
    a = pool.submit(client.generate, "a", SimpleNamespace())
    assert gate.entered.wait(5)
    try:
        for _ in range(20):
            request_id = client.reserve_request()
            stats = []
            b = pool.submit(
                client.generate,
                "cancelled",
                SimpleNamespace(),
                None,
                stats.append,
                request_id,
            )
            assert client.cancel(request_id)
            b.result(timeout=5)
            assert len(stats) == 1 and stats[0].cancelled
        with client._lock:
            assert len(client._writes) <= 2 * client.max_inflight_requests
    finally:
        gate.release.set()
    a.result(timeout=5)
    client.generate("last", SimpleNamespace())


def test_slow_mailbox_overflow_only_fails_its_request(harness, monkeypatch):
    start, pool = harness
    client = start(
        """
        a = recv()
        send(a, token='first')
        b = recv()
        for i in range(20):
            send(a, token=str(i))
        send(b, token='fast')
        send(b, done=True)
        cancel = recv()
        assert cancel['op'] == 'cancel' and cancel['target_request_id'] == a['request_id']
        send(cancel, cancelled=True)
        send(a, done=True, cancelled=True, finish_reason='stop')
    """,
        mailbox_capacity=2,
    )
    callback_entered, release_callback = threading.Event(), threading.Event()
    slow_stats, fast_tokens = [], []

    def slow_callback(_):
        callback_entered.set()
        assert release_callback.wait(5)

    slow = pool.submit(
        client.generate, "slow", SimpleNamespace(), slow_callback, slow_stats.append
    )
    assert callback_entered.wait(5)
    notifications = []
    with client._lock:
        changed = next(iter(client._requests.values())).changed
        notify = changed.notify

        def count_notify(n=1):
            notifications.append(1)
            notify(n)

        monkeypatch.setattr(changed, "notify", count_notify)
    try:
        fast = pool.submit(
            client.generate, "fast", SimpleNamespace(), fast_tokens.append
        )
        fast.result(timeout=5)
        assert fast_tokens == ["fast"]
        assert not slow.done()
        with client._lock:
            assert all(len(s.tokens) <= 2 for s in client._requests.values())
    finally:
        release_callback.set()
    with pytest.raises(WorkerError, match="mailbox overflow") as error:
        slow.result(timeout=5)
    assert error.value.code == "slow_consumer"
    assert slow_stats == []
    # Once overflowed, additional tokens do not enqueue work or wake consumers.
    assert len(notifications) <= 4
    assert client.healthy


@pytest.mark.parametrize(
    "failure",
    [
        "sys.exit(0)",
        "print('not JSON', flush=True)",
        "print('{\\\"request_id\\\":1', flush=True); sys.exit(0)",
        "send(requests[0], token=123)",
        "send(requests[0], done=True, token='conflict')",
        "send(requests[0], done=True, completion_tokens=True)",
        "send(requests[0], done=True, prompt_tokens=-1)",
        "send(requests[0], done=True, prefill_ms='slow')",
        "send(requests[0], done=True, decode_ms=-1)",
        "send(requests[0], done=True, finish_reason='unknown')",
        "send(requests[0], done=True, cancelled=1)",
        "send(requests[0], done=True, cancelled=True, finish_reason='length')",
        "send(requests[0], done=True, session_reset_reason=False)",
        "send(requests[0], done=True, generated_token_ids={})",
        "send(requests[0], done=True, generated_token_ids=[True])",
        "send(requests[0], done=True, generated_token_ids=[-1])",
        "print(json.dumps(dict(request_id=True, done=True)), flush=True)",
        "print(json.dumps(dict(request_id=0, done=True)), flush=True)",
        "print(json.dumps(dict(request_id=2**64, done=True)), flush=True)",
        "print(json.dumps(dict(request_id=99999, done=True)), flush=True)",
    ],
)
def test_eof_or_protocol_failure_settles_all_operations(harness, failure):
    start, pool = harness
    client = start("requests = [recv(), recv(), recv()]\n" + failure)
    operations = [
        pool.submit(client.generate, str(i), SimpleNamespace()) for i in range(2)
    ]
    operations.append(pool.submit(client.open_session, "s"))
    errors = []
    for operation in operations:
        with pytest.raises(WorkerError) as error:
            operation.result(timeout=5)
        errors.append(error.value)
    assert errors[0] is errors[1] is errors[2]
    assert client.failed and not client.healthy
    with pytest.raises(WorkerError) as error:
        client.reserve_request()
    assert error.value is errors[0]


def test_duplicate_terminal_is_protocol_failure_not_a_new_completion(harness):
    start, pool = harness
    client = start(
        """
        a, b = sorted([recv(), recv()], key=lambda r: r['prompt'])
        send(a, done=True)
        send(a, done=True)
    """
    )
    stats = []
    a = pool.submit(client.generate, "a", SimpleNamespace(), None, stats.append)
    b = pool.submit(client.generate, "b", SimpleNamespace())
    a.result(timeout=5)
    with pytest.raises(WorkerError, match="inactive request"):
        b.result(timeout=5)
    assert len(stats) == 1


def test_callback_failure_does_not_poison_other_requests(harness):
    start, pool = harness
    client = start(
        """
        a = recv()
        send(a, token='throw')
        rest = [recv(), recv()]
        c = next(r for r in rest if r['op'] == 'cancel')
        b = next(r for r in rest if r['op'] == 'generate')
        send(c, cancelled=True)
        send(a, done=True, cancelled=True, finish_reason='stop')
        send(b, done=True)
    """
    )

    def fail(_):
        raise ValueError("callback error")

    a = pool.submit(client.generate, "a", SimpleNamespace(), fail)
    with pytest.raises(ValueError, match="callback error"):
        a.result(timeout=5)
    client.generate("b", SimpleNamespace())
    assert client.healthy


def test_capacity_reservations_release_and_uint64_exhaustion(harness):
    start, pool = harness
    client = start("recv()", max_inflight_requests=2)
    a, b = client.reserve_request(), client.reserve_request()
    with pytest.raises(WorkerError) as error:
        client.reserve_request()
    assert error.value.code == "capacity_exhausted"
    assert not client.release_request(999)
    assert client.release_request(b)
    client._next_request_id = _UINT64_MAX
    last = client.reserve_request()
    assert last == _UINT64_MAX
    assert client.release_request(last)
    active = pool.submit(client.generate, "a", SimpleNamespace(), request_id=a)
    with pytest.raises(WorkerError, match="ids exhausted"):
        client.reserve_request()
    with pytest.raises(WorkerError, match="ids exhausted"):
        active.result(timeout=5)
    assert client.failed


def test_generation_shape_and_lifecycle_errors(harness):
    start, _ = harness
    client = start(
        """
        g = recv()
        assert g == dict(op='generate', request_id=1, prompt_segments=[dict(ids=[1,2])], session_id='s', max_new_tokens=8, temperature=.5, top_p=.9, top_k=3, seed=9, stop=['end'])
        send(g, done=True, generated_token_ids=[7], prompt_tokens=2)
        r = recv()
        assert r['op'] == 'reset' and r['request_id'] == 2
        send(r, reset=True)
        c = recv()
        assert c['op'] == 'close' and c['request_id'] == 3
        send(c, closed=True)
        o = recv()
        send(o, error='full', code='capacity_exhausted')
    """
    )
    stats = []
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
    client.generate("ignored", config, stats_callback=stats.append)
    assert stats[0].generated_token_ids == [7]
    client.reset_session("s")
    client.close_session("s")
    with pytest.raises(WorkerError) as error:
        client.open_session("s")
    assert error.value.code == "capacity_exhausted"
    assert client.healthy


@pytest.mark.parametrize("multiplexed", [False, True])
@pytest.mark.parametrize("generated_ids", [None, [], [7, 8]])
def test_generated_token_ids_presence_survives_both_transports(
    harness, multiplexed, generated_ids
):
    start, _ = harness
    metadata = {} if generated_ids is None else {"generated_token_ids": generated_ids}
    client = start(
        f"""
        print(json.dumps(dict(ready=True, multiplexed={multiplexed!r})), flush=True)
        request = recv()
        terminal = dict(done=True, finish_reason='stop', **{metadata!r})
        if {multiplexed!r}:
            send(request, **terminal)
        else:
            print(json.dumps(terminal), flush=True)
        """,
        negotiate=True,
    )
    assert WorkerStats().generated_token_ids is None
    stats = []
    client.generate("hi", SimpleNamespace(), stats_callback=stats.append)
    assert len(stats) == 1
    assert stats[0].generated_token_ids == generated_ids
    assert client.healthy


def test_positive_negotiation_closes_unused_control_pipe(harness):
    start, _ = harness
    client = start(
        """
        print(json.dumps(dict(ready=True, multiplexed=True, max_named_sessions=3, max_inflight_requests=5)), flush=True)
        fd = os.environ.get('EXECUTORCH_LLM_WORKER_CONTROL_FD')
        if fd is not None:
            assert os.read(int(fd), 8) == b''
            os.close(int(fd))
        g = recv()
        assert g['op'] == 'generate' and 'cancel_request_id' not in g
        send(g, done=True)
    """,
        negotiate=True,
    )
    assert isinstance(client, MultiplexedWorkerClient)
    assert client.supports_multiplexing and client.supports_cancel
    assert client.max_named_sessions == 3 and client.max_inflight_requests == 5
    client.generate("hi", SimpleNamespace())


@pytest.mark.parametrize("capability", ["", ", multiplexed=False"])
def test_legacy_negotiation_and_request_shape_stay_unchanged(harness, capability):
    start, _ = harness
    client = start(
        "print(json.dumps(dict(ready=True"
        + capability
        + ")), flush=True)\n"
        + """
g = recv()
assert 'op' not in g and 'request_id' not in g
print(json.dumps(dict(done=True)), flush=True)
""",
        negotiate=True,
    )
    assert isinstance(client, WorkerClient)
    assert not client.supports_multiplexing
    client.generate("hi", SimpleNamespace())


def test_close_wakes_all_waiters_and_stops_transport_threads(harness):
    start, pool = harness
    client = start(
        """
        requests = [recv(), recv()]
        a = next(r for r in requests if r['op'] == 'generate')
        send(a, token='started')
    """
    )
    started = threading.Event()
    a = pool.submit(client.generate, "a", SimpleNamespace(), lambda _: started.set())
    b = pool.submit(client.open_session, "s")
    assert started.wait(5)
    client.close()
    for future in (a, b):
        with pytest.raises(WorkerError, match="closed"):
            future.result(timeout=5)
    client.close()
    assert client.closed and not client.failed
    assert not client._reader.is_alive() and not client._writer.is_alive()


@pytest.mark.parametrize("request_id", [True, 0, -1, 2**64, "1"])
def test_invalid_request_handles(harness, request_id):
    start, _ = harness
    client = start("pass")
    with pytest.raises(WorkerError, match="request_id"):
        client.cancel(request_id)
    with pytest.raises(WorkerError, match="request_id"):
        client.generate("hi", SimpleNamespace(), request_id=request_id)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_inflight_requests": 0},
        {"max_inflight_requests": True},
        {"max_named_sessions": -1},
    ],
)
def test_invalid_negotiated_limits_are_rejected(harness, limits):
    start, _ = harness
    with pytest.raises(WorkerError, match="invalid"):
        start(
            "print(json.dumps(dict(ready=True, multiplexed=True, **"
            + repr(limits)
            + ")), flush=True)",
            negotiate=True,
        )


def test_oversized_response_fails_without_unbounded_buffering(harness):
    start, pool = harness
    client = start("a = recv(); send(a, token='x' * 256)", max_message_chars=128)
    future = pool.submit(client.generate, "a", SimpleNamespace())
    with pytest.raises(WorkerError, match="oversized"):
        future.result(timeout=5)
    assert client.failed


class _InterruptedStdin(_GatedStdin):
    def write(self, payload):
        self.entered.set()
        assert self.release.wait(5)
        self.stream.write(payload[:8])
        self.stream.flush()
        raise KeyboardInterrupt("interrupted mid-frame")


def test_interrupted_partial_write_permanently_fails_all_requests(harness):
    start, pool = harness
    client = start("pass", wrap_stdin=_InterruptedStdin)
    gate = client._proc.stdin
    a = pool.submit(client.generate, "a", SimpleNamespace())
    assert gate.entered.wait(5)
    b = pool.submit(client.generate, "b", SimpleNamespace())
    gate.release.set()
    for future in (a, b):
        with pytest.raises(WorkerError, match="write failed"):
            future.result(timeout=5)
    assert client.failed
    with pytest.raises(WorkerError, match="write failed"):
        client.reserve_request()


@pytest.mark.parametrize("extra_bytes", [0, 1])
@pytest.mark.parametrize("escaped", ['"\\\n', "\u00e9\U0001f600"])
def test_outbound_frame_limit_counts_encoded_jsonl(harness, extra_bytes, escaped):
    start, pool = harness
    limit = 1024 * 1024
    client = start(
        f"""
        line = sys.stdin.readline()
        assert len(line.encode('utf-8')) == {limit}
        request = json.loads(line)
        assert line == json.dumps(request, allow_nan=False) + '\\n'
        assert request['request_id'] == { _UINT64_MAX }
        send(request, done=True)
        """,
        max_message_chars=128,
    )
    client._next_request_id = _UINT64_MAX
    request = {"op": "generate", "prompt": escaped, "request_id": _UINT64_MAX}
    overhead = len((json.dumps(request) + "\n").encode("utf-8"))
    request["prompt"] += "x" * (limit - overhead + extra_bytes)
    expected = json.dumps(request) + "\n"
    assert len(expected.encode("utf-8")) == limit + extra_bytes
    del request["request_id"]
    if extra_bytes:
        with pytest.raises(WorkerError) as error:
            client._submit(request)
        assert error.value.code == "invalid_argument"
    else:
        state = client._submit(request)
        pool.submit(client._consume, state).result(timeout=5)
    with client._lock:
        assert not client._requests and not client._writes and not client._controls
    assert client.healthy


@pytest.mark.parametrize("reserved", [False, True])
@pytest.mark.parametrize("field", ["prompt", "stop", "prompt_segments", "session_id"])
def test_oversized_generation_settles_only_its_reservation(
    harness, monkeypatch, reserved, field
):
    start, pool = harness
    client = start("pass", max_inflight_requests=2)
    peer_id = client.reserve_request()
    request_id = client.reserve_request() if reserved else client._next_request_id
    entered = threading.Event()
    if reserved:
        state = client._requests[request_id]
        original_wait = state.changed.wait_for

        def observe_wait(*args, **kwargs):
            entered.set()
            return original_wait(*args, **kwargs)

        monkeypatch.setattr(state.changed, "wait_for", observe_wait)
        waiter = pool.submit(client.wait_for_request, request_id)
        assert entered.wait(5)
    oversized = "x" * (1024 * 1024)
    prompt = oversized if field == "prompt" else "small"
    config = SimpleNamespace()
    if field == "stop":
        config.stop = [oversized]
    elif field == "prompt_segments":
        config.prompt_segments = [{"text": oversized}]
    elif field == "session_id":
        config.session_id = oversized
    with pytest.raises(WorkerError) as error:
        client.generate(prompt, config, request_id=request_id if reserved else None)
    assert error.value.code == "invalid_argument"
    if reserved:
        waiter.result(timeout=5)
        assert state.wire_done and state.consumed
    pool.submit(client.wait_for_request, request_id).result(timeout=5)
    with client._lock:
        assert list(client._requests) == [peer_id]
        assert not client._writes and not client._controls
    replacement = client.reserve_request()
    assert client.release_request(replacement)
    assert client.release_request(peer_id)
    assert client.healthy


def test_oversized_generation_and_lifecycle_never_reach_active_peer(harness):
    start, pool = harness
    client = start(
        """
        active = recv()
        send(active, token='started')
        barrier = recv()
        assert barrier['op'] == 'open' and barrier['session_id'] == 'barrier'
        send(barrier, opened=True)
        send(active, token='survived')
        send(active, done=True)
        """,
        max_inflight_requests=2,
    )
    started = threading.Event()
    tokens = []

    def on_token(token):
        tokens.append(token)
        started.set()

    active = pool.submit(client.generate, "active", SimpleNamespace(), on_token)
    assert started.wait(5)
    oversized = "x" * (1024 * 1024)
    operations = [
        (client.generate, (oversized, SimpleNamespace())),
        (client.generate, ("small", SimpleNamespace(stop=[oversized]))),
        (client.open_session, (oversized,)),
        (client.reset_session, (oversized,)),
        (client.close_session, (oversized,)),
    ]
    for operation, args in operations:
        with pytest.raises(WorkerError) as error:
            pool.submit(operation, *args).result(timeout=5)
        assert error.value.code == "invalid_argument"
        with client._lock:
            assert len(client._requests) == 1
            assert not client._writes and not client._controls
        assert client.healthy and not active.done()
    pool.submit(client.open_session, "barrier").result(timeout=5)
    active.result(timeout=5)
    assert tokens == ["started", "survived"]
    assert client.healthy


def test_cancellation_encodes_before_mutating_control_state(harness, monkeypatch):
    start, _ = harness
    client = start("pass")
    request_id = client.reserve_request()
    state = client._requests[request_id]
    with client._lock:
        state.sending = True
        state.cancel_pending = True
        client._next_request_id = _UINT64_MAX
        encode = client._encode_request
        frames = []

        def observe_encode(request):
            assert state.cancel_pending and not state.cancel_requested
            assert not client._controls and not client._writes
            payload = encode(request)
            frames.append(payload)
            return payload

        monkeypatch.setattr(client, "_encode_request", observe_encode)
        assert client.cancel(request_id)
        assert len(frames) == 1
        assert json.loads(frames[0]) == {
            "op": "cancel",
            "request_id": _UINT64_MAX,
            "target_request_id": request_id,
        }
        assert len(frames[0].encode("utf-8")) < 1024 * 1024
        assert not state.cancel_pending and state.cancel_requested
        assert list(client._writes) == [(_UINT64_MAX, frames[0])]


def test_bad_caller_payload_releases_only_its_own_reservation(harness):
    start, _ = harness
    client = start("a = recv(); send(a, done=True)", max_inflight_requests=1)
    with pytest.raises(TypeError):
        client.generate(object(), SimpleNamespace())
    client.generate("valid", SimpleNamespace())
    assert client.healthy


class _GatedStdout(_GatedStdin):
    def readline(self, limit):
        self.entered.set()
        assert self.release.wait(5)
        return self.stream.readline(limit)


def test_interrupted_read_permanently_fails_all_requests(harness):
    class InterruptedStdout(_GatedStdin):
        def readline(self, limit):
            self.entered.set()
            assert self.release.wait(5)
            raise KeyboardInterrupt("interrupted read")

    start, pool = harness
    client = start("pass", wrap_stdout=InterruptedStdout)
    gate = client._proc.stdout
    assert gate.entered.wait(5)
    requests = [
        client._submit({"op": "generate", "prompt": prompt}) for prompt in ("a", "b")
    ]
    requests.append(
        client._submit({"op": "open", "session_id": "s"}, op="open", ack="opened")
    )
    operations = [pool.submit(client._consume, request) for request in requests]
    gate.release.set()
    errors = []
    for operation in operations:
        with pytest.raises(WorkerError, match="read failed: interrupted read") as error:
            operation.result(timeout=5)
        errors.append(error.value)
    assert errors[0] is errors[1] is errors[2]
    assert all(request.wire_done for request in requests)
    assert client.failed and not client.healthy
    with pytest.raises(WorkerError) as error:
        client.reserve_request()
    assert error.value is errors[0]
    with client._lock:
        assert not client._requests and not client._writes and not client._controls


class _FlushThenGateStdin(_GatedStdin):
    def write(self, payload):
        written = self.stream.write(payload)
        self.stream.flush()
        self.entered.set()
        assert self.release.wait(5)
        return written


def test_unsent_cancellation_ack_is_a_protocol_failure(harness):
    start, _ = harness
    client = start(
        "a = recv(); send(dict(request_id=2), cancelled=True)",
        wrap_stdin=_FlushThenGateStdin,
        wrap_stdout=_GatedStdout,
    )
    writer, reader = client._proc.stdin, client._proc.stdout
    state = client._submit({"op": "generate", "prompt": "a"})
    assert writer.entered.wait(5)
    try:
        assert client.cancel(state.request_id)
        reader.release.set()
        with state.changed:
            assert state.changed.wait_for(lambda: state.terminal is not None, timeout=5)
            assert isinstance(state.terminal, WorkerError)
            assert "unexpected cancellation response" in str(state.terminal)
        assert client.failed
    finally:
        reader.release.set()
        writer.release.set()


def test_reader_overflow_does_not_wait_for_blocked_writer(harness):
    start, _ = harness
    client = start(
        """
        a = recv()
        for i in range(10):
            send(a, token=str(i))
        c = recv()
        assert c['op'] == 'cancel' and c['target_request_id'] == a['request_id']
        send(c, cancelled=True)
        send(a, done=True, cancelled=True, finish_reason='stop')
        barrier = recv()
        send(barrier, opened=True)
    """,
        mailbox_capacity=1,
        wrap_stdin=_FlushThenGateStdin,
    )
    gate = client._proc.stdin
    state = client._submit({"op": "generate", "prompt": "a"})
    assert gate.entered.wait(5)
    try:
        with state.changed:
            assert state.changed.wait_for(lambda: state.terminal is not None, timeout=5)
            assert isinstance(state.terminal, WorkerError)
            assert len(client._controls) == 1
        with pytest.raises(WorkerError, match="mailbox overflow"):
            client._consume(state)
    finally:
        gate.release.set()
    client.open_session("barrier")
    assert client.healthy
    with client._lock:
        assert not client._requests and not client._controls


@pytest.mark.parametrize("settlement", ["done", "eof", "protocol", "close"])
def test_wire_wait_survives_local_failure_and_wakes_on_settlement(
    harness, monkeypatch, settlement
):
    from concurrent.futures import ThreadPoolExecutor

    start, _ = harness
    client = start(
        f"""
        g = recv()
        send(g, token='fail')
        c = recv()
        send(c, cancelled=True)
        gate = recv()
        if {settlement!r} == 'done':
            send(g, done=True, cancelled=True, finish_reason='stop')
            send(gate, opened=True)
        elif {settlement!r} == 'eof':
            sys.exit(0)
        else:
            print('invalid-json', flush=True)
    """
    )
    request_id = client.reserve_request()

    def fail(_):
        raise ValueError("local failure")

    with pytest.raises(ValueError, match="local failure"):
        client.generate("hi", SimpleNamespace(), fail, request_id=request_id)
    with client._lock:
        state = client._requests[request_id]
        assert state.consumed and not state.wire_done
    entered = threading.Event()
    original_wait = state.changed.wait_for

    def observe_wait(*args, **kwargs):
        entered.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(state.changed, "wait_for", observe_wait)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            future = pool.submit(client.wait_for_request, request_id)
            assert entered.wait(5)
            assert not future.done()
            if settlement == "close":
                client.close()
            elif settlement == "done":
                client.open_session("release")
            else:
                with pytest.raises(WorkerError):
                    client.open_session("release")
            future.result(timeout=5)
            assert state.wire_done
            client.wait_for_request(request_id)  # Retired IDs need no history.
            with client._lock:
                assert not client._requests
        finally:
            client.close()


def test_wire_wait_does_not_steal_consumer_token_notifications(harness, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    start, _ = harness
    client = start(
        """
        g = recv()
        send(g, token='ready')
        gate = recv()
        send(g, done=True)
        send(gate, opened=True)
    """
    )
    request_id = client.reserve_request()
    state = client._requests[request_id]
    entered = threading.Event()
    received = threading.Event()
    original_wait = state.changed.wait_for

    def observe_wait(*args, **kwargs):
        entered.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(state.changed, "wait_for", observe_wait)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            waiter = pool.submit(client.wait_for_request, request_id)
            assert entered.wait(5)
            consumer = pool.submit(
                client.generate,
                "hi",
                SimpleNamespace(),
                lambda _: received.set(),
                None,
                request_id,
            )
            assert received.wait(5)
            assert not waiter.done()
            client.open_session("release")
            consumer.result(timeout=5)
            waiter.result(timeout=5)
        finally:
            client.close()


@pytest.mark.parametrize("failure", ["callback", "overflow", "cancel", "stop"])
def test_cancel_retries_after_delayed_acks_free_control_budget(harness, failure):
    start, _ = harness
    client = start(
        f"""
        delayed = []
        for _ in range(2):
            g = recv()
            send(g, token='fail')
            c = recv()
            assert c['op'] == 'cancel' and c['target_request_id'] == g['request_id']
            delayed.append(c)
            send(g, done=True, cancelled=True, finish_reason='stop')
        barrier = recv()
        send(barrier, opened=True)
        target = recv()
        for _ in range({10 if failure == 'overflow' else 1}):
            send(target, token='fail')
        release = recv()
        assert release['op'] == 'open'
        for c in delayed:
            send(c, cancelled=True)
        send(release, opened=True)
        retried = recv()
        assert retried['op'] == 'cancel' and retried['target_request_id'] == target['request_id']
        send(retried, cancelled=True)
        send(target, done=True, cancelled=True, finish_reason='stop')
        final = recv()
        send(final, opened=True)
    """,
        max_inflight_requests=2,
        mailbox_capacity=1,
    )

    def fail_callback(_):
        raise ValueError("callback failed")

    for _ in range(2):
        with pytest.raises(ValueError, match="callback failed"):
            client.generate("seed", SimpleNamespace(), fail_callback)
    client.open_session("seed_barrier")
    target_id = client.reserve_request()
    if failure == "callback":
        with pytest.raises(ValueError, match="callback failed"):
            client.generate(
                "target", SimpleNamespace(), fail_callback, request_id=target_id
            )
    else:
        state = client._submit(
            {"op": "generate", "prompt": "target"}, request_id=target_id
        )
        with state.changed:
            if failure == "overflow":
                assert state.changed.wait_for(
                    lambda: state.terminal is not None, timeout=5
                )
            else:
                assert state.changed.wait_for(lambda: bool(state.tokens), timeout=5)
        if failure == "overflow":
            with pytest.raises(WorkerError, match="mailbox overflow"):
                client._consume(state)
        else:
            for _ in range(20):
                assert (
                    client.cancel(target_id) if failure == "cancel" else client.stop()
                )
    with client._lock:
        state = client._requests[target_id]
        assert state.consumed == (failure in ("callback", "overflow"))
        assert state.cancel_pending and not state.wire_done
        assert len(client._controls) == 2
        assert len(client._writes) <= 4
    client.open_session("release_acks")
    client.open_session("finished")
    if failure in ("cancel", "stop"):
        stats = []
        client._consume(state, stats_callback=stats.append)
        assert len(stats) == 1 and stats[0].cancelled
    with client._lock:
        assert not client._requests and not client._controls
    assert client.healthy
