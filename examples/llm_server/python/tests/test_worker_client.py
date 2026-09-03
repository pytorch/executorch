# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the generic WorkerClient JSONL protocol (no model/GPU/subprocess).

A fake process stands in for the C++ worker: it records what the client writes
and replays a scripted sequence of JSONL response lines.
"""

import errno
import json
import os
import subprocess
import threading
from dataclasses import dataclass, field

import pytest
from executorch.examples.llm_server.python import worker_client as worker_client_mod
from executorch.examples.llm_server.python.worker_client import (
    spawn_worker,
    WorkerClient,
    WorkerError,
)


class _FakeStdin:
    def __init__(self):
        self.written = []
        self.closed = False

    def write(self, s):
        self.written.append(s)
        return len(s)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, stdout_lines, returncode=None, wait_results=None):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self._returncode = returncode
        self._wait_results = list(wait_results or [])
        self.terminated = False
        self.killed = False
        self.waited = False
        self.events = []
        self.wait_timeouts = []

    def poll(self):
        return self._returncode

    @property
    def returncode(self):
        return self._returncode

    def terminate(self):
        self.events.append("terminate")
        self.terminated = True
        self._returncode = -15

    def kill(self):
        self.events.append("kill")
        self.killed = True
        self._returncode = -9

    def wait(self, timeout=None):
        self.events.append("wait")
        self.waited = True
        self.wait_timeouts.append(timeout)
        if self._wait_results:
            result = self._wait_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            self._returncode = result
        return self._returncode


@dataclass
class _Cfg:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0
    stop: list = field(default_factory=list)
    session_id: str = None


def _lines(*objs):
    return [json.dumps(o) + "\n" for o in objs]


def test_generate_streams_tokens_then_stats():
    proc = _FakeProc(
        _lines(
            {"token": "Hello"},
            {"token": " world"},
            {"done": True, "prompt_tokens": 4, "completion_tokens": 2},
        )
    )
    client = WorkerClient(proc)
    out, stats = [], {}
    client.generate(
        "hi",
        _Cfg(temperature=0.7, top_p=0.8, top_k=16, seed=123),
        token_callback=out.append,
        stats_callback=lambda s: stats.update(
            prompt=s.num_prompt_tokens, gen=s.num_generated_tokens
        ),
    )
    assert "".join(out) == "Hello world"
    assert stats == {"prompt": 4, "gen": 2}
    # The request carried prompt + sampling, one JSON line.
    sent = json.loads(proc.stdin.written[0])
    assert sent == {
        "prompt": "hi",
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 16,
        "seed": 123,
        "stop": [],
    }


def test_generate_forwards_stop_sequences():
    proc = _FakeProc(_lines({"done": True, "prompt_tokens": 1, "completion_tokens": 0}))
    WorkerClient(proc).generate("hi", _Cfg(stop=["STOP", "\n\n"]))
    sent = json.loads(proc.stdin.written[0])
    assert sent["stop"] == ["STOP", "\n\n"]


def test_generate_reports_finish_reason():
    proc = _FakeProc(
        _lines(
            {"token": "hi"},
            {
                "done": True,
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "finish_reason": "length",
            },
        )
    )
    seen = {}
    WorkerClient(proc).generate(
        "hi", _Cfg(), stats_callback=lambda s: seen.update(fr=s.finish_reason)
    )
    assert seen["fr"] == "length"


def test_error_message_raises_worker_error():
    proc = _FakeProc(_lines({"error": "boom"}))
    with pytest.raises(WorkerError, match="boom"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_generate_non_json_raises_worker_error():
    proc = _FakeProc(["worker log on stdout\n"])
    with pytest.raises(WorkerError, match="invalid worker JSON"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_generate_unexpected_json_raises_worker_error():
    proc = _FakeProc(_lines({"progress": "loading"}))
    with pytest.raises(WorkerError, match="unexpected worker response"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_exit_mid_request_raises():
    proc = _FakeProc([])  # readline() -> "" means the worker exited
    with pytest.raises(WorkerError, match="exited mid-request"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_generate_on_dead_worker_raises():
    proc = _FakeProc([], returncode=1)
    with pytest.raises(WorkerError, match="worker exited"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_generate_includes_session_id_when_set():
    proc = _FakeProc(_lines({"done": True, "prompt_tokens": 1, "completion_tokens": 0}))
    WorkerClient(proc).generate("hi", _Cfg(session_id="abc"))
    assert json.loads(proc.stdin.written[0])["session_id"] == "abc"


def test_generate_omits_session_id_when_unset():
    proc = _FakeProc(_lines({"done": True, "prompt_tokens": 1, "completion_tokens": 0}))
    WorkerClient(proc).generate("hi", _Cfg())
    assert "session_id" not in json.loads(proc.stdin.written[0])


def test_open_session_sends_op_and_acks():
    proc = _FakeProc(_lines({"opened": True, "session_id": "abc"}))
    WorkerClient(proc).open_session("abc")
    assert json.loads(proc.stdin.written[0]) == {"op": "open", "session_id": "abc"}


def test_open_session_capacity_error_carries_code():
    proc = _FakeProc(_lines({"error": "full", "code": "capacity_exhausted"}))
    with pytest.raises(WorkerError) as ei:
        WorkerClient(proc).open_session("abc")
    assert ei.value.code == "capacity_exhausted"


def test_op_non_json_raises_worker_error():
    proc = _FakeProc(["worker log on stdout\n"])
    with pytest.raises(WorkerError, match="invalid worker JSON"):
        WorkerClient(proc).open_session("abc")


def test_close_session_sends_op_and_acks():
    proc = _FakeProc(_lines({"closed": True, "session_id": "abc"}))
    WorkerClient(proc).close_session("abc")
    assert json.loads(proc.stdin.written[0]) == {"op": "close", "session_id": "abc"}


def test_reset_session_sends_op_and_acks():
    proc = _FakeProc(_lines({"reset": True, "session_id": "abc"}))
    WorkerClient(proc).reset_session("abc")
    assert json.loads(proc.stdin.written[0]) == {"op": "reset", "session_id": "abc"}


def test_generate_parses_warm_resume_metrics():
    proc = _FakeProc(
        _lines(
            {"token": "hi"},
            {
                "done": True,
                "prompt_tokens": 100,
                "completion_tokens": 1,
                "finish_reason": "stop",
                "reused_prompt_tokens": 90,
                "prefilled_prompt_tokens": 10,
                "session_reset_reason": "exact_prefix",
                "prefill_ms": 12.5,
                "decode_ms": 25.0,
                "total_ms": 40.0,
                "prefill_tok_s": 800.0,
                "decode_tok_s": 40.0,
                "vision_encoder_ms": 123.5,
            },
        )
    )
    seen = {}
    WorkerClient(proc).generate(
        "hi", _Cfg(session_id="s"), stats_callback=lambda s: seen.update(s=s)
    )
    st = seen["s"]
    assert st.reused_prompt_tokens == 90
    assert st.prefilled_prompt_tokens == 10
    assert st.session_reset_reason == "exact_prefix"
    assert st.prefill_ms == 12.5
    assert st.decode_ms == 25.0
    assert st.total_ms == 40.0
    assert st.prefill_tok_s == 800.0
    assert st.decode_tok_s == 40.0
    assert st.vision_encoder_ms == 123.5


def test_generate_defaults_missing_vision_encoder_metric_to_none():
    proc = _FakeProc(_lines({"done": True}))
    seen = {}
    WorkerClient(proc).generate(
        "hi", _Cfg(), stats_callback=lambda stats: seen.update(stats=stats)
    )
    assert seen["stats"].vision_encoder_ms is None


def test_spawn_worker_waits_for_ready():
    proc = _FakeProc(_lines({"ready": True, "max_named_sessions": 3}))
    client = spawn_worker(
        ["/fake/worker", "--model_path", "m"], popen=lambda *a, **k: proc
    )
    assert isinstance(client, WorkerClient)
    assert client.max_named_sessions == 3


def test_spawn_worker_not_ready_raises():
    proc = _FakeProc(_lines({"oops": True}))
    with pytest.raises(WorkerError, match="did not report ready"):
        spawn_worker(["/fake/worker"], popen=lambda *a, **k: proc)


def test_spawn_worker_non_json_raises_worker_error():
    proc = _FakeProc(["worker log on stdout\n"])
    with pytest.raises(WorkerError, match="invalid worker JSON"):
        spawn_worker(["/fake/worker"], popen=lambda *a, **k: proc)


def test_spawn_worker_no_output_raises():
    proc = _FakeProc([])
    with pytest.raises(WorkerError, match="failed to start"):
        spawn_worker(["/fake/worker"], popen=lambda *a, **k: proc)


def test_reservation_is_active_before_generation_submission():
    read_fd, write_fd = os.pipe()
    proc = _FakeProc(_lines({"done": True}))
    client = WorkerClient(proc, control_fd=write_fd)
    try:
        request_id = client.reserve_request()
        assert request_id == 1
        assert client.stop() is True
        assert os.read(read_fd, 8) == (1).to_bytes(8, "little")

        client.generate("hi", _Cfg(), request_id=request_id)
        assert json.loads(proc.stdin.written[0])["cancel_request_id"] == request_id
    finally:
        client.close()
        os.close(read_fd)


def test_release_request_handles_failed_executor_submission():
    client = WorkerClient(_FakeProc([]))
    request_id = client.reserve_request()
    assert client.release_request(request_id + 1) is False
    assert client.release_request(request_id) is True
    assert client.reserve_request() == request_id + 1
    client.close()


def test_stop_writes_one_little_endian_frame_and_caches_success(monkeypatch):
    read_fd, write_fd = os.pipe()
    client = WorkerClient(_FakeProc([]), control_fd=write_fd)
    frames = []

    def record_write(fd, frame):
        assert fd == write_fd
        frames.append(frame)
        return len(frame)

    monkeypatch.setattr(os, "write", record_write)
    try:
        request_id = client.reserve_request()
        with client._lock:
            assert client.stop() is True
            assert client.stop() is True
        assert frames == [request_id.to_bytes(8, "little")]
    finally:
        client.close()
        os.close(read_fd)


def test_stop_is_false_for_unsupported_worker_and_old_request_shape():
    proc = _FakeProc(_lines({"done": True}))
    client = WorkerClient(proc)
    request_id = client.reserve_request()
    assert client.supports_cancel is False
    assert client.stop() is False
    client.generate("hi", _Cfg(), request_id=request_id)
    assert "cancel_request_id" not in json.loads(proc.stdin.written[0])
    client.close()


def test_stop_eagain_is_cached_without_poisoning_channel(monkeypatch):
    read_fd, write_fd = os.pipe()
    client = WorkerClient(_FakeProc([]), control_fd=write_fd)
    attempts = []

    def would_block(fd, frame):
        attempts.append((fd, frame))
        raise BlockingIOError(errno.EAGAIN, "full")

    monkeypatch.setattr(os, "write", would_block)
    try:
        client.reserve_request()
        assert client.stop() is False
        assert client.stop() is False
        assert len(attempts) == 1
        assert client.supports_cancel is True
    finally:
        client.close()
        os.close(read_fd)


@pytest.mark.parametrize("written", [0, 4])
def test_stop_short_write_is_cached_and_poisons_channel(monkeypatch, written):
    read_fd, write_fd = os.pipe()
    client = WorkerClient(_FakeProc([]), control_fd=write_fd)
    attempts = []

    def short_write(fd, frame):
        attempts.append((fd, frame))
        return written

    monkeypatch.setattr(os, "write", short_write)
    client.reserve_request()
    assert client.stop() is False
    assert client.stop() is False
    assert len(attempts) == 1
    assert client.supports_cancel is False
    with pytest.raises(OSError):
        os.fstat(write_fd)
    client.close()
    os.close(read_fd)


def test_stop_epipe_is_cached_and_poisons_channel(monkeypatch):
    read_fd, write_fd = os.pipe()
    client = WorkerClient(_FakeProc([]), control_fd=write_fd)
    attempts = []

    def broken_pipe(fd, frame):
        attempts.append((fd, frame))
        raise BrokenPipeError(errno.EPIPE, "closed")

    monkeypatch.setattr(os, "write", broken_pipe)
    client.reserve_request()
    assert client.stop() is False
    assert client.stop() is False
    assert len(attempts) == 1
    assert client.supports_cancel is False
    with pytest.raises(OSError):
        os.fstat(write_fd)
    client.close()
    os.close(read_fd)


def test_request_ids_reach_uint64_max_then_fail_permanently():
    client = WorkerClient(_FakeProc([]))
    client._next_request_id = worker_client_mod._UINT64_MAX
    request_id = client.reserve_request()
    assert request_id == worker_client_mod._UINT64_MAX
    assert client.release_request(request_id) is True

    with pytest.raises(WorkerError, match="request ids exhausted") as first:
        client.reserve_request()
    with pytest.raises(WorkerError) as second:
        client.reserve_request()
    assert second.value is first.value
    assert client.failed is True
    assert client.healthy is False
    client.close()


def test_generation_exit_only_clears_matching_active_request():
    class _BlockingStdout:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.closed = False

        def readline(self):
            self.started.set()
            assert self.release.wait(timeout=5)
            return json.dumps({"done": True}) + "\n"

        def close(self):
            self.closed = True

    read_fd, write_fd = os.pipe()
    proc = _FakeProc([])
    proc.stdout = _BlockingStdout()
    client = WorkerClient(proc, control_fd=write_fd)
    request_id = client.reserve_request()
    errors = []

    def generate():
        try:
            client.generate("hi", _Cfg(), request_id=request_id)
        except (AssertionError, WorkerError) as error:  # pragma: no cover
            errors.append(error)

    thread = threading.Thread(target=generate)
    thread.start()
    assert proc.stdout.started.wait(timeout=5)
    successor_id = request_id + 1
    with client._state_lock:
        client._active_request_id = successor_id
        client._cancel_delivery = None
    proc.stdout.release.set()
    thread.join(timeout=5)

    try:
        assert not thread.is_alive()
        assert errors == []
        assert client.stop() is True
        assert os.read(read_fd, 8) == successor_id.to_bytes(8, "little")
    finally:
        client.close()
        os.close(read_fd)


def test_protocol_failure_is_permanent_and_fail_fast():
    proc = _FakeProc(["worker log on stdout\n", *_lines({"done": True})])
    client = WorkerClient(proc)
    with pytest.raises(WorkerError, match="invalid worker JSON") as first:
        client.generate("hi", _Cfg())
    with pytest.raises(WorkerError) as second:
        client.generate("again", _Cfg())
    with pytest.raises(WorkerError) as third:
        client.open_session("session")

    assert second.value is first.value
    assert third.value is first.value
    assert len(proc.stdin.written) == 1
    assert client.failed is True
    assert client.closed is False
    assert client.healthy is False
    assert client.stop() is False
    client.abort()


def test_abort_is_idempotent_and_terminates_then_kills_and_reaps():
    timeout = subprocess.TimeoutExpired("worker", 5)
    proc = _FakeProc([], wait_results=[timeout, -9])
    stdin = proc.stdin
    stdout = proc.stdout
    read_fd, write_fd = os.pipe()
    client = WorkerClient(proc, control_fd=write_fd)

    client.abort()
    first_events = list(proc.events)
    client.abort()

    assert first_events == ["terminate", "wait", "kill", "wait"]
    assert proc.events == first_events
    assert proc.wait_timeouts == [5, 5]
    assert proc.stdin is None
    assert proc.stdout is None
    assert stdin.closed is True
    assert stdout.closed is True
    assert client.failed is True
    assert client.healthy is False
    with pytest.raises(OSError):
        os.fstat(write_fd)
    os.close(read_fd)


def test_concurrent_abort_waits_for_cleanup_owner(monkeypatch):
    proc = _FakeProc([])
    client = WorkerClient(proc)
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()

    def blocking_shutdown(process, streams):
        assert process is proc
        assert streams == (proc.stdin, proc.stdout)
        cleanup_started.set()
        assert release_cleanup.wait(timeout=5)
        return True

    monkeypatch.setattr(worker_client_mod, "_shutdown_process", blocking_shutdown)
    first = threading.Thread(target=client.abort)
    second = threading.Thread(target=client.abort)
    first.start()
    assert cleanup_started.wait(timeout=5)
    second.start()
    second.join(timeout=0.05)
    assert second.is_alive()

    release_cleanup.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive()
    assert not second.is_alive()


def test_abort_retries_cleanup_after_failed_final_reap():
    timeout = subprocess.TimeoutExpired("worker", 5)
    proc = _FakeProc([], wait_results=[timeout, timeout, 0])
    client = WorkerClient(proc)

    client.abort()
    assert proc.events == ["terminate", "wait", "kill", "wait"]
    client.abort()
    assert proc.events == ["terminate", "wait", "kill", "wait", "wait"]
    client.abort()
    assert proc.events == ["terminate", "wait", "kill", "wait", "wait"]


def test_close_is_idempotent_and_invalidates_all_transports():
    proc = _FakeProc([])
    stdin = proc.stdin
    stdout = proc.stdout
    read_fd, write_fd = os.pipe()
    client = WorkerClient(proc, control_fd=write_fd)

    client.close()
    first_events = list(proc.events)
    client.close()

    assert proc.events == first_events == ["terminate", "wait"]
    assert proc.stdin is None
    assert proc.stdout is None
    assert stdin.closed is True
    assert stdout.closed is True
    assert client.closed is True
    assert client.failed is False
    assert client.healthy is False
    with pytest.raises(WorkerError, match="closed"):
        client.reserve_request()
    with pytest.raises(OSError):
        os.fstat(write_fd)
    os.close(read_fd)


@pytest.mark.skipif(os.name != "posix", reason="pass_fds is POSIX-only")
def test_spawn_worker_negotiates_nonblocking_control_pipe_and_cleans_on_close(
    monkeypatch,
):
    descriptors = os.pipe()
    proc = _FakeProc(_lines({"ready": True, "supports_cancel": True}))
    captured = {}
    monkeypatch.setattr(os, "pipe", lambda: descriptors)

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return proc

    client = spawn_worker(["/fake/worker"], popen=fake_popen)
    read_fd, write_fd = descriptors
    assert captured["pass_fds"] == (read_fd,)
    assert captured["env"]["EXECUTORCH_LLM_WORKER_CONTROL_FD"] == str(read_fd)
    assert os.get_blocking(write_fd) is False
    with pytest.raises(OSError):
        os.fstat(read_fd)

    client.close()
    with pytest.raises(OSError):
        os.fstat(write_fd)


@pytest.mark.skipif(os.name != "posix", reason="pass_fds is POSIX-only")
def test_spawn_worker_popen_failure_closes_both_control_descriptors(monkeypatch):
    descriptors = os.pipe()
    monkeypatch.setattr(os, "pipe", lambda: descriptors)

    def fail_popen(*args, **kwargs):
        raise RuntimeError("spawn failed")

    with pytest.raises(RuntimeError, match="spawn failed"):
        spawn_worker(["/fake/worker"], popen=fail_popen)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.skipif(os.name != "posix", reason="pass_fds is POSIX-only")
def test_spawn_worker_readiness_failure_closes_descriptors_and_reaps(monkeypatch):
    descriptors = os.pipe()
    proc = _FakeProc(_lines({"not_ready": True}))
    monkeypatch.setattr(os, "pipe", lambda: descriptors)

    with pytest.raises(WorkerError, match="did not report ready"):
        spawn_worker(["/fake/worker"], popen=lambda *args, **kwargs: proc)

    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert proc.events == ["terminate", "wait"]


@pytest.mark.skipif(os.name != "posix", reason="pass_fds is POSIX-only")
def test_spawn_worker_old_ready_closes_writer_and_keeps_old_request_shape(monkeypatch):
    descriptors = os.pipe()
    proc = _FakeProc(_lines({"ready": True}, {"done": True}))
    monkeypatch.setattr(os, "pipe", lambda: descriptors)

    client = spawn_worker(["/fake/worker"], popen=lambda *args, **kwargs: proc)
    assert client.supports_cancel is False
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    with pytest.raises(OSError):
        os.fstat(descriptors[1])

    request_id = client.reserve_request()
    client.generate("hi", _Cfg(), request_id=request_id)
    assert "cancel_request_id" not in json.loads(proc.stdin.written[0])
    assert client.stop() is False
    client.close()


def test_spawn_worker_non_posix_keeps_normal_jsonl_behavior(monkeypatch):
    proc = _FakeProc(
        _lines({"ready": True, "max_named_sessions": 2, "supports_cancel": True})
    )
    captured = {}
    monkeypatch.setattr(worker_client_mod.os, "name", "nt")

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return proc

    client = spawn_worker(["C:\\fake-worker.exe"], env={"BASE": "1"}, popen=fake_popen)
    assert "pass_fds" not in captured
    assert captured["env"] == {"BASE": "1"}
    assert client.supports_cancel is False
    assert client.stop() is False
    client.close()
