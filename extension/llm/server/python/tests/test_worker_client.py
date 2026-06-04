# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the generic WorkerClient JSONL protocol (no model/GPU/subprocess).

A fake process stands in for the C++ worker: it records what the client writes
and replays a scripted sequence of JSONL response lines.
"""

import json
from dataclasses import dataclass

import pytest

from executorch.extension.llm.server.python.worker_client import (
    spawn_worker,
    WorkerClient,
    WorkerError,
)


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _FakeProc:
    def __init__(self, stdout_lines, returncode=None):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self._returncode = returncode

    def poll(self):
        return self._returncode

    @property
    def returncode(self):
        return self._returncode


@dataclass
class _Cfg:
    max_new_tokens: int = 64
    temperature: float = 0.0


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
        _Cfg(temperature=0.7),
        token_callback=out.append,
        stats_callback=lambda s: stats.update(
            prompt=s.num_prompt_tokens, gen=s.num_generated_tokens
        ),
    )
    assert "".join(out) == "Hello world"
    assert stats == {"prompt": 4, "gen": 2}
    # The request carried prompt + sampling, one JSON line.
    sent = json.loads(proc.stdin.written[0])
    assert sent == {"prompt": "hi", "max_new_tokens": 64, "temperature": 0.7}


def test_error_message_raises_worker_error():
    proc = _FakeProc(_lines({"error": "boom"}))
    with pytest.raises(WorkerError, match="boom"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_exit_mid_request_raises():
    proc = _FakeProc([])  # readline() -> "" means the worker exited
    with pytest.raises(WorkerError, match="exited mid-request"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_generate_on_dead_worker_raises():
    proc = _FakeProc([], returncode=1)
    with pytest.raises(WorkerError, match="worker exited"):
        WorkerClient(proc).generate("hi", _Cfg())


def test_spawn_worker_waits_for_ready():
    proc = _FakeProc(_lines({"ready": True}))
    client = spawn_worker(
        ["/fake/worker", "--model_path", "m"], popen=lambda *a, **k: proc
    )
    assert isinstance(client, WorkerClient)


def test_spawn_worker_not_ready_raises():
    proc = _FakeProc(_lines({"oops": True}))
    with pytest.raises(WorkerError, match="did not report ready"):
        spawn_worker(["/fake/worker"], popen=lambda *a, **k: proc)


def test_spawn_worker_no_output_raises():
    proc = _FakeProc([])
    with pytest.raises(WorkerError, match="failed to start"):
        spawn_worker(["/fake/worker"], popen=lambda *a, **k: proc)
