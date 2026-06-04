# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Qwen3.5 MoE process-isolated OpenAI launcher (serve.py).

Hermetic: no model, GPU, or worker subprocess. Covers layering (Qwen stays an
example; the control plane runs no CUDA), the single-slot CLI guard, and the
WorkerRunner JSONL protocol against a fake worker. The live HTTP smoke test is
documented in README.md and run on a CUDA box.
"""

import json
import pathlib

import pytest

from executorch.examples.models.qwen3_5_moe import serve

_HERE = pathlib.Path(serve.__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]  # qwen3_5_moe -> models -> examples -> repo root


# --- Layering ---------------------------------------------------------------


def test_generic_runner_pybind_has_no_qwen_include():
    src = (_REPO_ROOT / "extension/llm/runner/pybindings.cpp").read_text()
    assert "qwen3_5_moe" not in src and "qwen35_moe" not in src


def test_generic_server_does_not_import_qwen():
    server_dir = _REPO_ROOT / "extension/llm/server"
    offenders = [
        p
        for p in server_dir.rglob("*.py")
        if "qwen3_5_moe" in p.read_text() or "_qwen35_moe" in p.read_text()
    ]
    assert offenders == [], f"generic server must not reference Qwen: {offenders}"


def test_control_plane_runs_no_cuda_model():
    # serve.py is the control plane: it must NOT construct the CUDA engine; only
    # the worker (worker.py) calls create_engine on the model module.
    assert "create_engine" not in (_HERE / "serve.py").read_text()
    assert "create_engine" in (_HERE / "worker.py").read_text()


# --- WorkerRunner JSONL protocol (fake worker) ------------------------------


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _FakeProc:
    def __init__(self, lines):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(lines)
        self.returncode = None

    def poll(self):
        return None


class _Cfg:
    __slots__ = ("max_new_tokens", "temperature")

    def __init__(self, max_new_tokens=16, temperature=0.0):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature


def test_worker_runner_streams_tokens_and_stats():
    proc = _FakeProc(
        [
            '{"token": "Hello"}\n',
            '{"token": " world"}\n',
            '{"done": true, "prompt_tokens": 5, "completion_tokens": 2}\n',
        ]
    )
    wr = serve.WorkerRunner(proc)
    out, stats = [], {}
    wr.generate(
        "p",
        _Cfg(temperature=0.7),
        token_callback=out.append,
        stats_callback=lambda s: stats.update(
            p=s.num_prompt_tokens, g=s.num_generated_tokens
        ),
    )
    assert out == ["Hello", " world"]
    assert stats == {"p": 5, "g": 2}
    sent = json.loads(proc.stdin.written[0])
    assert sent["prompt"] == "p" and sent["temperature"] == 0.7


def test_worker_runner_error_raises():
    proc = _FakeProc(['{"error": "boom"}\n'])
    with pytest.raises(RuntimeError, match="boom"):
        serve.WorkerRunner(proc).generate("p", _Cfg(), token_callback=lambda t: None)


def test_worker_runner_exit_midrequest_raises():
    proc = _FakeProc([])  # readline() -> "" means the worker exited
    with pytest.raises(RuntimeError, match="exited"):
        serve.WorkerRunner(proc).generate("p", _Cfg())


# --- CLI guard --------------------------------------------------------------


def test_rejects_multiple_runners(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve.py",
            "--model-path",
            "m.pte",
            "--tokenizer-path",
            "t.json",
            "--hf-tokenizer",
            "hf",
            "--num-runners",
            "2",
        ],
    )
    with pytest.raises(SystemExit):
        serve.main()
