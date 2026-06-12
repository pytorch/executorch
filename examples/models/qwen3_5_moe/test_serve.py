# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Qwen3.5 MoE process-isolated OpenAI launcher (serve.py).

Hermetic: no model, GPU, or worker subprocess. Covers layering (Qwen stays an
example; the control plane runs no CUDA and imports no model pybind), the worker
spawn command, and the single-slot CLI guard. The generic JSONL protocol is
covered by extension/llm/server/python/tests/test_worker_client.py; the live
HTTP smoke test is documented in README.md and run on a CUDA box.
"""

import pathlib
from types import SimpleNamespace

import pytest

from executorch.examples.models.qwen3_5_moe import serve

_HERE = pathlib.Path(serve.__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]  # qwen3_5_moe -> models -> examples -> repo root


# --- Layering ---------------------------------------------------------------


def test_generic_runner_pybind_has_no_qwen_include():
    src = (_REPO_ROOT / "extension/llm/runner/pybindings.cpp").read_text()
    assert "qwen3_5_moe" not in src and "qwen35_moe" not in src


def test_generic_server_does_not_reference_qwen():
    server_dir = _REPO_ROOT / "extension/llm/server"
    offenders = [
        p
        for p in server_dir.rglob("*.py")
        if "qwen3_5_moe" in p.read_text() or "_qwen35_moe" in p.read_text()
    ]
    assert offenders == [], f"generic server must not reference Qwen: {offenders}"


def test_control_plane_runs_no_model_code():
    # serve.py is the control plane: it constructs no engine and imports no model
    # pybind. Model execution lives entirely in the C++ worker.
    serve_src = (_HERE / "serve.py").read_text()
    assert "Qwen35MoEEngine" not in serve_src
    assert "_qwen35_moe" not in serve_src
    worker_src = (_HERE / "qwen35_moe_worker.cpp").read_text()
    assert "Qwen35MoEEngine" in worker_src


def test_python_worker_and_pybind_are_gone():
    # The Python worker and the model pybind have been replaced by the C++ worker.
    assert not (_HERE / "worker.py").exists()
    assert not (_HERE / "qwen35_moe_pybindings.cpp").exists()


# --- Worker spawn wiring ----------------------------------------------------


def test_spawn_builds_worker_command(monkeypatch):
    captured = {}

    def fake_spawn(cmd, env=None):
        captured["cmd"] = cmd
        return object()  # stand-in WorkerClient

    monkeypatch.setattr(serve, "spawn_worker", fake_spawn)
    serve._spawn(
        SimpleNamespace(
            worker_bin="/bin/qwen_worker",
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path="d.ptd",
        )
    )
    assert captured["cmd"] == [
        "/bin/qwen_worker",
        "--model_path",
        "m.pte",
        "--tokenizer_path",
        "t.json",
        "--data_path",
        "d.ptd",
    ]


def test_spawn_defaults_worker_bin_and_omits_empty_data_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        serve, "spawn_worker", lambda cmd, env=None: captured.update(cmd=cmd)
    )
    serve._spawn(
        SimpleNamespace(
            worker_bin=None, model_path="m.pte", tokenizer_path="t.json", data_path=None
        )
    )
    cmd = captured["cmd"]
    assert cmd[0].endswith("qwen3_5_moe_worker")  # default binary path
    assert "--data_path" not in cmd  # omitted when no .ptd


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
