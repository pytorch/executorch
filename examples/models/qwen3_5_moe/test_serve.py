# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic tests for the Qwen3.5 MoE OpenAI serving launcher."""

import pathlib
from types import SimpleNamespace

import pytest

from executorch.examples.models.qwen3_5_moe import serve

_HERE = pathlib.Path(serve.__file__).resolve().parent
_REPO_ROOT = serve._repo_root()


def test_generic_runner_pybind_has_no_qwen_include():
    src = (_REPO_ROOT / "extension/llm/runner/pybindings.cpp").read_text()
    assert "qwen3_5_moe" not in src and "qwen35_moe" not in src


def test_generic_server_does_not_reference_qwen():
    server_dir = _REPO_ROOT / "examples/llm_server"
    offenders = []
    for p in server_dir.rglob("*.py"):
        text = p.read_text()
        if "qwen3_5_moe" in text or "_qwen35_moe" in text:
            offenders.append(p)
    assert offenders == [], f"generic server must not reference Qwen: {offenders}"


def test_control_plane_runs_no_model_code():
    serve_src = (_HERE / "serve.py").read_text()
    assert "Qwen35MoEEngine" not in serve_src
    worker_src = (_HERE / "qwen35_moe_worker.cpp").read_text()
    assert "Qwen35MoEEngine" in worker_src


def test_python_worker_and_pybind_are_absent():
    assert not (_HERE / "worker.py").exists()
    assert not (_HERE / "qwen35_moe_pybindings.cpp").exists()


def test_spawn_builds_worker_command(monkeypatch):
    captured = {}

    def fake_spawn(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return object()

    monkeypatch.setattr(serve, "spawn_worker", fake_spawn)
    serve._spawn(
        SimpleNamespace(
            worker_bin="/bin/qwen_worker",
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path="d.ptd",
            max_sessions=4,
            warm_resume=True,
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
        "--max_sessions",
        "4",
        "--warm_resume=true",
    ]


def test_spawn_defaults_worker_bin_and_omits_empty_data_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        serve, "spawn_worker", lambda cmd, env=None: captured.update(cmd=cmd)
    )
    serve._spawn(
        SimpleNamespace(
            worker_bin=None,
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path=None,
            max_sessions=4,
            warm_resume=False,
        )
    )
    cmd = captured["cmd"]
    assert cmd[0].endswith("qwen3_5_moe_worker")
    assert "--data_path" not in cmd
    assert "--warm_resume=false" in cmd


def test_build_app_uses_qwen_tool_parser(monkeypatch):
    captured = {}

    class _FakeTemplate:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeRuntime:
        def close_worker(self):
            pass

    monkeypatch.setattr(serve, "ChatTemplate", _FakeTemplate)
    monkeypatch.setattr(serve, "_spawn", lambda args: object())
    monkeypatch.setattr(serve, "SessionRuntime", lambda worker: _FakeRuntime())

    def fake_serving(*args, **kwargs):
        captured["tool_detector_cls"] = kwargs["tool_detector_cls"]
        return object()

    class _FakeApp:
        def on_event(self, event):
            captured["event"] = event
            return lambda fn: fn

    monkeypatch.setattr(serve, "ServingChat", fake_serving)
    monkeypatch.setattr(serve, "build_app", lambda serving, model_id: _FakeApp())
    serve.build_app_from_args(
        SimpleNamespace(
            hf_tokenizer="hf",
            no_think=True,
            model_id="qwen3.5-moe",
            max_context=10,
        )
    )
    assert captured["tool_detector_cls"] is serve.QwenFunctionCallDetector
    assert captured["event"] == "shutdown"


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
