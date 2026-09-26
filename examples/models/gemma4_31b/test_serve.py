# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic tests for the Gemma 4 31B OpenAI serving launcher."""

import pathlib
from types import SimpleNamespace

import pytest

from executorch.examples.models.gemma4_31b import serve

_HERE = pathlib.Path(serve.__file__).resolve().parent
_REPO_ROOT = serve._repo_root()


def test_generic_server_does_not_reference_gemma4_31b():
    server_dir = _REPO_ROOT / "examples/llm_server"
    offenders = []
    for p in server_dir.rglob("*.py"):
        text = p.read_text()
        if "gemma4_31b" in text or "Gemma4_31B" in text:
            offenders.append(p)
    assert offenders == [], f"generic server must not reference Gemma: {offenders}"


def test_control_plane_runs_no_model_code():
    serve_src = (_HERE / "serve.py").read_text()
    assert "Gemma4_31BEngine" not in serve_src
    worker_src = (_HERE / "gemma4_31b_worker.cpp").read_text()
    assert "Gemma4_31BEngine" in worker_src


def test_python_worker_and_pybind_are_absent():
    assert not (_HERE / "worker.py").exists()
    assert not (_HERE / "gemma4_31b_pybindings.cpp").exists()


def test_spawn_builds_worker_command(monkeypatch):
    captured = {}

    def fake_spawn(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return object()

    monkeypatch.setattr(serve, "spawn_worker", fake_spawn)
    serve._spawn(
        SimpleNamespace(
            worker_bin="/bin/gemma_worker",
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path="d.ptd",
            max_sessions=4,
            warm_resume=True,
            bos_id=2,
            eos_id=1,
        )
    )
    assert captured["cmd"] == [
        "/bin/gemma_worker",
        "--model_path",
        "m.pte",
        "--tokenizer_path",
        "t.json",
        "--max_sessions",
        "4",
        "--warm_resume=true",
        "--bos_id",
        "2",
        "--eos_id",
        "1",
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
            worker_bin=None,
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path=None,
            max_sessions=1,
            warm_resume=False,
            bos_id=2,
            eos_id=1,
        )
    )
    cmd = captured["cmd"]
    assert cmd[0].endswith("gemma4_31b_worker")
    assert "--data_path" not in cmd
    assert "--warm_resume=false" in cmd


def test_build_app_uses_gemma_options(monkeypatch):
    captured = {}

    class _FakeTemplate:
        def __init__(self, *args, **kwargs):
            captured["template_kwargs"] = kwargs

    class _FakeRuntime:
        def close_worker(self):
            pass

    class _FakeApp:
        def on_event(self, event):
            captured["event"] = event
            return lambda fn: fn

    monkeypatch.setattr(serve, "ChatTemplate", _FakeTemplate)
    monkeypatch.setattr(serve, "_spawn", lambda args: object())
    monkeypatch.setattr(serve, "SessionRuntime", lambda worker: _FakeRuntime())

    def fake_serving(*args, **kwargs):
        captured["serving_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(serve, "ServingChat", fake_serving)
    monkeypatch.setattr(serve, "build_app", lambda serving, model_id: _FakeApp())
    serve.build_app_from_args(
        SimpleNamespace(
            hf_tokenizer="hf",
            model_id="gemma4_31b",
            max_context=10,
            tool_parser="gemma",
        )
    )

    assert captured["template_kwargs"] == {
        "assistant_header": "<|turn>model\n",
        "strip_rendered_bos": True,
        "append_generation_prompt_after_tool_response": True,
    }
    assert (
        captured["serving_kwargs"]["tool_detector_cls"] is serve.GemmaToolCallDetector
    )
    assert captured["serving_kwargs"]["prompt_token_offset"] == 1
    assert captured["serving_kwargs"]["content_filter"] is serve._strip_gemma_channels
    assert captured["event"] == "shutdown"


def test_strip_gemma_channels_returns_visible_answer():
    text = "<|channel>thought\nscratch work\n<channel|>The answer."
    assert serve._strip_gemma_channels(text) == "The answer."


def test_strip_gemma_channels_cuts_unclosed_channel():
    assert serve._strip_gemma_channels("Lead <|channel>thought") == "Lead"


def test_strip_gemma_channels_removes_stray_close():
    assert serve._strip_gemma_channels("Visible<channel|>") == "Visible"


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
