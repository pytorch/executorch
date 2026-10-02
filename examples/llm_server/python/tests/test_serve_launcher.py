# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import uvicorn

from executorch.examples.llm_server.python import serve
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tool_parsers import HermesDetector
from executorch.examples.llm_server.python.worker_client import WorkerStats
from fastapi.testclient import TestClient


@pytest.fixture
def argv(tmp_path):
    model = tmp_path / "model.pte"
    model.write_bytes(b"model fixture")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    return [
        "--worker-bin",
        sys.executable,
        "--model-path",
        str(model),
        "--tokenizer-path",
        str(tokenizer),
        "--max-context",
        "1024",
    ]


@pytest.fixture
def launch(monkeypatch):
    template = Mock(spec=["generation_preamble"])
    worker = Mock(spec=["close"])
    runtime = Mock(spec=["close_worker"])
    stages = {
        "template": Mock(return_value=template),
        "preamble": template.generation_preamble,
        "spawn": Mock(return_value=worker),
        "runtime": Mock(return_value=runtime),
        "serving": Mock(),
        "app": Mock(),
        "uvicorn": Mock(),
    }
    monkeypatch.setattr(serve, "ChatTemplate", stages["template"])
    monkeypatch.setattr(serve, "spawn_worker", stages["spawn"])
    monkeypatch.setattr(serve, "SessionRuntime", stages["runtime"])
    monkeypatch.setattr(serve, "ServingChat", stages["serving"])
    monkeypatch.setattr(serve, "build_app", stages["app"])
    monkeypatch.setattr(uvicorn, "run", stages["uvicorn"])
    return SimpleNamespace(
        stages=stages, template=template, worker=worker, runtime=runtime
    )


def test_default_launch_requires_multiplexing_and_shares_context(argv, launch):
    args = serve._parse_args(argv)
    serve.main(argv)
    launch.stages["spawn"].assert_called_once_with(
        [
            sys.executable,
            "--pte",
            args.model_path,
            "--tokenizer",
            args.tokenizer_path,
            "--max_sessions",
            "16",
            "--max_session_tokens",
            "1024",
            "--max_decode_sequences",
            "8",
            "--max_inflight_requests",
            "64",
            "--prefix_cache_entries",
            "0",
        ],
        require_multiplexing=True,
    )
    launch.stages["template"].assert_called_once_with(
        hf_tokenizer_path=args.tokenizer_path,
        assistant_header="<|im_start|>assistant\n",
    )
    launch.stages["serving"].assert_called_once_with(
        launch.runtime,
        launch.template,
        "executorch",
        max_context=1024,
        tool_detector_cls=HermesDetector,
    )
    launch.stages["uvicorn"].assert_called_once_with(
        launch.stages["app"].return_value, host="127.0.0.1", port=8000
    )
    launch.runtime.close_worker.assert_called_once_with()


_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]
_TOOL_TOKENS = [
    "<tool_",
    'call>\n{"name": "get_weather", "arguments": {"city": "Pa',
    'ris"}}\n</tool',
    "_call>",
]


@pytest.fixture
def launcher_request(argv, monkeypatch):
    def post(tokens, **options):
        template = Mock(spec=serve.ChatTemplate)
        template.generation_preamble.return_value = ""
        template.turn_stop_sequences.return_value = []
        template.special_tokens.return_value = []
        template.count_tokens.return_value = 5
        template.render.return_value = "PROMPT"

        def generate(prompt, config, token_callback, stats_callback, request_id):
            assert prompt == "PROMPT"
            assert request_id == 1
            for token in tokens:
                token_callback(token)
            stats_callback(
                WorkerStats(
                    num_prompt_tokens=5,
                    num_generated_tokens=len(tokens),
                    finish_reason="stop",
                )
            )

        worker = SimpleNamespace(
            supports_multiplexing=True,
            max_inflight_requests=4,
            reserve_request=Mock(return_value=1),
            wait_for_request=Mock(),
            generate=Mock(side_effect=generate),
            close=Mock(),
        )
        responses = []

        def run(app, **kwargs):
            # Exercise the app while main still owns the live runtime.
            with TestClient(app) as client:
                responses.append(
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "executorch",
                            "messages": [{"role": "user", "content": "weather?"}],
                            **options,
                        },
                    )
                )

        monkeypatch.setattr(serve, "ChatTemplate", Mock(return_value=template))
        monkeypatch.setattr(serve, "spawn_worker", Mock(return_value=worker))
        monkeypatch.setattr(uvicorn, "run", run)
        serve.main(argv)
        worker.generate.assert_called_once()
        worker.wait_for_request.assert_called_once_with(1)
        worker.close.assert_called_once_with()
        return responses[0], template

    return post


def _launcher_choices(response, stream):
    assert response.status_code == 200
    if not stream:
        return response.json()["choices"]
    assert response.headers["content-type"].startswith("text/event-stream")
    events = response.text.strip().split("\n\n")
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    return [chunk["choices"][0] for chunk in chunks]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("options", [{}, {"tool_choice": "auto"}])
def test_launcher_app_parses_split_tool_calls(launcher_request, stream, options):
    response, template = launcher_request(
        _TOOL_TOKENS, tools=_TOOLS, stream=stream, **options
    )
    choices = _launcher_choices(response, stream)
    assert choices[-1]["finish_reason"] == "tool_calls"
    messages = [choice["delta" if stream else "message"] for choice in choices]
    calls = [call for message in messages for call in message.get("tool_calls", [])]
    assert len(calls) == 1
    assert calls[0]["id"].startswith("call-")
    assert calls[0]["index"] == 0
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert not any(message.get("content") for message in messages)
    assert template.render.call_args.kwargs["tools"] == _TOOLS


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "tokens,options",
    [
        (["Hello", ", world!"], {}),
        (["Hello", ", world!"], {"tools": _TOOLS}),
        (_TOOL_TOKENS, {"tools": _TOOLS, "tool_choice": "none"}),
    ],
    ids=["plain", "plain-with-tools", "tools-disabled"],
)
def test_launcher_app_preserves_text(launcher_request, stream, tokens, options):
    response, template = launcher_request(tokens, stream=stream, **options)
    choices = _launcher_choices(response, stream)
    assert choices[-1]["finish_reason"] == "stop"
    messages = [choice["delta" if stream else "message"] for choice in choices]
    assert not any(message.get("tool_calls") for message in messages)
    assert "".join(message.get("content", "") for message in messages) == "".join(
        tokens
    )
    expected_tools = (
        None if options.get("tool_choice") == "none" else options.get("tools")
    )
    assert template.render.call_args.kwargs["tools"] == expected_tools
    assert template.generation_preamble.call_args.kwargs["tools"] == expected_tools


def test_overrides_and_capability_based_concurrency(argv, launch, monkeypatch):
    header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    worker = SimpleNamespace(
        supports_multiplexing=True, max_inflight_requests=23, close=Mock()
    )
    launch.stages["spawn"].return_value = worker
    monkeypatch.setattr(serve, "SessionRuntime", SessionRuntime)
    serve.main(
        argv
        + [
            "--max-sessions",
            "32",
            "--max-decode-sequences",
            "16",
            "--max-inflight-requests",
            "23",
            "--prefix-cache-entries",
            "5",
            "--hf-tokenizer",
            "organization/model",
            "--assistant-header",
            header,
            "--model-id",
            "llama1b",
            "--host",
            "localhost",
            "--port",
            "8123",
        ]
    )
    command = launch.stages["spawn"].call_args.args[0]
    assert command[5:] == [
        "--max_sessions",
        "32",
        "--max_session_tokens",
        "1024",
        "--max_decode_sequences",
        "16",
        "--max_inflight_requests",
        "23",
        "--prefix_cache_entries",
        "5",
    ]
    launch.stages["template"].assert_called_once_with(
        hf_tokenizer_path="organization/model", assistant_header=header
    )
    runtime = launch.stages["serving"].call_args.args[0]
    assert runtime.max_concurrent_requests == 23
    assert launch.stages["serving"].call_args.args[2] == "llama1b"
    assert launch.stages["app"].call_args.args[1] == "llama1b"
    assert launch.stages["uvicorn"].call_args.kwargs == {
        "host": "localhost",
        "port": 8123,
    }
    worker.close.assert_called_once_with()


def test_json_file_uses_sibling_config_and_directory(argv, launch):
    directory = serve._parse_args(argv).tokenizer_path
    serve.main(argv + ["--tokenizer-path", directory + "/tokenizer.json"])
    assert launch.stages["spawn"].call_args.args[0][4] == directory
    assert launch.stages["template"].call_args.kwargs["hf_tokenizer_path"] == directory


@pytest.mark.parametrize("filename", ["tokenizer.model", "tokenizer.json"])
def test_file_without_sibling_config_requires_explicit_template(
    argv, tmp_path, filename
):
    tokenizer = tmp_path / filename
    tokenizer.write_text("fixture")
    with pytest.raises(SystemExit) as error:
        serve._parse_args(argv + ["--tokenizer-path", str(tokenizer)])
    assert error.value.code == 2
    args = serve._parse_args(
        argv + ["--tokenizer-path", str(tokenizer), "--hf-tokenizer", "org/model"]
    )
    assert args.tokenizer_path == str(tokenizer)
    assert args.hf_tokenizer == "org/model"


@pytest.mark.parametrize(
    "flag",
    [
        "--max-context",
        "--max-sessions",
        "--max-decode-sequences",
        "--max-inflight-requests",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", str(1 << 31)])
def test_invalid_positive_capacities_fail_before_template_or_spawn(
    argv, launch, flag, value
):
    with pytest.raises(SystemExit) as error:
        serve.main(argv + [flag, value])
    assert error.value.code == 2
    launch.stages["template"].assert_not_called()
    launch.stages["spawn"].assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        ["--prefix-cache-entries", "-1"],
        ["--prefix-cache-entries", str(1 << 31)],
        ["--max-sessions", str((1 << 31) - 1), "--prefix-cache-entries", "1"],
        ["--max-sessions", "1", "--prefix-cache-entries", str((1 << 31) - 2)],
        ["--port", "0"],
        ["--port", "65536"],
    ],
)
def test_invalid_cache_headroom_and_ports(argv, launch, options):
    with pytest.raises(SystemExit):
        serve.main(argv + options)
    launch.stages["spawn"].assert_not_called()


@pytest.mark.parametrize("port", [1, 65535])
def test_numeric_limit_boundaries_are_inclusive(argv, port):
    max_int32 = (1 << 31) - 1
    args = serve._parse_args(
        argv
        + [
            "--max-context",
            str(max_int32),
            "--max-sessions",
            str(max_int32 - 2),
            "--max-decode-sequences",
            str(max_int32),
            "--max-inflight-requests",
            str(max_int32),
            "--prefix-cache-entries",
            "1",
            "--port",
            str(port),
        ]
    )
    assert args.max_context == max_int32
    assert args.max_sessions + args.prefix_cache_entries + 1 == max_int32
    assert args.max_decode_sequences == args.max_inflight_requests == max_int32
    assert args.port == port


def test_numeric_errors_precede_path_checks(argv, launch, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        serve.main(
            argv + ["--model-path", str(tmp_path / "missing"), "--max-context", "0"]
        )
    assert error.value.code == 2
    assert (
        capsys.readouterr()
        .err.rstrip()
        .endswith("error: --max-context must be between 1 and 2147483647")
    )
    launch.stages["template"].assert_not_called()
    launch.stages["spawn"].assert_not_called()


def test_zero_cache_needs_no_extra_physical_row(argv):
    args = serve._parse_args(argv + ["--max-sessions", str((1 << 31) - 1)])
    assert args.max_sessions == (1 << 31) - 1


@pytest.mark.parametrize("flag", ["--worker-bin", "--model-path", "--tokenizer-path"])
def test_missing_paths_fail_before_spawn(argv, launch, tmp_path, flag):
    with pytest.raises(SystemExit):
        serve.main(argv + [flag, str(tmp_path / "missing")])
    launch.stages["spawn"].assert_not_called()


@pytest.mark.parametrize("command", ["./worker", "worker"])
def test_relative_and_path_worker_commands(argv, tmp_path, monkeypatch, command):
    executable = tmp_path / "worker"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    args = serve._parse_args(argv + ["--worker-bin", command])
    assert args.worker_bin == command


def test_empty_tokenizer_directory_is_rejected(argv, tmp_path):
    with pytest.raises(SystemExit):
        serve._parse_args(argv + ["--tokenizer-path", str(tmp_path)])


def test_context_is_required(argv, launch):
    with pytest.raises(SystemExit):
        serve.main(argv[:-2])
    launch.stages["spawn"].assert_not_called()


def test_template_probe_precedes_spawn(argv, launch):
    def spawn(*args, **kwargs):
        launch.template.generation_preamble.assert_called_once_with()
        return launch.worker

    launch.stages["spawn"].side_effect = spawn
    serve.main(argv)


@pytest.mark.parametrize(
    "stage", ["template", "preamble", "spawn", "runtime", "serving", "app", "uvicorn"]
)
@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit])
def test_startup_failures_release_owned_worker(argv, launch, stage, error_type):
    launch.stages[stage].side_effect = error_type("startup failed")
    with pytest.raises(error_type):
        serve.main(argv)
    if stage in ("template", "preamble", "spawn"):
        launch.worker.close.assert_not_called()
        launch.runtime.close_worker.assert_not_called()
    elif stage == "runtime":
        launch.worker.close.assert_called_once_with()
        launch.runtime.close_worker.assert_not_called()
    else:
        launch.runtime.close_worker.assert_called_once_with()
