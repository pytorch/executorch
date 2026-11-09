# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batching launcher wiring; model formatting is shared with the legacy launcher."""

import asyncio
import sys

import pytest

pytest.importorskip("pydantic", reason="requires llm_server serving dependencies")
pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")

from executorch.examples.models.muse_glimmer.serving import (  # noqa: E402
    serve,
    serve_batching,
)
from executorch.examples.models.muse_glimmer.tests.test_serve import (  # noqa: E402
    _StubTemplate,
)


@pytest.fixture
def arguments(tmp_path):
    model = tmp_path / "model.pte"
    tokenizer = tmp_path / "tokenizer.json"
    model.touch()
    tokenizer.write_text("{}")
    return [
        "--worker-bin",
        sys.executable,
        "--model-path",
        str(model),
        "--tokenizer-path",
        str(tokenizer),
        "--hf-tokenizer",
        "template",
        "--max-context",
        "4096",
    ]


@pytest.mark.parametrize("captures", [0, 1, 8])
def test_prefix_capture_override_reaches_worker(arguments, captures):
    args = serve_batching._parse_args(
        arguments
        + ["--prefix-cache-entries", "4", "--prefix-capture-slots", str(captures)]
    )
    assert f"--prefix_capture_slots={captures}" in serve_batching._worker_command(args)


def test_prompt_end_capture_is_opt_in_and_reaches_worker(arguments, capsys):
    default = serve_batching._parse_args(arguments + ["--prefix-cache-entries", "4"])
    assert not default.prefix_checkpoint_prompt_end
    assert not any(
        "prefix_checkpoint_prompt_end" in flag
        for flag in serve_batching._worker_command(default)
    )
    enabled = serve_batching._parse_args(arguments + ["--prefix-checkpoint-prompt-end"])
    assert enabled.prefix_cache_entries == 8
    assert "--prefix_checkpoint_prompt_end=true" in serve_batching._worker_command(
        enabled
    )
    assert "--prefix_cache_entries=8" in serve_batching._worker_command(enabled)
    with pytest.raises(SystemExit):
        serve_batching._parse_args(
            arguments
            + ["--prefix-checkpoint-prompt-end", "--prefix-cache-entries", "0"]
        )
    with pytest.raises(SystemExit):
        serve_batching._parse_args(
            arguments
            + [
                "--prefix-checkpoint-prompt-end",
                "--prefix-checkpoint-interval",
                "64",
            ]
        )
    assert "cannot be combined" in capsys.readouterr().err


def test_session_checkpoint_capacity_override_and_row_overflow(arguments):
    args = serve_batching._parse_args(
        arguments + ["--prefix-checkpoint-prompt-end", "--prefix-cache-entries", "2"]
    )
    assert "--prefix_cache_entries=2" in serve_batching._worker_command(args)
    for capacity in (0, -1, (1 << 31) - 1):
        with pytest.raises(SystemExit):
            serve_batching._parse_args(
                arguments
                + [
                    "--prefix-checkpoint-prompt-end",
                    "--prefix-cache-entries",
                    str(capacity),
                ]
            )


def test_worker_forwards_mg_options(arguments):
    args = serve_batching._parse_args(
        arguments
        + [
            "--max-inflight-requests",
            "17",
            "--max-vision-patches",
            "2048",
            "--prefix-cache-entries",
            "3",
            "--prefix-checkpoint-interval",
            "512",
        ]
    )
    command = serve_batching._worker_command(args)
    flags = dict(arg.split("=", 1) for arg in command[1:])
    assert flags["--pte"] == args.model_path
    assert flags["--tokenizer"] == args.tokenizer_path
    assert flags["--backend"] == "mlx"
    assert flags["--max_session_tokens"] == "4096"
    assert flags["--max_inflight_requests"] == "17"
    assert flags["--max_vision_patches"] == "2048"
    assert flags["--max_image_bytes"] == str(20 * 1024 * 1024)
    assert flags["--max_input_frame_bytes"] == str(32 * 1024 * 1024)
    assert flags["--bos_id"] == "200000"
    assert flags["--eos_id"] == "200001"
    assert flags["--prefix_checkpoint_interval"] == "512"


@pytest.mark.parametrize("options,enabled", [([], "false"), (["--dflash"], "true")])
def test_dflash_is_opt_in(arguments, options, enabled):
    args = serve_batching._parse_args(arguments + options)
    command = serve_batching._worker_command(args)
    flags = dict(arg.split("=", 1) for arg in command[1:])
    assert flags["--dflash"] == enabled
    assert "--draft_argmax" not in flags
    assert flags["--dflash_n_draft"] == "0"


@pytest.mark.parametrize("n_draft", [0, 1, 2, 3])
def test_draft_count_forwarded(arguments, n_draft):
    args = serve_batching._parse_args(
        arguments + ["--dflash", "--dflash-n-draft", str(n_draft)]
    )
    command = serve_batching._worker_command(args)
    flags = dict(arg.split("=", 1) for arg in command[1:])
    assert flags["--dflash_n_draft"] == str(n_draft)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--backend", "vulkan"),
        ("--dflash-n-draft", "-1"),
        ("--dflash-n-draft", str((1 << 31) - 1)),
        ("--dflash-n-draft", "2"),
        ("--max-vision-patches", "3"),
        ("--max-vision-patches", str(1 << 31)),
        ("--prefix-checkpoint-interval", "8"),
        ("--prefix-checkpoint-interval", "-1"),
        ("--prefix-capture-slots", "-1"),
        ("--prefix-capture-slots", str(1 << 31)),
    ],
)
def test_launcher_rejects_invalid_options(arguments, flag, value):
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(arguments + [flag, value])
    assert error.value.code == 2


def test_cuda_worker_takes_the_weights_file(arguments, tmp_path):
    data = tmp_path / "aoti_cuda_blob.ptd"
    data.touch()
    args = serve_batching._parse_args(
        arguments + ["--backend", "cuda", "--data-path", str(data)]
    )
    command = serve_batching._worker_command(args)
    flags = dict(arg.split("=", 1) for arg in command[1:])
    assert flags["--backend"] == "cuda"
    assert flags["--data_path"] == str(data)
    assert "--pos_embed_path" not in flags


def test_cuda_launcher_forwards_dflash(arguments, tmp_path):
    data = tmp_path / "aoti_cuda_blob.ptd"
    data.touch()
    args = serve_batching._parse_args(
        arguments
        + ["--backend", "cuda", "--data-path", str(data), "--dflash"]
        + ["--dflash-n-draft", "15"]
    )
    flags = dict(arg.split("=", 1) for arg in serve_batching._worker_command(args)[1:])
    assert (flags["--backend"], flags["--dflash"], flags["--dflash_n_draft"]) == (
        "cuda",
        "true",
        "15",
    )


def test_cuda_launcher_requires_the_weights_file(arguments):
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(arguments + ["--backend", "cuda"])
    assert error.value.code == 2


def test_cuda_worker_takes_the_vision_table(arguments, tmp_path):
    data = tmp_path / "aoti_cuda_blob.ptd"
    table = tmp_path / "pos_embed.bin"
    data.touch()
    table.touch()
    args = serve_batching._parse_args(
        arguments
        + ["--backend", "cuda", "--data-path", str(data)]
        + ["--pos-embed-path", str(table)]
    )
    flags = dict(arg.split("=", 1) for arg in serve_batching._worker_command(args)[1:])
    assert flags["--pos_embed_path"] == str(table)


@pytest.mark.parametrize("shortfall", [0, 1])
def test_image_frame_capacity(arguments, shortfall):
    # Four image bytes need eight base64 bytes plus 1024 bytes of JSON framing.
    request_bytes = 1032 - shortfall
    argv = arguments + [
        "--max-image-bytes",
        "4",
        "--max-request-bytes",
        str(request_bytes),
    ]
    if shortfall:
        with pytest.raises(SystemExit) as error:
            serve_batching._parse_args(argv)
        assert error.value.code == 2
    else:
        args = serve_batching._parse_args(argv)
        assert args.max_request_bytes == request_bytes


@pytest.mark.parametrize("constructor_failure", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_chat_factory_closes_worker(
    arguments, monkeypatch, constructor_failure, bounded
):
    arguments = arguments + (
        ["--api-dialect", "openrouter", "--streaming-preflight-timeout", "0.25"]
        if bounded
        else []
    )
    captured = {}

    class Worker:
        closed = False

        async def close(self):
            self.closed = True

    worker = Worker()

    class Runtime:
        def __init__(self, instance):
            assert instance is worker
            if constructor_failure:
                raise RuntimeError("construction failed")

        async def aclose_worker(self):
            await worker.close()

        @staticmethod
        async def _finish_cleanup(task):
            await task

    async def spawn(command, **kwargs):
        captured["command"] = command
        return worker

    monkeypatch.setattr(
        serve_batching, "ChatTemplate", lambda *args, **kwargs: _StubTemplate()
    )
    monkeypatch.setattr(serve_batching, "SessionRuntime", Runtime)
    monkeypatch.setattr(serve_batching, "spawn_multiplexed_worker", spawn)
    monkeypatch.setattr(
        serve_batching,
        "build_app",
        lambda _, model_id, *, serving_factory, **kwargs: serving_factory,
    )
    args = serve_batching._parse_args(arguments)
    factory = serve_batching.build_app_from_args(args)
    assert "command" not in captured

    async def scenario():
        async with factory() as chat:
            assert isinstance(chat, serve.MuseGlimmerServingChat)
            assert chat.api_dialect == ("openrouter" if bounded else "openai")
            assert chat._streaming_preflight_timeout == (0.25 if bounded else None)
            assert (
                chat._reasoning_token_counter.__func__
                is _StubTemplate.count_reasoning_tokens
            )
            assert chat._prompt_token_offset == 1
            assert chat._max_image_bytes == args.max_image_bytes
            assert not worker.closed

    if constructor_failure:
        with pytest.raises(RuntimeError, match="construction failed"):
            asyncio.run(scenario())
    else:
        asyncio.run(scenario())
    assert worker.closed
    assert captured["command"] == serve_batching._worker_command(args)


@pytest.mark.parametrize("value,expected", [("none", None), ("0", 0.0), ("0.25", 0.25)])
def test_api_dialect_and_preflight_options(arguments, value, expected):
    args = serve_batching._parse_args(
        arguments
        + ["--api-dialect", "openrouter", "--streaming-preflight-timeout", value]
    )
    assert args.api_dialect == "openrouter"
    assert args.streaming_preflight_timeout == expected


@pytest.mark.parametrize("value", ["-1", "inf", "nan", "invalid"])
def test_invalid_preflight_option(arguments, value):
    with pytest.raises(SystemExit):
        serve_batching._parse_args(arguments + ["--streaming-preflight-timeout", value])
