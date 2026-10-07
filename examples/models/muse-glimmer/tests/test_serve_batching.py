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


def test_batching_worker_command_preserves_image_limits_and_solo_scope(arguments):
    args = serve_batching._parse_args(arguments)
    command = serve_batching._worker_command(args)
    flags = dict(zip(command[1::2], command[2::2]))
    assert flags["--pte"] == args.model_path
    assert flags["--tokenizer"] == args.tokenizer_path
    assert flags["--backend"] == "mlx"
    assert flags["--max_session_tokens"] == "4096"
    assert flags["--max_inflight_requests"] == "4"
    assert flags["--max_image_bytes"] == str(20 * 1024 * 1024)
    assert flags["--max_input_frame_bytes"] == str(32 * 1024 * 1024)
    assert flags["--bos_id"] == "200000"
    assert flags["--eos_id"] == "200001"
    assert "--data_path" not in flags
    assert "--pos_embed_path" not in flags
    assert not any("dflash" in flag or "artifact_mode" in flag for flag in flags)


@pytest.mark.parametrize(
    "extra",
    [
        ["--backend", "cuda"],
        ["--backend", "dflash"],
        ["--max-context", "1"],
        ["--max-context", "0"],
        ["--max-context", str(1 << 31)],
        ["--max-image-bytes", "0"],
        ["--max-image-bytes", str(20 * 1024 * 1024 + 1)],
        ["--max-request-bytes", "0"],
        ["--max-request-bytes", str(32 * 1024 * 1024 + 1)],
        ["--max-request-bytes", "1048576"],
        ["--max-inflight-requests", "0"],
        ["--max-inflight-requests", str(1 << 31)],
        ["--bos-id", "-1"],
        ["--bos-id", str(1 << 64)],
        ["--eos-id", "-1"],
        ["--eos-id", str(1 << 64)],
        ["--artifact-mode", "dflash"],
    ],
)
def test_batching_launcher_rejects_invalid_limits_and_legacy_modes(arguments, extra):
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(arguments + extra)
    assert error.value.code == 2


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--backend", "mlx"),
        ("--max-context", 2),
        ("--max-context", (1 << 31) - 1),
        ("--max-inflight-requests", 1),
        ("--max-inflight-requests", 17),
        ("--max-inflight-requests", (1 << 31) - 1),
        ("--max-image-bytes", 1),
        ("--max-image-bytes", 20 * 1024 * 1024),
        ("--max-request-bytes", 32 * 1024 * 1024),
        ("--bos-id", 0),
        ("--bos-id", (1 << 64) - 1),
        ("--eos-id", 0),
        ("--eos-id", (1 << 64) - 1),
    ],
)
def test_batching_launcher_accepts_limit_boundaries(arguments, flag, value):
    args = serve_batching._parse_args(arguments + [flag, str(value)])
    assert getattr(args, flag[2:].replace("-", "_")) == value
    if flag == "--max-inflight-requests":
        command = serve_batching._worker_command(args)
        assert command[command.index("--max_inflight_requests") + 1] == str(value)


@pytest.mark.parametrize("image_bytes", [1, 2, 3, 4, 20 * 1024 * 1024])
@pytest.mark.parametrize("shortfall", [0, 1])
def test_batching_launcher_requires_base64_and_framing_capacity(
    arguments, image_bytes, shortfall
):
    request_bytes = 4 * ((image_bytes + 2) // 3) + 1024 - shortfall
    argv = arguments + [
        "--max-image-bytes",
        str(image_bytes),
        "--max-request-bytes",
        str(request_bytes),
    ]
    if shortfall:
        with pytest.raises(SystemExit) as error:
            serve_batching._parse_args(argv)
        assert error.value.code == 2
    else:
        args = serve_batching._parse_args(argv)
        command = serve_batching._worker_command(args)
        flags = dict(zip(command[1::2], command[2::2]))
        assert flags["--max_image_bytes"] == str(image_bytes)
        assert flags["--max_input_frame_bytes"] == str(request_bytes)


@pytest.mark.parametrize("constructor_failure", [False, True])
def test_batching_launcher_owns_async_worker_and_reuses_mg_chat(
    arguments, monkeypatch, constructor_failure
):
    captured = {}

    class Template(_StubTemplate):
        def __init__(self, *args, **kwargs):
            captured["template"] = kwargs

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
        captured["spawn"] = kwargs
        return worker

    monkeypatch.setattr(serve_batching, "ChatTemplate", Template)
    monkeypatch.setattr(serve_batching, "SessionRuntime", Runtime)
    monkeypatch.setattr(serve_batching, "spawn_multiplexed_worker", spawn)
    monkeypatch.setattr(
        serve_batching,
        "build_app",
        lambda _, model_id, *, serving_factory: serving_factory,
    )
    args = serve_batching._parse_args(
        arguments + ["--max-image-bytes", "1024", "--max-request-bytes", "4096"]
    )
    factory = serve_batching.build_app_from_args(args)
    assert "command" not in captured

    async def scenario():
        async with factory() as chat:
            assert isinstance(chat, serve.MuseGlimmerServingChat)
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
    assert captured["spawn"] == {
        "max_request_bytes": args.max_request_bytes,
        "max_message_bytes": 1024 * 1024,
    }
    assert captured["template"] == {
        "assistant_header": "<|start|>assistant",
        "strip_rendered_bos": True,
        "append_generation_prompt_after_tool_response": True,
    }
