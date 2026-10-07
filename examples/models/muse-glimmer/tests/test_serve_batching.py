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


def test_worker_forwards_mg_options(arguments):
    args = serve_batching._parse_args(arguments + ["--max-inflight-requests", "17"])
    command = serve_batching._worker_command(args)
    flags = dict(zip(command[1::2], command[2::2]))
    assert flags["--pte"] == args.model_path
    assert flags["--tokenizer"] == args.tokenizer_path
    assert flags["--backend"] == "mlx"
    assert flags["--max_session_tokens"] == "4096"
    assert flags["--max_inflight_requests"] == "17"
    assert flags["--max_image_bytes"] == str(20 * 1024 * 1024)
    assert flags["--max_input_frame_bytes"] == str(32 * 1024 * 1024)
    assert flags["--bos_id"] == "200000"
    assert flags["--eos_id"] == "200001"


def test_launcher_requires_mlx(arguments):
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(arguments + ["--backend", "cuda"])
    assert error.value.code == 2


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
def test_chat_factory_closes_worker(arguments, monkeypatch, constructor_failure):
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
        lambda _, model_id, *, serving_factory: serving_factory,
    )
    args = serve_batching._parse_args(arguments)
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
