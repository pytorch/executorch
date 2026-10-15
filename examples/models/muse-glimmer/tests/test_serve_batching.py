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
    _FakeRuntime,
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
    flags = dict(zip(command[1::2], command[2::2]))
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
    assert not args.dflash
    assert args.dflash_n_draft == 0
    assert flags["--n_draft"] == "0"
    assert command[-1] == "--dflash=false"
    assert "--dflash" not in command


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--backend", "vulkan"),
        ("--max-vision-patches", "3"),
        ("--max-vision-patches", str(1 << 31)),
        ("--prefix-checkpoint-interval", "8"),
        ("--prefix-checkpoint-interval", "-1"),
    ],
)
def test_launcher_rejects_invalid_options(arguments, flag, value):
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(arguments + [flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize("n_draft", [None, 0, 1, 7, (1 << 31) - 1])
def test_cuda_worker_takes_the_weights_file(arguments, tmp_path, n_draft):
    data = tmp_path / "aoti_cuda_blob.ptd"
    data.touch()
    argv = arguments + ["--backend", "cuda", "--data-path", str(data)]
    if n_draft is not None:
        argv += ["--dflash"]
        if n_draft:
            argv += ["--dflash-n-draft", str(n_draft)]
    args = serve_batching._parse_args(argv)
    command = serve_batching._worker_command(args)
    flags = dict(zip(command[1::2], command[2::2]))
    assert flags["--backend"] == "cuda"
    assert flags["--data_path"] == str(data)
    assert flags["--max_session_tokens"] == "4096"
    assert flags["--max_sessions"] == "4"
    assert flags["--max_decode_sequences"] == "4"
    assert flags["--max_inflight_requests"] == "4"
    assert flags["--prefix_cache_entries"] == "0"
    assert flags["--n_draft"] == str(n_draft or 0)
    assert args.dflash is (n_draft is not None)
    assert command[-1] == ("--dflash=false" if n_draft is None else "--dflash=true")
    assert "--dflash" not in command
    assert "--dflash-n-draft" not in command
    assert "--pos_embed_path" not in flags


@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize("pos_embed", [False, True])
def test_cuda_launcher_rejects_missing_weights_or_vision(
    arguments, tmp_path, pos_embed, dflash
):
    data = tmp_path / "aoti_cuda_blob.ptd"
    table = tmp_path / "pos_embed.bin"
    data.touch()
    table.touch()
    argv = arguments + ["--backend", "cuda"]
    if dflash:
        argv += ["--dflash"]
    argv += (
        ["--data-path", str(data), "--pos-embed-path", str(table)] if pos_embed else []
    )
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize(
    "options,message",
    [
        (["--dflash"], "--dflash supports only --backend cuda"),
        (
            ["--backend", "mlx", "--dflash", "--dflash-n-draft", "4"],
            "the MLX reference adapter is not ported here",
        ),
        (["--dflash-n-draft", "4"], "--dflash-n-draft requires --dflash"),
        (
            ["--backend", "cuda", "--dflash-n-draft", "4"],
            "--dflash-n-draft requires --dflash",
        ),
        (["--dflash-n-draft", "-1"], "--dflash-n-draft must be between"),
        (
            ["--backend", "cuda", "--dflash", "--dflash-n-draft", "-1"],
            "--dflash-n-draft must be between",
        ),
        (
            ["--backend", "cuda", "--dflash", "--dflash-n-draft", str(1 << 31)],
            "--dflash-n-draft must be between",
        ),
        (["--dflash-n-draft", "not-an-int"], "invalid int value"),
    ],
)
def test_launcher_rejects_invalid_dflash_options(arguments, capsys, options, message):
    with pytest.raises(SystemExit) as error:
        serve_batching._parse_args(arguments + options)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_zero_draft_count_without_dflash_is_allowed(arguments):
    args = serve_batching._parse_args(arguments + ["--dflash-n-draft", "0"])
    command = serve_batching._worker_command(args)
    flags = dict(zip(command[1::2], command[2::2]))
    assert flags["--backend"] == "mlx"
    assert flags["--n_draft"] == "0"
    assert command[-1] == "--dflash=false"


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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("temperature", [0.01, 0.5, 2.0])
def test_dflash_http_rejects_sampling_before_worker_admission(stream, temperature):
    from fastapi.testclient import TestClient

    runtime = _FakeRuntime([])
    chat = serve_batching._DFlashServingChat(runtime, _StubTemplate(), "muse_glimmer")
    app = serve_batching.build_app(chat, "muse_glimmer")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "muse_glimmer",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": temperature,
                "stream": stream,
                "max_tokens": 512,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert "temperature=0" in response.json()["error"]["message"]
    assert runtime.prompts == []


@pytest.mark.parametrize("temperature", [None, 0.0])
@pytest.mark.parametrize("dflash", [False, True])
def test_four_http_stream_requests_preserve_512_token_accounting(temperature, dflash):
    import json

    from fastapi.testclient import TestClient

    runtime = _FakeRuntime(
        [
            {"text": "answer", "completion_tokens": 512, "finish_reason": "length"}
            for _ in range(4)
        ]
    )
    chat_type = (
        serve_batching._DFlashServingChat if dflash else serve.MuseGlimmerServingChat
    )
    chat = chat_type(runtime, _StubTemplate(), "muse_glimmer")
    app = serve_batching.build_app(chat, "muse_glimmer")
    with TestClient(app) as client:
        for _ in range(4):
            request = {
                "model": "muse_glimmer",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 512,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if temperature is not None:
                request["temperature"] = temperature
            response = client.post("/v1/chat/completions", json=request)
            assert response.status_code == 200
            events = [
                line[6:]
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert events[-1] == "[DONE]"
            chunks = [json.loads(event) for event in events[:-1]]
            usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
            assert usage["completion_tokens"] == 512
            assert any(
                choice.get("finish_reason") == "length"
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
    assert len(runtime.prompts) == 4
