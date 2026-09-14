# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic tests for the Muse Glimmer OpenAI serving launcher."""

import asyncio
import base64
import pathlib
from types import SimpleNamespace

import pytest

# The serving stack is built on pydantic, which ships with the llm_server
# extras (examples/llm_server/python/requirements.txt) rather than core
# ExecuTorch, so skip instead of failing collection when it is absent.
pytest.importorskip("pydantic", reason="requires llm_server serving dependencies")

from executorch.examples.llm_server.python import chat_template  # noqa: E402
from executorch.examples.llm_server.python.errors import APIError  # noqa: E402
from executorch.examples.llm_server.python.protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from executorch.examples.llm_server.python.session_runtime import (  # noqa: E402
    PromptInput,
)
from executorch.examples.models.muse_glimmer.serving import serve  # noqa: E402

_HERE = pathlib.Path(serve.__file__).resolve().parent
_LLM_SERVER_PYTHON = pathlib.Path(chat_template.__file__).resolve().parent


def test_generic_server_does_not_reference_muse_glimmer():
    # The generic control plane must stay model-agnostic. Model-specific
    # tool-call parsers legitimately live in the tool_parsers/ package
    # (gemma.py, qwen.py, hermes.py, muse_glimmer.py) with matching tests/ -- the same
    # established pattern as the other models -- so those are NOT part of the
    # generic plumbing and are intentionally excluded here. This asserts the
    # shared plumbing files themselves carry no muse_glimmer coupling.
    plumbing = _LLM_SERVER_PYTHON
    generic_files = [
        plumbing / "server.py",
        plumbing / "serving_chat.py",
        plumbing / "chat_template.py",
        plumbing / "session_runtime.py",
        plumbing / "worker_client.py",
        plumbing / "protocol.py",
        plumbing / "errors.py",
        plumbing / "openai_transcript.py",
    ]
    offenders = []
    for p in generic_files:
        text = p.read_text()
        if "muse_glimmer" in text.lower() or "MuseGlimmerEngine" in text:
            offenders.append(p)
    assert (
        offenders == []
    ), f"generic server must not reference Muse Glimmer: {offenders}"


def test_control_plane_runs_no_model_code():
    serve_src = (_HERE / "serve.py").read_text()
    assert "MuseGlimmerEngine" not in serve_src
    worker_src = (
        _HERE.parent / "runtime" / "runners" / "muse_glimmer_worker.cpp"
    ).read_text()
    assert "MuseGlimmerEngine" in worker_src


def test_python_worker_and_pybind_are_absent():
    assert not (_HERE / "worker.py").exists()
    assert not (_HERE / "muse_glimmer_pybindings.cpp").exists()


def test_spawn_builds_worker_command(monkeypatch):
    captured = {}

    def fake_spawn(cmd, env=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return object()

    monkeypatch.setattr(serve, "spawn_worker", fake_spawn)
    serve._spawn(
        SimpleNamespace(
            worker_bin="/bin/muse_glimmer_worker",
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path="d.ptd",
            pos_embed_path=None,
            max_image_bytes=1024,
            max_sessions=4,
            warm_resume=True,
            bos_id=200000,
            eos_id=200001,
            artifact_mode="auto",
            dflash_block_length=4,
            dflash_n_draft=3,
            dflash_draft_argmax=True,
            cuda_graph=True,
        )
    )
    assert captured["cmd"] == [
        "/bin/muse_glimmer_worker",
        "--model_path",
        "m.pte",
        "--tokenizer_path",
        "t.json",
        "--max_sessions",
        "4",
        "--warm_resume=true",
        "--bos_id",
        "200000",
        "--eos_id",
        "200001",
        "--artifact_mode",
        "auto",
        "--dflash_block_length",
        "4",
        "--dflash_n_draft",
        "3",
        "--dflash_draft_argmax=true",
        "--cuda_graph=true",
        "--data_path",
        "d.ptd",
        "--max_image_bytes",
        "1024",
    ]


def test_spawn_forwards_explicit_dflash_and_vision_options(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        serve, "spawn_worker", lambda cmd, env=None: captured.update(cmd=cmd)
    )
    serve._spawn(
        SimpleNamespace(
            worker_bin="/bin/muse_glimmer_worker",
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path=None,
            pos_embed_path="pos.bin",
            max_image_bytes=2048,
            max_sessions=1,
            warm_resume=True,
            bos_id=200000,
            eos_id=200001,
            artifact_mode="dflash",
            dflash_block_length=4,
            dflash_n_draft=3,
            dflash_draft_argmax=False,
            cuda_graph=False,
        )
    )

    assert captured["cmd"][-12:] == [
        "--artifact_mode",
        "dflash",
        "--dflash_block_length",
        "4",
        "--dflash_n_draft",
        "3",
        "--dflash_draft_argmax=false",
        "--cuda_graph=false",
        "--max_image_bytes",
        "2048",
        "--pos_embed_path",
        "pos.bin",
    ]


def test_spawn_defaults_worker_bin_and_omits_empty_data_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        serve, "spawn_worker", lambda cmd, env=None: captured.update(cmd=cmd)
    )
    monkeypatch.setattr(
        serve, "_default_worker_bin", lambda: "/bin/muse_glimmer_worker"
    )
    serve._spawn(
        SimpleNamespace(
            worker_bin=None,
            model_path="m.pte",
            tokenizer_path="t.json",
            data_path=None,
            pos_embed_path=None,
            max_image_bytes=serve._DEFAULT_MAX_IMAGE_BYTES,
            max_sessions=1,
            warm_resume=False,
            bos_id=200000,
            eos_id=200001,
            artifact_mode="auto",
            dflash_block_length=4,
            dflash_n_draft=3,
            dflash_draft_argmax=True,
            cuda_graph=True,
        )
    )
    cmd = captured["cmd"]
    assert cmd[0].endswith("muse_glimmer_worker")
    assert "--data_path" not in cmd
    assert "--max_image_bytes" in cmd
    assert "--warm_resume=false" in cmd


def test_build_app_uses_muse_glimmer_options(monkeypatch):
    assert issubclass(serve.MuseGlimmerServingChat, serve.ServingChat)
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

    monkeypatch.setattr(serve, "MuseGlimmerServingChat", fake_serving)
    monkeypatch.setattr(serve, "build_app", lambda serving, model_id: _FakeApp())
    serve.build_app_from_args(
        SimpleNamespace(
            hf_tokenizer="hf",
            model_id="muse_glimmer",
            max_context=10,
            max_image_bytes=1234,
            tool_parser="none",
        )
    )

    assert captured["template_kwargs"] == {
        "assistant_header": "<|start|>assistant",
        "strip_rendered_bos": True,
        "append_generation_prompt_after_tool_response": True,
    }
    assert captured["serving_kwargs"]["tool_detector_cls"] is None
    assert captured["serving_kwargs"]["max_image_bytes"] == 1234
    assert captured["serving_kwargs"]["prompt_token_offset"] == 1
    assert (
        captured["serving_kwargs"]["content_filter"] is serve._strip_muse_glimmer_header
    )
    assert (
        captured["serving_kwargs"]["reasoning_extractor"]
        is serve._extract_muse_glimmer_reasoning
    )
    assert (
        captured["serving_kwargs"]["content_filter_specials"]
        == serve._MUSE_GLIMMER_HEADER_SPECIALS
    )
    assert captured["event"] == "shutdown"


def _image_request(url, *, parts=None):
    content = parts or [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "after"},
    ]
    return ChatCompletionRequest(
        model="muse_glimmer", messages=[{"role": "user", "content": content}]
    )


@pytest.mark.parametrize(
    "mime_type,payload",
    [("image/jpeg", b"jpeg-bytes"), ("image/png", b"png-bytes")],
)
def test_normalize_image_request_preserves_text_and_original(mime_type, payload):
    data = base64.b64encode(payload).decode("ascii")
    request = _image_request(f"data:{mime_type};base64,{data}")
    original = request.model_dump()

    normalized, image = serve._normalize_image_request(request, len(payload))

    assert image == {
        "image": {"encoding": "base64", "mime_type": mime_type, "data": data}
    }
    assert normalized.messages[0].content == (f"before{serve._IMAGE_PLACEHOLDER}after")
    assert request.model_dump() == original


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/image.jpg",  # @lint-ignore
        "/tmp/image.jpg",
        "data:image/gif;base64,Z2lm",
        "data:image/jpg;base64,anBn",
        "data:image/png;utf8,png",
        "data:image/png;base64,",
        "data:image/png;base64,***",
        "data:image/png;base64,QQ=",
    ],
)
def test_normalize_image_request_rejects_invalid_urls(url):
    with pytest.raises(APIError) as error:
        serve._normalize_image_request(_image_request(url), 1024)

    assert error.value.status == 400
    assert error.value.err_type == "invalid_request_error"
    assert error.value.code == "invalid_image"


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url"},
        {"type": "image_url", "image_url": "data:image/png;base64,eA=="},
        {"type": "image_url", "image_url": {}},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,eA==", "detail": "auto"},
        },
        {"type": "text", "image_url": {"url": "data:image/png;base64,eA=="}},
    ],
)
def test_normalize_image_request_rejects_bad_schema(part):
    with pytest.raises(APIError) as error:
        serve._normalize_image_request(_image_request("unused", parts=[part]), 1024)

    assert error.value.status == 400
    assert error.value.err_type == "invalid_request_error"
    assert error.value.code == "invalid_image"


def test_normalize_image_request_rejects_oversize_image():
    data = base64.b64encode(b"too large").decode("ascii")
    with pytest.raises(APIError) as error:
        serve._normalize_image_request(
            _image_request(f"data:image/jpeg;base64,{data}"), 3
        )

    assert error.value.code == "invalid_image"
    assert "3-byte limit" in error.value.message


def test_normalize_image_request_rejects_multiple_images():
    data = base64.b64encode(b"image").decode("ascii")
    image = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{data}"},
    }
    request = _image_request(
        "unused", parts=[image, {"type": "text", "text": "middle"}, image]
    )

    with pytest.raises(APIError) as error:
        serve._normalize_image_request(request, 1024)

    assert error.value.code == "invalid_image"
    assert "Only one" in error.value.message


class _ImageTemplate:
    def __init__(self):
        self.rendered_messages = None

    def turn_stop_sequences(self):
        return []

    def special_tokens(self):
        return []

    def assistant_header(self):
        return "<|start|>assistant"

    def render(self, messages, *, tools, template_kwargs):
        self.rendered_messages = messages
        if any(not isinstance(message.content, str) for message in messages):
            raise TypeError("template requires string content")
        return f"<|start|>user<|message|>{messages[0].content}<|eot|>"

    def generation_preamble(self, template_kwargs, *, tools):
        return ""

    def count_tokens(self, text):
        return len(text.split())


class _ImageRuntime:
    def __init__(self):
        self.prompt = None

    async def generate_stream(self, session_id, prompt, options, stats):
        self.prompt = prompt
        stats.prompt_tokens = 2
        stats.completion_tokens = 1
        yield "answer"


def test_muse_glimmer_serving_injects_image_segment_into_runtime():
    template = _ImageTemplate()
    runtime = _ImageRuntime()
    serving = serve.MuseGlimmerServingChat(
        runtime, template, "muse_glimmer", max_image_bytes=1024
    )
    data = base64.b64encode(b"png").decode("ascii")
    request = _image_request(f"data:image/png;base64,{data}")

    response = asyncio.run(serving.create(request))

    assert response.choices[0].message.content == "answer"
    assert runtime.prompt == PromptInput(
        segments=[
            {"text": "<|start|>user<|message|>before"},
            {
                "image": {
                    "encoding": "base64",
                    "mime_type": "image/png",
                    "data": data,
                }
            },
            {"text": "after<|eot|>"},
        ]
    )
    assert template.rendered_messages[0].content == (
        f"before{serve._IMAGE_PLACEHOLDER}after"
    )
    assert len(request.messages[0].content) == 3


def test_muse_glimmer_prompt_count_ignores_image_segment():
    serving = serve.MuseGlimmerServingChat(object(), _ImageTemplate(), "muse_glimmer")
    prompt = PromptInput(
        segments=[
            {"image": {"encoding": "base64", "mime_type": "image/png", "data": "eA=="}},
            {"text": "two tokens"},
            {"ids": [1, 2, 3]},
        ]
    )

    assert serving._count_prompt_tokens(prompt) == 5


def test_strip_muse_glimmer_header_returns_visible_answer():
    # The model self-emits " to=user<|message|>ANSWER" ahead of the visible text
    # because the generation prompt ends at "<|start|>assistant".
    assert serve._strip_muse_glimmer_header(" to=user<|message|>Paris.") == "Paris."


def test_strip_muse_glimmer_header_strips_tool_recipient():
    text = ' to=functions.get_weather<|message|>{"city": "Paris"}'
    assert serve._strip_muse_glimmer_header(text) == '{"city": "Paris"}'


def test_strip_muse_glimmer_header_passthrough_plain_text():
    assert (
        serve._strip_muse_glimmer_header("The capital of France is Paris.")
        == "The capital of France is Paris."
    )


def test_strip_muse_glimmer_header_removes_message_terminators():
    assert serve._strip_muse_glimmer_header("Test complete.<|eom|>") == "Test complete."
    assert serve._strip_muse_glimmer_header("Final answer.<|eot|>") == "Final answer."


def test_strip_muse_glimmer_header_strips_constrain_field():
    text = ' to=functions.get_weather constrain=json<|message|>{"city": "Paris"}'
    assert serve._strip_muse_glimmer_header(text) == '{"city": "Paris"}'


def test_extract_muse_glimmer_reasoning_preserves_final_answer():
    text = (
        " to=self<|message|>We should calculate first.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    reasoning, visible = serve._extract_muse_glimmer_reasoning(text)
    assert reasoning == "We should calculate first."
    assert visible == "The answer is 42."


def test_extract_muse_glimmer_reasoning_handles_multiple_private_messages():
    text = (
        "to=self<|message|>First step.<|eom|>"
        "<|start|>assistant to=self<|message|>Second step.<|eom|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    reasoning, visible = serve._extract_muse_glimmer_reasoning(text)
    assert reasoning == "First step.\n\nSecond step."
    assert visible == "Done."


def test_extract_muse_glimmer_reasoning_plain_text_fallback():
    text = "The capital of France is Paris."
    assert serve._extract_muse_glimmer_reasoning(text) == (None, text)


def test_self_reasoning_then_tool_call_content_clean_calls_unchanged():
    # End-to-end of the parser -> content_filter path: a to=self reasoning message
    # precedes an ATEM tool call. The reasoning body is surfaced as content with NO
    # leaked "to=self<|message|>" prefix, and the tool call is still extracted.
    from executorch.examples.models.muse_glimmer.serving.tool_parsers import (
        AtemToolCallDetector,
    )

    block = (
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    text = (
        " to=self<|message|>We should check the weather.<|eom|>"
        "<|start|>assistant to=get_weather<|message|>" + block + "<|eom|>"
    )
    tools = {"get_weather": {"properties": {"city": {"type": "string"}}}}
    parsed = AtemToolCallDetector().detect_and_parse(text, tools)
    # tool_calls extraction is unchanged: the get_weather call is still parsed.
    assert [c.name for c in parsed.calls] == ["get_weather"]
    reasoning, visible = serve._extract_muse_glimmer_reasoning(parsed.normal_text)
    assert reasoning == "We should check the weather."
    assert visible == ""


def test_extract_muse_glimmer_reasoning_preserves_visible_prefix():
    text = (
        "Visible prefix.\n"
        "to=self<|message|>Private step.<|eom|>"
        "<|start|>assistant to=user<|message|>Visible final.<|eot|>"
    )
    reasoning, visible = serve._extract_muse_glimmer_reasoning(text)
    assert reasoning == "Private step."
    assert visible == "Visible prefix.\n\nVisible final."


def test_strip_muse_glimmer_header_preserves_prose_lookalikes():
    # Negative cases: real prose tokens that merely CONTAIN "to=" as a substring
    # (goto=, auto=, photo=) must NOT be stripped -- only a genuine harmony header
    # (start-of-string / whitespace / ">"-introduced "to=<recipient><|message|>").
    text = "Use goto=label; set auto=true; field photo=front."
    assert serve._strip_muse_glimmer_header(text) == text


def test_tool_parser_atem_returns_detector():
    from executorch.examples.models.muse_glimmer.serving.tool_parsers import (
        AtemToolCallDetector,
    )

    assert serve._tool_detector("atem") is AtemToolCallDetector


def test_tool_parser_none_returns_none():
    assert serve._tool_detector("none") is None


def test_tool_parser_unknown_raises():
    with pytest.raises(ValueError):
        serve._tool_detector("nonexistent")


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
