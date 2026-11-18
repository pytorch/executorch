# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic tests for the Muse Glimmer OpenAI serving launcher."""

import asyncio
import base64
import json
import pathlib
from types import SimpleNamespace

import pytest

# The serving stack is built on pydantic, which ships with the llm_server
# extras (examples/llm_server/python/requirements.txt) rather than core
# ExecuTorch, so skip instead of failing collection when it is absent.
pytest.importorskip("pydantic", reason="requires llm_server serving dependencies")
pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")

from executorch.examples.llm_server.python import chat_template  # noqa: E402
from executorch.examples.llm_server.python.errors import APIError  # noqa: E402
from executorch.examples.llm_server.python.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatMessage,
)
from executorch.examples.llm_server.python.serving_chat import ServingChat  # noqa: E402
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

        def count_reasoning_tokens(self, ids):
            return None

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
        captured["serving_kwargs"]["reasoning_token_counter"].__func__
        is _FakeTemplate.count_reasoning_tokens
    )
    assert "content_filter" not in captured["serving_kwargs"]
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

    async def generate_stream(
        self, session_id, prompt, options, stats, *, observer=None
    ):
        self.prompt = prompt
        stats.prompt_tokens = 2
        stats.completion_tokens = 1
        yield "answer"


@pytest.mark.parametrize("stream", [False, True])
def test_muse_glimmer_serving_injects_image_segment_into_runtime(stream):
    template = _ImageTemplate()
    runtime = _ImageRuntime()
    serving = serve.MuseGlimmerServingChat(
        runtime, template, "muse_glimmer", max_image_bytes=1024
    )
    data = base64.b64encode(b"png").decode("ascii")
    request = _image_request(f"data:image/png;base64,{data}")

    response = asyncio.run(
        _create_message(serving, request.model_copy(update={"stream": stream}))
    )

    assert response["content"] == "answer"
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


def test_muse_glimmer_prompt_count_defers_images_to_native_worker():
    serving = serve.MuseGlimmerServingChat(object(), _ImageTemplate(), "muse_glimmer")
    prompt = PromptInput(
        segments=[
            {"image": {"encoding": "base64", "mime_type": "image/png", "data": "eA=="}},
            {"text": "two tokens"},
            {"ids": [1, 2, 3]},
        ]
    )

    assert serving._count_prompt_tokens(prompt) is None


def test_extract_muse_glimmer_reasoning_preserves_final_answer():
    text = (
        " to=self<|message|>We should calculate first.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    reasoning, visible = serve._extract_muse_glimmer_reasoning(text)
    assert reasoning == "We should calculate first."
    assert visible == "The answer is 42."


class _StubTemplate:
    """Minimal template double for serving-layer tests.

    Turn-1 assertions below only need the response fields, so the render is a
    constant; the two-turn test supplies its own Harmony-mimicking template.
    """

    def turn_stop_sequences(self):
        return []

    def count_reasoning_tokens(self, ids):
        return None

    def special_tokens(self):
        return []

    def generation_preamble(self, template_kwargs=None, tools=None):
        return ""

    def assistant_header(self):
        return "<|start|>assistant"

    def render(self, messages, tools=None, template_kwargs=None):
        return "PROMPT"


class _HarmonyTemplate(_StubTemplate):
    """Template double mimicking the ATEM emission rule: an assistant turn's
    thinking re-emits first as a `to=self` block, then the visible content as
    `to=user` -- and the generation prompt ends at the bare header."""

    HDR = "<|start|>assistant"

    def assistant_header(self):
        return self.HDR

    def render(self, messages, tools=None, template_kwargs=None):
        out = ["<bos>"]
        for m in messages:
            c = m.content if isinstance(m.content, str) else ""
            if m.role == "assistant":
                if m.reasoning_content:
                    out.append(
                        f"{self.HDR} to=self<|message|>{m.reasoning_content}<|eom|>"
                    )
                out.append(f"{self.HDR} to=user<|message|>{c}<|eot|>")
            elif m.role == "user":
                out.append(f"<|start|>user<|message|>{c}<|eot|>")
        out.append(self.HDR)
        return "".join(out)


class _FakeRuntime:
    """Canned worker: yields fixed raw text per call, fills stats, and captures
    each prompt, so tests assert on the worker-facing contract (responses and
    PromptInputs) instead of private helpers."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.prompts = []

    @property
    def healthy(self):
        return True

    async def open(self, session_id):
        return None

    def stop(self):
        pass

    async def generate_stream(
        self, session_id, prompt, options, stats=None, *, observer=None
    ):
        self.prompts.append(prompt)
        out = self._outputs.pop(0)
        if stats is not None:
            stats.prompt_tokens = out.get("prompt_tokens", 0)
            stats.completion_tokens = out.get("completion_tokens", 0)
            stats.finish_reason = out.get("finish_reason", "stop")
            stats.matched_stop = out.get("matched_stop")
            stats.generated_token_ids = (
                list(out["gen_ids"]) if out.get("gen_ids") is not None else None
            )
            stats.session_reset_reason = out.get("reason")
            stats.reused_prompt_tokens = out.get("reused", 0)
            stats.prefilled_prompt_tokens = out.get("prefilled", 0)
        yield out["text"]


def _benc(text):
    """Deterministic byte encoding: id-sequence equality <=> string equality,
    so assembled PromptInputs can be compared without peeking at segments."""

    return [ord(c) for c in text]


def _assemble(prompt):
    if prompt.text is not None:
        return _benc(prompt.text)
    out = []
    for seg in prompt.segments or []:
        out += _benc(seg["text"]) if "text" in seg else list(seg["ids"])
    return out


def _glimmer_serving(raw_texts, template=None):
    # Mirrors production wiring (build_app_from_args): the real extractor is
    # plugged into the serving layer; assertions below target only responses
    # and worker prompts, never the extractor itself.
    runtime = _FakeRuntime(raw_texts)
    serving = serve.MuseGlimmerServingChat(
        runtime,
        template or _StubTemplate(),
        "test-model",
        reasoning_extractor=serve._extract_muse_glimmer_reasoning,
        content_filter_specials=serve._MUSE_GLIMMER_HEADER_SPECIALS,
    )
    return serving, runtime


@pytest.mark.parametrize("serving_cls", [ServingChat, serve.MuseGlimmerServingChat])
@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_consumes_budget_and_preserves_visible_turn(serving_cls, stream):
    class Template(_StubTemplate):
        _hf = SimpleNamespace(
            eos_token="<|end_of_text|>",
            all_special_tokens=["<|end_of_text|>", "<|eot|>", "<|eom|>"],
        )
        turn_stop_sequences = chat_template.ChatTemplate.turn_stop_sequences

    class Runtime(_FakeRuntime):
        supports_ignore_eos = True
        default_stop_strings = ["<|eot|>", "<|end_of_text|>"]

        async def generate_stream(
            self, session_id, prompt, options, stats=None, *, observer=None
        ):
            assert options.ignore_eos is True
            assert options.stop == ["USER_STOP"]
            assert options.max_new_tokens == 3
            yield "answer"
            yield "<|eot|>"
            yield "hidden"
            self.drained = True
            stats.completion_tokens = 3
            stats.finish_reason = "length"

    async def scenario():
        runtime = Runtime([])
        runtime.drained = False
        serving = serving_cls(runtime, Template(), "test-model")
        response = await _create_message(
            serving,
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                stream=stream,
                ignore_eos=True,
                max_tokens=3,
                stop="USER_STOP",
            ),
        )
        assert response["content"] == "answer"
        assert runtime.drained

    asyncio.run(scenario())


@pytest.mark.parametrize("serving_cls", [ServingChat, serve.MuseGlimmerServingChat])
@pytest.mark.parametrize("stream", [False, True])
def test_bare_generator_stop_is_scoped_and_finalizes(serving_cls, stream):
    class Runtime(_FakeRuntime):
        def __init__(self):
            super().__init__(
                [
                    {
                        "text": "keepSTOPdrop",
                        "finish_reason": "length",
                        "completion_tokens": 3,
                    }
                ]
            )
            self.stops = 0
            self.closed = []

        def stop(self):
            self.stops += 1

        async def generate_stream(
            self, session_id, prompt, options, stats=None, *, observer=None
        ):
            try:
                async for token in super().generate_stream(
                    session_id, prompt, options, stats
                ):
                    yield token
            finally:
                self.closed.append(session_id)

    async def scenario():
        runtime = Runtime()
        serving = serving_cls(runtime, _StubTemplate(), "test-model")
        response = await _create_message(
            serving,
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                session_id="s",
                stream=stream,
                stop="STOP",
            ),
        )
        assert response["content"] == "keep"
        assert runtime.stops == 1
        assert runtime.closed == ["s"]
        assert serving._transcript._turns["s"][0]["ids"] is None
        assert not serving._transactions._entries

    asyncio.run(scenario())


@pytest.mark.parametrize("serving_cls", [ServingChat, serve.MuseGlimmerServingChat])
@pytest.mark.parametrize("stream", [False, True])
def test_bare_generator_cancellation_keeps_transaction_until_cleanup(
    serving_cls, stream
):
    class Runtime(_FakeRuntime):
        def __init__(self):
            super().__init__([])
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closing = asyncio.Event()
            self.finish_cleanup = asyncio.Event()
            self.closed = asyncio.Event()
            self.reset_called = asyncio.Event()

        async def generate_stream(
            self, session_id, prompt, options, stats=None, *, observer=None
        ):
            try:
                yield "first"
                self.started.set()
                await self.release.wait()
            finally:
                self.closing.set()
                await self.finish_cleanup.wait()
                self.closed.set()

        async def reset(self, session_id):
            assert self.closed.is_set()
            self.reset_called.set()

    async def scenario():
        runtime = Runtime()
        serving = serving_cls(runtime, _StubTemplate(), "test-model")
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            session_id="s",
            stream=stream,
        )
        tasks = []
        try:
            if stream:
                response = await serving.create(request)

                async def consume():
                    async for _ in response:
                        pass

                reader = asyncio.create_task(consume())
            else:
                reader = asyncio.create_task(serving.create(request))
            tasks.append(reader)
            await asyncio.wait_for(runtime.started.wait(), 2)
            if stream:
                close = asyncio.create_task(response.aclose())
                tasks.append(close)
            else:
                reader.cancel()
            await asyncio.wait_for(runtime.closing.wait(), 2)
            reset = asyncio.create_task(serving.reset_session("s"))
            tasks.append(reset)
            await asyncio.sleep(0)
            assert not runtime.reset_called.is_set()
            assert serving._transactions._entries["s"].users == 2
            runtime.finish_cleanup.set()
            if stream:
                await asyncio.wait_for(close, 2)
            with pytest.raises(asyncio.CancelledError):
                await reader
            await asyncio.wait_for(reset, 2)
            assert runtime.closed.is_set() and runtime.reset_called.is_set()
            assert not serving._transactions._entries
            assert "s" not in serving._transcript._turns
        finally:
            runtime.finish_cleanup.set()
            runtime.release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


def test_thinking_survives_response_verbatim():
    # Fix-1 at the API seam: raw thinking bytes (boundary newlines included)
    # must reach the client unmodified in reasoning_content.
    raw = (
        " to=self<|message|>\nWe should calculate first.\n<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
    )
    serving, _ = _glimmer_serving([{"text": raw}])
    resp = asyncio.run(
        serving.create(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="What is 6*7?")],
                # Explicit: this tests the returned reasoning, not the default.
                chat_template_kwargs={"return_reasoning": True},
            )
        )
    )
    msg = resp.choices[0].message
    assert msg.reasoning_content == "\nWe should calculate first.\n"
    assert msg.content == "The answer is 42."


@pytest.mark.parametrize("recipient", ["self", "user"])
def test_whitespace_only_channel_does_not_add_response_gaps(recipient):
    raw = (
        "to=self<|message|>\nThink first.\n<|eom|>"
        "<|start|>assistant to=user<|message|>Part one.<|eom|>"
        f"<|start|>assistant to={recipient}<|message|> \t\n<|eom|>"
        "<|start|>assistant to=user<|message|>Part two.<|eot|>"
    )
    serving, _ = _glimmer_serving([{"text": raw}])
    response = asyncio.run(
        serving.create(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        )
    )
    message = response.choices[0].message
    assert message.reasoning_content == "\nThink first.\n"
    assert message.content == "Part one.\n\nPart two."


def test_whitespace_only_reasoning_is_omitted():
    raw = (
        "to=self<|message|> \t\n<|eom|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    serving, _ = _glimmer_serving([{"text": raw}])
    response = asyncio.run(
        serving.create(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        )
    )
    message = response.choices[0].message
    assert message.reasoning_content is None
    assert message.content == "Done."


def test_multiple_thinking_blocks_joined_without_protocol_markers():
    # Thinking blocks are client-facing text: they must be joined with a
    # newline (SGLang parity) so no <|eom|>/<|start|>/to=self framing leaks
    # into reasoning_content. Replay fidelity does not depend on this text
    # (warm resume splices the stored generated ids over the echo).
    raw = (
        "to=self<|message|>First step.<|eom|>"
        "<|start|>assistant to=self<|message|>Second step.<|eom|>"
        "<|start|>assistant to=user<|message|>Done.<|eot|>"
    )
    serving, _ = _glimmer_serving([{"text": raw}])
    resp = asyncio.run(
        serving.create(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="What is 6*7?")],
                # Explicit: this tests the returned reasoning, not the default.
                chat_template_kwargs={"return_reasoning": True},
            )
        )
    )
    reasoning = resp.choices[0].message.reasoning_content
    assert reasoning == "First step.\nSecond step."
    assert "<|" not in reasoning and "to=self" not in reasoning


async def _create_message(serving, request):
    response = await serving.create(request)
    if not request.stream:
        return response.choices[0].message.model_dump(exclude_none=True)
    message = {"role": "assistant"}
    done = False
    async for event in response:
        payload = event.removeprefix("data: ").strip()
        if payload == "[DONE]":
            done = True
            continue
        chunk = json.loads(payload)
        assert "error" not in chunk
        delta = chunk["choices"][0]["delta"]
        for field in ("content", "reasoning_content"):
            if field in delta:
                message[field] = message.get(field, "") + delta[field]
        if delta.get("tool_calls"):
            message.setdefault("tool_calls", []).extend(delta["tool_calls"])
    assert done
    return message


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "echo_mode", ["edited", "unchanged", "omitted", "null", "opt-out", "opt-out-null"]
)
def test_thinking_turn_replays_ids_only_for_unchanged_history(stream, echo_mode):
    # The worker reports non-terminal ids; the next rendered turn supplies EOS.
    # Two raw thinking blocks become one client-visible reasoning string.
    raw1 = (
        " to=self<|message|>\nFirst step.\n<|eom|>"
        "<|start|>assistant to=self<|message|>Second step.<|eom|>"
        "<|start|>assistant to=user<|message|>156"
    )
    raw2 = " to=user<|message|>210"
    template = _HarmonyTemplate()
    serving, runtime = _glimmer_serving(
        [
            {"text": raw1, "gen_ids": _benc(raw1)},
            {"text": raw2, "gen_ids": _benc(raw2)},
        ],
        template=template,
    )
    u1 = ChatMessage(role="user", content="What is 12*13?")
    first = asyncio.run(
        _create_message(
            serving,
            ChatCompletionRequest(
                messages=[u1],
                session_id="s",
                stream=stream,
                chat_template_kwargs=(
                    {"return_reasoning": False}
                    if echo_mode.startswith("opt-out")
                    else None
                ),
            ),
        )
    )
    assert first["content"] == "156"
    if echo_mode.startswith("opt-out"):
        assert "reasoning_content" not in first
    else:
        assert first["reasoning_content"] == "\nFirst step.\n\nSecond step."
    if echo_mode == "edited":
        first["reasoning_content"] = "Use this corrected reasoning instead."
    elif echo_mode == "omitted":
        first.pop("reasoning_content")
    elif echo_mode in ("null", "opt-out-null"):
        first["reasoning_content"] = None
    request2 = ChatCompletionRequest(
        messages=[u1, ChatMessage(**first), ChatMessage(role="user", content="And?")],
        session_id="s",
    )
    second = asyncio.run(serving.create(request2))
    assert second.choices[0].message.content == "210"
    prompt2 = runtime.prompts[1]
    resident = _benc(runtime.prompts[0].text + raw1)
    trailing = "<|eot|><|start|>user<|message|>And?<|eot|><|start|>assistant"
    if echo_mode == "edited":
        assert _assemble(prompt2) == _benc(template.render(request2.messages))
        assert _assemble(prompt2) != resident + _benc(trailing)
    else:
        assert _assemble(prompt2) == resident + _benc(trailing)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("edited_field", ["content", "reasoning_content"])
@pytest.mark.parametrize("restore_history", [False, True])
def test_replay_recovers_after_edit_without_restoring_stale_turns(
    stream, edited_field, restore_history
):
    raw = [
        f" to=self<|message|>reasoning {i}<|eom|>"
        f"<|start|>assistant to=user<|message|>a{i}"
        for i in range(1, 6)
    ]
    generated = [_benc(text) for text in raw]
    template = _HarmonyTemplate()
    serving, runtime = _glimmer_serving(
        [{"text": text, "gen_ids": ids} for text, ids in zip(raw, generated)],
        template=template,
    )

    async def conversation():
        messages = []
        for i in range(5):
            if i == 3:
                original = messages[3]
                messages[3] = original.model_copy(update={edited_field: "EDITED"})
            elif i == 4 and restore_history:
                messages[3] = original
            messages.append(ChatMessage(role="user", content=f"u{i + 1}"))
            response = await _create_message(
                serving,
                ChatCompletionRequest(
                    messages=list(messages), session_id="s", stream=stream
                ),
            )
            assert response["content"] == f"a{i + 1}"
            prompt = runtime.prompts[-1]
            assert _assemble(prompt) == _benc(template.render(messages))
            expected = generated[:i] if i < 3 else [generated[0]]
            if i == 4:
                if not restore_history:
                    expected.append(generated[3])
                    resident = _assemble(runtime.prompts[3]) + generated[3]
                    assert _assemble(prompt)[: len(resident)] == resident
            assert [s["ids"] for s in prompt.segments or [] if "ids" in s] == expected
            messages.append(ChatMessage(**response))

    asyncio.run(conversation())


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
    from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import (
        ToolSchema,
    )

    parsed = AtemToolCallDetector().detect_and_parse(
        text, {name: ToolSchema(schema) for name, schema in tools.items()}
    )
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


@pytest.mark.parametrize(
    "control", [{"reasoning": {"effort": "low"}}, {"reasoning_effort": "low"}]
)
def test_openrouter_reasoning_effort_reaches_template(control):
    class Template(_StubTemplate):
        seen = None

        def render(self, messages, tools=None, template_kwargs=None):
            self.seen = template_kwargs
            return super().render(
                messages, tools=tools, template_kwargs=template_kwargs
            )

    template = Template()
    serving, _ = _glimmer_serving(
        [{"text": " to=user<|message|>OK"}], template=template
    )
    asyncio.run(
        serving.create(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], **control
            )
        )
    )
    assert template.seen["reasoning_strength"] == "low"


def test_bundled_template_preserves_developer_and_canonical_tool_history():
    import os

    hf = os.environ.get("MUSE_GLIMMER_HF_DIR")
    if not hf:
        pytest.skip("set MUSE_GLIMMER_HF_DIR to a local Muse tokenizer")
    from executorch.examples.llm_server.python.protocol import FunctionCall, ToolCall
    from executorch.examples.models.muse_glimmer.serving.chat_template import (
        MuseGlimmerChatTemplate,
    )

    template = MuseGlimmerChatTemplate(
        hf, assistant_header="<|start|>assistant", strip_rendered_bos=True
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    messages = [
        ChatMessage(role="system", content="SYSTEM_RULE"),
        ChatMessage(role="developer", content="DEVELOPER_RULE"),
        ChatMessage(role="user", content="Read it"),
        ChatMessage(
            role="assistant",
            content="Opening the file",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(
                        name="read_file", arguments='{"path":"policy.md"}'
                    ),
                )
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_1", content="FILE_RESULT"),
    ]
    prompt = template.render(messages, tools=tools)
    assert "DEVELOPER_RULE" in prompt and "SYSTEM_RULE" in prompt
    assert "Opening the file" in prompt and "FILE_RESULT" in prompt
    assert 'name="read_file.read_file"' in prompt
    assert tools[0]["function"]["name"] == "read_file"
    assert messages[3].tool_calls[0].function.name == "read_file"
    assert prompt.count("// Function schemas") == 1


@pytest.mark.parametrize("stream", [False, True])
def test_tool_reasoning_stream_preserves_boundary_whitespace(stream):
    from executorch.examples.models.muse_glimmer.serving.stream_parser import (
        MuseGlimmerToolReasoningParser,
    )
    from executorch.examples.models.muse_glimmer.serving.tool_parsers import (
        AtemToolCallDetector,
    )

    raw = (
        " to=self<|message|>First step. \n<|eom|>"
        "<|start|>assistant to=self<|message|>Second step.\n<|eom|>"
        "<|start|>assistant to=f.f<|message|><atem:function_calls>"
        '<atem:invoke name="f.f"></atem:invoke></atem:function_calls>'
    )
    serving, _ = _glimmer_serving([{"text": raw}])
    serving._tool_detector_cls = AtemToolCallDetector
    serving._tool_reasoning_parser_factory = MuseGlimmerToolReasoningParser
    message = asyncio.run(
        _create_message(
            serving,
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Use f")],
                stream=stream,
                tools=[
                    {"type": "function", "function": {"name": "f", "parameters": {}}}
                ],
            ),
        )
    )
    assert message["reasoning_content"] == "First step. \n\nSecond step.\n"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("text_tokens,completion", [(10, 1), (6, 2)])
def test_image_overflow_uses_worker_counts(stream, dialect, text_tokens, completion):
    from executorch.examples.llm_server.python.server import build_app
    from executorch.examples.llm_server.python.worker_client import WorkerError
    from fastapi.testclient import TestClient

    image_positions = 16
    expected = text_tokens + 1 + image_positions

    class Template(_ImageTemplate):
        def render(self, messages, *, tools, template_kwargs):
            return "text " * text_tokens + serve._IMAGE_PLACEHOLDER

    class Runtime(_ImageRuntime):
        async def generate_stream(
            self, session_id, prompt, options, stats, *, observer=None
        ):
            self.prompt = prompt
            raise WorkerError(
                "image-expanded prompt exceeds context",
                code="context_length_exceeded",
                context_overflow={
                    "prompt_tokens": expected,
                    "context_length": 8,
                    "completion_tokens": completion,
                },
            )
            yield

    runtime = Runtime()
    serving = serve.MuseGlimmerServingChat(
        runtime,
        Template(),
        "muse_glimmer",
        max_context=8,
        prompt_token_offset=1,
        api_dialect=dialect,
    )
    body = _image_request("data:image/png;base64,eA==").model_dump(exclude_none=True)
    body.update(stream=stream, max_tokens=completion)
    response = TestClient(build_app(serving, "muse_glimmer")).post(
        "/v1/chat/completions", json=body
    )
    assert runtime.prompt is not None
    assert any("image" in segment for segment in runtime.prompt.segments)
    if stream:
        assert response.status_code == 200
        errors = [
            json.loads(line[6:])["error"]
            for line in response.text.splitlines()
            if line.startswith('data: {"error"')
        ]
        assert len(errors) == 1
        error = errors[0]
    else:
        assert response.status_code == 400
        error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert str(expected) in error["message"]
    if dialect == "openrouter":
        assert error["metadata"]["prompt_tokens"] == expected
        assert error["metadata"]["completion_tokens"] == completion
    else:
        assert "metadata" not in error


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("context_kind", ["image", "external_reasoning"])
@pytest.mark.parametrize("changed", [False, True])
def test_changed_model_context_cannot_restore_old_generated_tokens(
    stream, context_kind, changed
):
    async def scenario():
        def history(value):
            if context_kind == "image":
                return [
                    ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Describe this image"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(value.encode()).decode()
                                },
                            },
                        ],
                    )
                ]
            return [
                ChatMessage(role="user", content="Earlier question"),
                ChatMessage(
                    role="assistant", content="Earlier answer", reasoning_content=value
                ),
                ChatMessage(role="user", content="Continue"),
            ]

        serving, runtime = _glimmer_serving(
            [
                {
                    "text": " to=self<|message|>Old context.<|eom|><|start|>assistant to=user<|message|>OK",
                    "gen_ids": [777],
                },
                {"text": " to=user<|message|>Fresh answer"},
            ],
            template=_HarmonyTemplate(),
        )
        answer = await _create_message(
            serving,
            ChatCompletionRequest(
                messages=history("before"),
                session_id="s",
                stream=stream,
            ),
        )
        assert answer["content"] == "OK"
        await _create_message(
            serving,
            ChatCompletionRequest(
                messages=history("after" if changed else "before")
                + [
                    ChatMessage(role="assistant", content="OK"),
                    ChatMessage(role="user", content="Next"),
                ],
                session_id="s",
                stream=stream,
            ),
        )
        assert any(
            segment.get("ids") == [777] for segment in runtime.prompts[1].segments or []
        ) == (not changed)

    asyncio.run(scenario())
