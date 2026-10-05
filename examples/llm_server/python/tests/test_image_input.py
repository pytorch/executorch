# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inline-image bounds, real Jinja bindings, and native async HTTP contracts."""

import asyncio
import base64
import hashlib
import json
import struct
import zlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.errors import APIError
from executorch.examples.llm_server.python.image_input import (
    bind_image_segments,
    has_images,
    ImageLimits,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    prepare_image_bindings,
    validate_image,
)
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.openai_transcript import (
    OpenAITranscriptState,
)
from executorch.examples.llm_server.python.protocol import ChatMessage
from executorch.examples.llm_server.python.request_body import BoundedChatBody
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tests.test_concurrent_serving import (
    _call,
    _NativeWorker,
)
from executorch.examples.llm_server.python.tests.test_multiplexed_worker_client import (
    _AsyncProc,
    _fake_client,
    _run,
)
from executorch.examples.llm_server.python.worker_client import WorkerError
from jinja2 import Environment

LIMITS = ImageLimits(1, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS)
# Actual 1x1 RGB JPEG, generated once; runtime tests need no Pillow dependency.
JPEG_BASE64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA"
    "AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcY"
    "GRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKT"
    "lJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP0"
    "9fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3"
    "AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaan"
    "qKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIR"
    "AxEAPwDi6KKK+ZP3E//Z"
)


def png(width=1, height=1, padding=0):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + (chunk(b"tEXt", b"x" * padding) if padding else b"")
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


def encoded(data):
    return base64.b64encode(data).decode("ascii")


def image_part(data=None, mime="image/png"):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{encoded(png() if data is None else data)}"
        },
    }


class JinjaTokenizer:
    all_special_tokens = []
    eos_token = None
    chat_template = (
        "{% for m in messages %}<|im_start|>{{m.role}}\n"
        "{% if m.content is string %}{{m.content}}{% else %}"
        "{% for p in m.content or [] %}{{p.text}}{% endfor %}{% endif %}"
        "{% if m.tool_calls %}{{m.tool_calls|tojson}}{% endif %}<|im_end|>\n"
        "{% endfor %}<|im_start|>assistant\n"
    )

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, **kwargs
    ):
        return (
            Environment(autoescape=False)
            .from_string(self.chat_template)
            .render(messages=messages, tools=tools, **kwargs)
        )

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


def template():
    result = ChatTemplate(allow_fallback=True)
    result._hf = JinjaTokenizer()
    return result


def test_exact_hard_byte_limit():
    data = png(padding=MAX_IMAGE_BYTES - len(png()) - 12)
    assert len(data) == MAX_IMAGE_BYTES
    assert validate_image("image/png", encoded(data), LIMITS)
    with pytest.raises(APIError):
        validate_image("image/png", encoded(data + b"x"), LIMITS)


def test_oversized_base64_is_rejected_before_decode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized image must not allocate decoded bytes")

    data = encoded(b"x" * (MAX_IMAGE_BYTES + 1))
    monkeypatch.setattr(base64, "b64decode", forbidden)
    with pytest.raises(APIError):
        validate_image("image/png", data, LIMITS)


def test_real_jpeg_fixture():
    assert validate_image("image/jpeg", JPEG_BASE64, LIMITS)


def test_png_and_jpeg_metadata_without_pixel_decode():
    assert validate_image("image/png", encoded(png()), LIMITS) == {
        "image": {"mime_type": "image/png", "data": encoded(png())}
    }
    # A frame + scan header is enough for metadata validation, not a claim of
    # successful entropy decoding (native preparation owns full image decode).
    jpeg = bytes.fromhex("ffd8 ffc0000b080001000101011100 ffda0008010100003f00 00 ffd9")
    assert (
        validate_image("image/jpeg", encoded(jpeg), LIMITS)["image"]["mime_type"]
        == "image/jpeg"
    )


@pytest.mark.parametrize(
    "value",
    ["", "AA", "AA===", "AB==", "a b=", "\u00e9", "!!!!", encoded(png()) + "\n"],
)
def test_reject_noncanonical_base64(value):
    with pytest.raises(APIError):
        validate_image("image/png", value, LIMITS)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a.png",
        "file:///tmp/a.png",
        "/tmp/a.png",
        "data:image/gif;base64,AAAA",
        "data:image/png;charset=utf-8;base64,AAAA",
        "data:image/png,AAAA",
    ],
)
def test_no_url_or_file_fetch(url):
    message = ChatMessage(
        role="user", content=[{"type": "image_url", "image_url": {"url": url}}]
    )
    with pytest.raises(APIError):
        prepare_image_bindings([message], LIMITS)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not an image",
        png()[:-1],
        png() + b"trailing",
        png()[:20] + b"bad" + png()[23:],
    ],
)
def test_reject_bad_png_structure(data):
    with pytest.raises(APIError):
        validate_image("image/png", encoded(data), LIMITS)


@pytest.mark.parametrize(
    "width,height", [(0, 1), (1, 0), (4097, 1), (2049, 2048), (2**32 - 1, 2**32 - 1)]
)
def test_reject_dimensions_before_pixel_allocation(width, height):
    with pytest.raises(APIError, match="dimensions"):
        validate_image("image/png", encoded(png(width, height)), LIMITS)


def test_byte_and_pixel_limits_are_inclusive():
    data = png()
    assert validate_image("image/png", encoded(data), ImageLimits(1, len(data), 1))
    with pytest.raises(APIError):
        validate_image("image/png", encoded(data), ImageLimits(1, len(data) - 1, 1))
    assert validate_image("image/png", encoded(png(2048, 2048)), LIMITS)
    with pytest.raises(APIError):
        validate_image("image/jpeg", encoded(data), LIMITS)


def test_image_binding_preserves_tool_history():
    messages = [
        ChatMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call-1",
                    "function": {"name": "look", "arguments": '{"x":1}'},
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call-1", content="tool output"),
        ChatMessage(role="user", content=[image_part()]),
    ]
    modified, bindings = prepare_image_bindings(messages, LIMITS)
    assert modified[:2] == messages[:2]
    segments = bind_image_segments([{"text": template().render(modified)}], bindings)
    text = "".join(segment.get("text", "") for segment in segments)
    assert '"name": "look"' in text
    assert '"arguments": {"x": 1}' in text
    assert "tool output" in text
    assert sum("image" in segment for segment in segments) == 1


def test_image_count_is_across_entire_history():
    messages = [
        ChatMessage(role="user", content=[image_part()]),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content=[image_part()]),
    ]
    with pytest.raises(APIError, match="one image"):
        prepare_image_bindings(messages, replace(LIMITS, max_images=99))
    with pytest.raises(APIError) as unsupported:
        prepare_image_bindings(messages[:1], None)
    assert unsupported.value.code == "unsupported_image"


def test_jinja_binding_order_and_token_splice_preserve_nonimage_content():
    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "before"},
                image_part(),
                {"type": "text", "text": "after"},
            ],
        )
    ]
    modified, bindings = prepare_image_bindings(messages, LIMITS)
    rendered = template().render(modified)
    result = bind_image_segments(
        [{"text": rendered}, {"ids": [21, 22]}, {"text": "tail"}], bindings
    )
    assert result == [
        {"text": "<|im_start|>user\nbefore"},
        {"image": {"mime_type": "image/png", "data": encoded(png())}},
        {"text": "after<|im_end|>\n<|im_start|>assistant"},
        {"ids": [21, 22]},
        {"text": "tail"},
    ]
    assert messages[0].content[1] == image_part()


@pytest.mark.parametrize("rendered", ["", "marker marker", "MARKER", "second marker"])
def test_missing_duplicate_changed_or_reordered_bindings_fail(rendered):
    bindings = (
        {"marker": {"image": {}}, "second": {"image": {}}}
        if rendered == "second marker"
        else {"marker": {"image": {}}}
    )
    with pytest.raises(APIError):
        bind_image_segments([{"text": rendered}], bindings)


def test_splice_cannot_split_image_marker():
    with pytest.raises(APIError):
        bind_image_segments(
            [{"text": "mar"}, {"ids": [1]}, {"text": "ker"}], {"marker": {"image": {}}}
        )


def _image_prompt(state, messages):
    modified, bindings = prepare_image_bindings(messages, LIMITS)
    return state.build_prompt_input(
        session_id="s",
        messages=messages,
        render_messages=modified,
        rendered_prompt=state._template.render(modified),
        tools=None,
        template_kwargs={},
        image_bindings=bindings,
    )


def _record(state, messages, content="reply", ids=(700, 701), **kwargs):
    state.record_assistant_turn(
        session_id="s",
        content=content,
        tool_calls=kwargs.pop("tool_calls", None),
        generated_token_ids=list(ids) if ids is not None else None,
        prior_turns=sum(m.role == "assistant" for m in messages),
        source_messages=messages,
        **kwargs,
    )


@pytest.mark.parametrize("positions", [set(), {0}, {0, 2, 4, 6}, set(range(7))])
def test_history_prefixes_match_full_canonical_json(positions):
    messages = [
        ChatMessage(role="assistant", content="first"),
        ChatMessage(role="system", content="\u00e9\U0001f600", name="\ud800"),
        ChatMessage(
            role="user", content=[image_part(), {"type": "text", "text": "after"}]
        ),
        ChatMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call-1",
                    "function": {
                        "name": "look",
                        "arguments": ' { "b": [2, null], "a": 1 } ',
                    },
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call-1", content="\udfff"),
        ChatMessage(role="assistant", content=None, reasoning_content="reason"),
    ]
    state = OpenAITranscriptState(template())
    prefixes = state._history_prefixes(messages, positions)
    assert prefixes == {
        pos: (state._history_fingerprint(messages[:pos]), has_images(messages[:pos]))
        for pos in positions
    }
    assert state._history_prefixes([], {0}) == {
        0: (state._history_fingerprint([]), False)
    }


@pytest.mark.parametrize("turns", [2, 32])
def test_image_prefix_serialization_is_linear(turns, monkeypatch):
    state = OpenAITranscriptState(template())
    messages = [ChatMessage(role="user", content=[image_part(png(padding=128 * 1024))])]
    for turn in range(turns):
        _record(state, messages, content=f"reply-{turn}", ids=[700 + turn])
        messages.extend(
            [
                ChatMessage(role="assistant", content=f"reply-{turn}"),
                ChatMessage(role="user", content="next"),
            ]
        )
    original_dump = ChatMessage.model_dump
    serialized = []

    def counted_dump(message, **kwargs):
        if kwargs.get("mode") == "json":
            serialized.append(id(message))
        return original_dump(message, **kwargs)

    monkeypatch.setattr(ChatMessage, "model_dump", counted_dump)
    prompt = _image_prompt(state, messages)
    last_assistant = len(messages) - 2
    assert serialized == [id(message) for message in messages[:last_assistant]]
    assert [s["ids"] for s in prompt.segments if "ids" in s] == [
        [700 + turn] for turn in range(turns)
    ]


@pytest.mark.parametrize("later_image", [False, True])
@pytest.mark.parametrize("recorded", [False, True])
def test_no_prefix_hashing_without_image_related_candidates(
    later_image, recorded, monkeypatch
):
    state = OpenAITranscriptState(template())
    messages = [ChatMessage(role="user", content="original")]
    if recorded:
        _record(state, messages)
    messages.append(ChatMessage(role="assistant", content="reply"))
    if later_image:
        messages.append(ChatMessage(role="user", content=[image_part()]))

    def forbidden(*args, **kwargs):
        raise AssertionError("text-only candidates must not hash source history")

    monkeypatch.setattr(state, "_history_prefixes", forbidden)
    monkeypatch.setattr(state, "_history_fingerprint", forbidden)
    prompt = _image_prompt(state, messages)
    assert [s["ids"] for s in prompt.segments or [] if "ids" in s] == (
        [[700, 701]] if recorded else []
    )


@pytest.mark.parametrize("name", ["\ud800", "\udfff"])
def test_image_provenance_escapes_surrogates_in_ignored_metadata(name):
    state = OpenAITranscriptState(template())
    source = [ChatMessage(role="user", content=[image_part()], name=name)]
    canonical = json.dumps(
        [message.model_dump(mode="json") for message in source],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode("ascii")).digest()
    _record(state, source)
    assert state._turns["s"][0]["history_fp"] == expected
    echoed = [ChatMessage(**message) for message in json.loads(canonical)]
    assert state._history_fingerprint(echoed) == expected
    prompt = _image_prompt(
        state, echoed + [ChatMessage(role="assistant", content="reply")]
    )
    assert [s["ids"] for s in prompt.segments if "ids" in s] == [[700, 701]]
    assert "".join(s.get("text", "") for s in prompt.segments).encode("utf-8")
    assert state._turns["s"][0]["history_fp"] == expected
    echoed[0].name = "changed"
    assert state._history_fingerprint(echoed) != expected


@pytest.mark.parametrize("ids", [None, [], [700, 701]])
def test_image_provenance_is_independent_of_rendering_uuid(ids):
    state = OpenAITranscriptState(template())
    source = [ChatMessage(role="user", content=[image_part()])]
    _record(state, source, ids=ids)
    history = source + [ChatMessage(role="assistant", content="reply")]
    record = dict(state._turns["s"][0])
    prompts = []
    for token in ("first", "second"):
        with patch("uuid.uuid4", return_value=SimpleNamespace(hex=token)):
            prompts.append(_image_prompt(state, history))
    assert prompts[0] == prompts[1]
    segments = prompts[0].segments
    assert [s["ids"] for s in segments if "ids" in s] == ([] if ids is None else [ids])
    assert sum("image" in s for s in segments) == 1
    assert ("reply" in "".join(s.get("text", "") for s in segments)) == (ids is None)
    assert state._turns["s"][0] == record


@pytest.mark.parametrize(
    "edit", ["bytes", "order", "text", "system", "removed", "missing_provenance"]
)
def test_image_history_edit_invalidates_first_mismatch_and_tail(edit):
    state = OpenAITranscriptState(template())
    source = [ChatMessage(role="system", content="original")]
    _record(state, source, content="early", ids=[600])
    source += [
        ChatMessage(role="assistant", content="early"),
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "before"},
                image_part(),
                {"type": "text", "text": "after"},
            ],
        ),
    ]
    _record(state, source)
    source += [
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="next"),
    ]
    _record(state, source, content="last", ids=[800])
    history = [m.model_copy(deep=True) for m in source] + [
        ChatMessage(role="assistant", content="last")
    ]
    if edit == "bytes":
        history[2].content[1] = image_part(png(padding=1))
    elif edit == "order":
        history[2].content.reverse()
    elif edit == "text":
        history[2].content[0]["text"] = "edited"
    elif edit == "system":
        history[0].content = "edited"
    elif edit == "removed":
        history[2].content = "before after"
    else:
        del state._turns["s"][1]["history_fp"]
        del state._turns["s"][1]["history_has_images"]
    prompt = _image_prompt(state, history)
    assert [s["ids"] for s in prompt.segments if "ids" in s] == [[600]]
    assert set(state._turns["s"]) == {0}
    # Restoring an old history cannot resurrect the invalidated tail.
    prompt = _image_prompt(
        state, source + [ChatMessage(role="assistant", content="last")]
    )
    assert [s["ids"] for s in prompt.segments if "ids" in s] == [[600]]


@pytest.mark.parametrize("later_image", [False, True])
@pytest.mark.parametrize("provenance", [False, True])
def test_text_only_prefix_keeps_existing_matching_behavior(later_image, provenance):
    state = OpenAITranscriptState(template())
    source = [ChatMessage(role="user", content="original")]
    _record(state, source)
    if not provenance:
        del state._turns["s"][0]["history_fp"]
        del state._turns["s"][0]["history_has_images"]
    history = [
        ChatMessage(role="user", content="edited"),
        ChatMessage(role="assistant", content="reply"),
    ]
    if later_image:
        history.append(ChatMessage(role="user", content=[image_part()]))
    prompt = _image_prompt(state, history)
    assert [s["ids"] for s in prompt.segments if "ids" in s] == [[700, 701]]
    assert set(state._turns["s"]) == {0}


@pytest.mark.parametrize("provenance", [False, True])
def test_new_image_in_prefix_cannot_authorize_old_text_ids(provenance):
    state = OpenAITranscriptState(template())
    source = [ChatMessage(role="user", content="original")]
    _record(state, source)
    if not provenance:
        del state._turns["s"][0]["history_fp"]
        del state._turns["s"][0]["history_has_images"]
    history = [
        ChatMessage(role="user", content=[image_part()]),
        ChatMessage(role="assistant", content="reply"),
    ]
    prompt = _image_prompt(state, history)
    assert not any("ids" in s for s in prompt.segments)
    assert not state._turns["s"]


@pytest.mark.parametrize("reasoning", [None, "original", "edited", ""])
@pytest.mark.parametrize(
    "arguments,expected", [(' { "b": 2, "a": 1 } ', True), ('{"a":9,"b":2}', False)]
)
def test_image_tool_splice_keeps_semantic_arguments_and_explicit_reasoning_checks(
    reasoning, arguments, expected
):
    state = OpenAITranscriptState(template())
    source = [ChatMessage(role="user", content=[image_part()])]
    call = {
        "id": "original-id",
        "function": {"name": "look", "arguments": '{"a":1,"b":2}'},
    }
    _record(
        state, source, content=None, tool_calls=[call], reasoning_content="original"
    )
    echoed = {"id": "new-id", "function": {"name": "look", "arguments": arguments}}
    history = source + [
        ChatMessage(role="assistant", tool_calls=[echoed], reasoning_content=reasoning)
    ]
    prompt = _image_prompt(state, history)
    valid = expected and reasoning in (None, "original")
    assert [s["ids"] for s in prompt.segments if "ids" in s] == (
        [[700, 701]] if valid else []
    )
    assert sum("image" in s for s in prompt.segments) == 1
    assert bool(state._turns["s"]) == valid


@pytest.mark.parametrize("advertised", [None, False, True])
def test_readiness_image_negotiation(advertised):
    async def scenario():
        proc = _AsyncProc()
        ready = {
            "ready": True,
            "multiplexed": True,
            "max_images": 1,
            "max_image_bytes": 512,
            "max_image_pixels": 4,
        }
        if advertised is not None:
            ready["supports_images"] = advertised
        proc.stdout.feed_data((json.dumps(ready) + "\n").encode())

        async def spawn(*args, **kwargs):
            return proc

        with patch("asyncio.create_subprocess_exec", spawn):
            client = await spawn_multiplexed_worker(["fake"])
        try:
            assert client.supports_images is (advertised is True)
            assert client.image_limits == (
                ImageLimits(1, 512, 4) if advertised is True else None
            )
        finally:
            await client.close()
        assert proc.reaped

    _run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("supports_images", 1),
        ("max_images", None),
        ("max_image_bytes", True),
        ("max_image_pixels", 0),
        ("max_image_dimension", -1),
    ],
)
def test_invalid_image_readiness_reaps(field, value):
    async def scenario():
        proc = _AsyncProc()
        ready = {
            "ready": True,
            "multiplexed": True,
            "supports_images": True,
            "max_images": 1,
            "max_image_bytes": 512,
            "max_image_pixels": 4,
            field: value,
        }
        proc.stdout.feed_data((json.dumps(ready) + "\n").encode())

        async def spawn(*args, **kwargs):
            return proc

        with patch("asyncio.create_subprocess_exec", spawn), pytest.raises(WorkerError):
            await spawn_multiplexed_worker(["fake"])
        assert proc.reaped and proc.stdin.closed

    _run(scenario())


@pytest.mark.parametrize("extra", [0, 1])
def test_full_image_frame_limit_and_sync_enqueue(extra):
    async def scenario():
        async with _fake_client(image_limits=LIMITS) as (client, proc):
            image = validate_image("image/png", encoded(png()), LIMITS)
            request = {
                "op": "generate",
                "max_new_tokens": -1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 0,
                "stop": [],
                "prompt_segments": [{"text": '\u00e9\n"'}, image],
                "request_id": 1,
            }
            overhead = len((json.dumps(request, ensure_ascii=True) + "\n").encode())
            request["prompt_segments"][0]["text"] += "x" * (
                1024 * 1024 - overhead + extra
            )
            config = SimpleNamespace(prompt_segments=request["prompt_segments"])
            if extra:
                with pytest.raises(WorkerError) as error:
                    client.generate("", config)
                assert error.value.code == "invalid_argument"
                assert not client._requests and not client._writes
            else:
                stream = client.generate("", config)
                assert not asyncio.iscoroutine(stream)
                assert len(client._writes) == 1 and not proc.stdin.written
                frame = await proc.stdin.frames.get()
                assert frame["prompt_segments"] == config.prompt_segments
                assert len(proc.stdin.written[0]) == 1024 * 1024
                proc.send(
                    stream.request_id,
                    done=True,
                    generated_token_ids=[4, 5],
                    prompt_tokens=3,
                    prompt_positions=12,
                    prefilled_prompt_positions=12,
                    reused_prompt_positions=0,
                )
                assert [x async for x in stream] == []
                stats = await stream.wait()
                assert stats.generated_token_ids == [4, 5]
                assert stats.prompt_positions == 12 and stats.num_prompt_tokens == 3
            assert client.healthy

    _run(scenario())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "ignored_name", [None, "\ud800"], ids=["ordinary", "surrogate"]
)
def test_http_image_history_cold_replay_and_lifecycle(stream, ignored_name):
    async def scenario():
        worker = _NativeWorker(image_limits=LIMITS)
        runtime = SessionRuntime(worker)
        serving = ServingChat(runtime, template(), "test-model", max_context=1)
        # Expanded positions are checked by native preparation, never the text
        # tokenizer's count of the marker or a synthetic image patch token.
        app = build_app(serving, "test-model")

        body = {
            "session_id": "images",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        image_part(),
                        {"type": "text", "text": "after"},
                    ],
                    "name": ignored_name,
                }
            ],
            "stream": stream,
            "stream_options": {"include_usage": True},
        }
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                for turn in range(2):
                    pending = asyncio.create_task(
                        http.post(
                            "/v1/chat/completions",
                            content=json.dumps(body, ensure_ascii=True),
                            headers={"content-type": "application/json"},
                        )
                    )
                    _, request_id, config = await _call(worker)
                    segments = config.prompt_segments
                    assert [list(s) for s in segments] == (
                        [["text"], ["image"], ["text"], ["ids"], ["text"]]
                        if turn
                        else [["text"], ["image"], ["text"]]
                    )
                    assert segments[0]["text"].endswith("before")
                    assert segments[1] == validate_image(
                        "image/png", encoded(png()), LIMITS
                    )
                    assert segments[2]["text"].startswith("after")
                    if turn:
                        assert segments[3] == {"ids": [700, 701]}
                        assert "next" in segments[4]["text"]
                        assert "reply" not in "".join(
                            s.get("text", "") for s in segments
                        )
                    worker.finish(
                        request_id,
                        generated_token_ids=[700, 701],
                        prompt_positions=12,
                        reused_prompt_positions=0,
                        prefilled_prompt_positions=12,
                    )
                    response = await pending
                    assert response.status_code == 200
                    if stream:
                        assert "[DONE]" in response.text
                    else:
                        assert (
                            response.json()["choices"][0]["message"]["content"]
                            == "reply"
                        )
                        assert (
                            response.json()["usage"]["prompt_tokens_details"][
                                "cached_tokens"
                            ]
                            == 0
                        )
                    record = serving._transcript._turns["images"][turn]
                    assert record["ids"] == [700, 701]
                    assert record["history_has_images"]
                    assert record[
                        "history_fp"
                    ] == serving._transcript._history_fingerprint(
                        [ChatMessage(**message) for message in body["messages"]]
                    )
                    assert record["ids"] != template()._hf.encode("reply")
                    body["messages"] += [
                        {"role": "assistant", "content": "reply"},
                        {"role": "user", "content": "next"},
                    ]
                for method, path, operation in [
                    ("post", "/v1/sessions/images/reset", "reset"),
                    ("delete", "/v1/sessions/images", "close"),
                ]:
                    response = await getattr(http, method)(path)
                    assert response.status_code == 200
                    assert await _call(worker) == (operation, "images")
                assert (
                    not runtime._session_locks._entries
                    and not serving._transactions._entries
                )
                assert runtime._executor is None
        finally:
            await runtime.aclose_worker()
        assert worker._proc.reaped

    _run(scenario())


def test_unsupported_worker_rejects_images_before_generation(make_client):
    client, worker = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": [image_part()]}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_image"
    assert worker.captured_config is None


@pytest.mark.parametrize("header", [None, b"8", b"0008"])
@pytest.mark.parametrize("size", [8, 9])
def test_asgi_body_limit_before_parser_and_preserves_disconnect(header, size):
    async def scenario():
        messages = [
            {"type": "http.request", "body": b"x" * 4, "more_body": True},
            {"type": "http.request", "body": b"x" * (size - 4)},
            {"type": "http.disconnect"},
        ]
        responses, parsed = [], []

        async def receive():
            return messages.pop(0)

        async def send(message):
            responses.append(message)

        async def app(scope, receive, send):
            parsed.append(await receive())
            assert await receive() == {"type": "http.disconnect"}

        middleware = BoundedChatBody(app, max_bytes=8)
        await middleware(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "headers": [] if header is None else [(b"content-length", header)],
            },
            receive,
            send,
        )
        if size == 9:
            assert not parsed and responses[0]["status"] == 413
        else:
            assert parsed == [
                {"type": "http.request", "body": b"xxxxxxxx", "more_body": False}
            ]

    _run(scenario())


@pytest.mark.parametrize(
    "headers,status",
    [
        ([(b"content-length", b"9")], 413),
        ([(b"content-length", b"-1")], 400),
        ([(b"content-length", b"1,1")], 400),
        ([(b"content-length", b"1"), (b"content-length", b"1")], 400),
        ([(b"content-length", b"999999999999999999999999999")], 413),
    ],
)
def test_bad_or_oversized_content_length_rejected_without_receive(headers, status):
    async def scenario():
        responses = []

        async def forbidden(*args):
            raise AssertionError("must reject before body aggregation or JSON parsing")

        async def send(message):
            responses.append(message)

        await BoundedChatBody(forbidden, max_bytes=8)(
            {"type": "http", "path": "/v1/chat/completions", "headers": headers},
            forbidden,
            send,
        )
        assert responses[0]["status"] == status

    _run(scenario())
