# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

pytest.importorskip("pydantic", reason="requires llm_server serving dependencies")
pytest.importorskip("fastapi", reason="requires llm_server serving dependencies")
pytest.importorskip("httpx", reason="requires llm_server serving dependencies")
pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")

from executorch.examples.llm_server.python.server import build_app  # noqa: E402
from executorch.examples.models.muse_glimmer.serving.stream_parser import (  # noqa: E402
    MuseGlimmerToolReasoningParser,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers.atem import (  # noqa: E402
    AtemToolCallDetector,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import (  # noqa: E402
    ToolOutputError,
    ToolSchema,
)
from executorch.examples.models.muse_glimmer.tests.test_atem_tool_parser import (  # noqa: E402
    _atem,
)
from executorch.examples.models.muse_glimmer.tests.test_serve import (  # noqa: E402
    _FakeRuntime,
    _glimmer_serving,
)
from fastapi.testclient import TestClient  # noqa: E402


def client_for(raw, split=1, template=None):
    serving, _ = _glimmer_serving([], template=template)

    class ChunkedRuntime(_FakeRuntime):
        async def generate_stream(self, *args, **kwargs):
            async for text in super().generate_stream(*args, **kwargs):
                for i in range(0, len(text), split):
                    yield text[i : i + split]

    serving._runtime = ChunkedRuntime([{"text": raw, "completion_tokens": 50}])
    serving._tool_detector_cls = AtemToolCallDetector
    serving._tool_reasoning_parser_factory = MuseGlimmerToolReasoningParser
    return TestClient(build_app(serving, "test-model")), serving


def request(schema, stream=False, strict=False):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "edit"}],
        "stream": stream,
        "tools": [
            {
                "type": "function",
                "function": {"name": "edit", "strict": strict, "parameters": schema},
            }
        ],
    }


def response_message(response, stream):
    if not stream:
        choice = response.json()["choices"][0]
        return choice["message"], choice["finish_reason"]
    message = {"content": "", "reasoning_content": "", "tool_calls": []}
    finish = None
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        assert "error" not in chunk, chunk
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason", finish)
            for key in message:
                if key in choice["delta"]:
                    message[key] += choice["delta"][key]
    return message, finish


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 7, 10000])
def test_parameter_mention_in_reasoning_does_not_mask_real_call(stream, split):
    thought = 'The format has <atem:parameter name="q"> here.'
    raw = (
        " to=self<|message|>"
        + thought
        + "<|eom|><|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("q", "hello")])
    )
    client, _ = client_for(raw, split)
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert message["reasoning_content"] == thought
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "q": "hello"
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefix", ["", _atem("edit.edit", [("q", "complete")]) + "\n"])
def test_length_truncation_never_emits_incomplete_call_set(stream, prefix):
    raw = (
        prefix
        + '<atem:function_calls><atem:invoke name="edit.edit"><atem:parameter name="q">unfinished'
    )
    client, serving = client_for(raw)
    serving._runtime._outputs[0]["finish_reason"] = "length"
    serving._runtime._outputs[0]["gen_ids"] = [1, 2, 3]
    body = request({"properties": {"q": {"type": "string"}}}, stream)
    body["session_id"] = "truncated"
    response = client.post(
        "/v1/chat/completions",
        json=body,
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "length"
    assert not message.get("tool_calls")
    assert "atem:" not in (message.get("content") or "")

    assert serving._transcript._turns["truncated"][0]["ids"] is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "The format starts with `<atem:function_calls>` followed by an invocation.",
        "Example:\n```xml\n" + _atem("edit.edit", [("q", "example")]) + "\n```",
        _atem("edit.edit", [("q", "user-directed example")]),
    ],
)
def test_user_directed_atem_examples_remain_text(stream, text):
    client, _ = client_for(" to=user<|message|>" + text)
    response = client.post("/v1/chat/completions", json=request({}, stream))
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert not message.get("tool_calls")
    assert message["content"] == text


@pytest.mark.parametrize(
    "dialect",
    [
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
def test_schema_dialects_and_root_identifier_use_local_refs(dialect):
    schema = {
        "$schema": dialect,
        "$id": "https://example.invalid/tool",  # @lint-ignore
        "definitions": {"text": {"type": "string"}},
        "properties": {"q": {"$ref": "#/definitions/text"}},
    }
    client, _ = client_for(_atem("edit.edit", [("q", "123")]))
    response = client.post("/v1/chat/completions", json=request(schema))
    assert response.status_code == 200
    assert json.loads(
        response.json()["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ]
    ) == {"q": "123"}


def test_draft7_tuple_items_are_validated_using_draft7_rules():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": {
            "q": {
                "type": "array",
                "items": [{"type": "integer"}, {"type": "string"}],
                "additionalItems": False,
            }
        },
    }
    tool = ToolSchema(schema)
    tool.validate("edit", {"q": [1, "text"]})
    with pytest.raises(ToolOutputError):
        tool.validate("edit", {"q": ["wrong", "text"]})


def test_unused_schemas_do_not_reject_tool_choice_none():
    client, _ = client_for(" to=user<|message|>Hello")
    body = request(
        {
            "$schema": "https://example.invalid/unsupported-dialect",  # @lint-ignore
        }
    )
    body["tool_choice"] = "none"
    assert client.post("/v1/chat/completions", json=body).status_code == 200


def test_excessive_schema_alternatives_fail_before_generation():
    client, serving = client_for("unused")
    schema = {"allOf": [{"anyOf": [{}, {}]} for _ in range(10)]}
    response = client.post("/v1/chat/completions", json=request(schema))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert not serving._runtime.prompts


def test_deep_parameter_json_returns_tool_error():
    with pytest.raises(ToolOutputError, match="nested too deeply"):
        ToolSchema({"properties": {"q": {"type": "array"}}}).coerce(
            "q", "[" * 10000 + "0" + "]" * 10000
        )


def test_schema_is_compiled_once_for_multiple_invocations(monkeypatch):
    from executorch.examples.models.muse_glimmer.serving.tool_parsers import schema

    check = schema.Draft202012Validator.check_schema
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return check(*args, **kwargs)

    monkeypatch.setattr(schema.Draft202012Validator, "check_schema", counted)
    client, _ = client_for("\n".join([_atem("edit.edit", [("q", "hello")])] * 3))
    assert (
        client.post(
            "/v1/chat/completions",
            json=request({"properties": {"q": {"type": "string"}}}),
        ).status_code
        == 200
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "reasoning", [{}, {"effort": None}, {"enabled": True}, {"exclude": False}]
)
def test_noop_reasoning_options_are_accepted(reasoning):
    client, _ = client_for(" to=user<|message|>Hello")
    body = request({})
    body["reasoning"] = reasoning
    assert client.post("/v1/chat/completions", json=body).status_code == 200


def test_bundled_template_accepts_tokenizer_without_embedded_template(monkeypatch):
    from executorch.examples.models.muse_glimmer.serving.chat_template import (
        MuseGlimmerChatTemplate,
    )
    from transformers import AutoTokenizer

    class Tokenizer:
        chat_template = None

    tokenizer = Tokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    template = MuseGlimmerChatTemplate("unused")
    assert template._hf is tokenizer
    assert "<atem:function_calls>" in tokenizer.chat_template


def test_bundled_prompt_golden_without_hf_tokenizer():
    from pathlib import Path

    from executorch.examples.models.muse_glimmer.serving import chat_template
    from jinja2 import Environment

    template = Environment().from_string(
        Path(chat_template.__file__).with_suffix(".jinja").read_text()
    )
    prompt = template.render(
        messages=[
            {"role": "system", "content": "Be precise."},
            {"role": "developer", "content": "Keep details."},
            {"role": "user", "content": "Edit"},
            {
                "role": "assistant",
                "reasoning_content": "Think",
                "tool_calls": [
                    {
                        "id": "x",
                        "function": {"name": "edit.edit", "arguments": {"q": "hello"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "x", "content": "OK"},
        ],
        bos_token="<s>",
        add_generation_prompt=True,
    )
    assert prompt == (
        '<s><|start|>system<|message|>Be precise.\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|>'
        "<|start|>system<|message|>Keep details.<|eot|>"
        "<|start|>user<|message|>Edit<|eot|>"
        "<|start|>assistant to=self<|message|>Think<|eom|>"
        "<|start|>assistant to=edit.edit<|message|>"
        '<atem:function_calls>\n<atem:invoke name="edit.edit">\n<atem:parameter name="q">hello</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>'
        '<|start|>tool edit.edit<|message|><tool_output name="edit.edit">\nOK\n</tool_output><|eot|><|start|>assistant'
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 7, 10000])
def test_harmony_literals_survive_complete_tool_response(stream, split):
    values = [
        "  before to=self<|message|>after  ",
        "<|start|>assistant to=user<|message|>literal",
    ]
    raw = " to=self<|message|>Thinking.<|eom|>" + "<|eom|>".join(
        "<|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("content", value)])
        for value in values
    )
    client, _ = client_for(raw, split)
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"content": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        calls = [
            call
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for call in choice["delta"].get("tool_calls", [])
        ]
        reasoning = "".join(
            choice["delta"].get("reasoning_content", "")
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    else:
        message = response.json()["choices"][0]["message"]
        calls, reasoning = message["tool_calls"], message["reasoning_content"]
    assert [
        json.loads(call["function"]["arguments"])["content"] for call in calls
    ] == values
    assert reasoning == "Thinking."


@pytest.mark.parametrize(
    "property_schema",
    [
        {"anyOf": [{"type": "string"}, {"type": "null"}]},
        {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        {"$ref": "#/$defs/string"},
    ],
)
def test_union_and_local_ref_preserve_numeric_strings(property_schema):
    schema = {
        "$defs": {"string": {"type": "string"}},
        "properties": {"content": property_schema},
    }
    calls = (
        AtemToolCallDetector()
        .detect_and_parse(
            _atem("edit", [("content", "123")]), {"edit": ToolSchema(schema)}
        )
        .calls
    )
    assert json.loads(calls[0].arguments) == {"content": "123"}


@pytest.mark.parametrize(
    "schema,args",
    [
        (
            {"properties": {"content": {"type": ["integer", "null"]}}},
            [("content", "1.5")],
        ),
        ({"properties": {"content": {"type": "array"}}}, [("content", "123")]),
        ({"required": ["content"]}, []),
        ({"properties": {}, "additionalProperties": False}, [("content", "text")]),
        ({"properties": {"content": {"enum": ["ok"]}}}, [("content", "bad")]),
        (
            {
                "properties": {
                    "content": {"type": "array", "items": {"type": "integer"}}
                }
            },
            [("content", '[1,"bad"]')],
        ),
        (
            {"properties": {"content": {"type": "object"}}},
            [("content", '{"nested":NaN}')],
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_invalid_tool_output_is_an_error_not_a_successful_call(
    schema, args, stream, strict
):
    raw = _atem("edit.edit", args)
    client, serving = client_for(raw)
    response = client.post("/v1/chat/completions", json=request(schema, stream, strict))
    if stream:
        assert response.status_code == 200
        assert '"code": "invalid_tool_output"' in response.text
        assert '"finish_reason"' not in response.text
        assert '"tool_calls"' not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_tool_output"
    assert not serving._transcript._turns


@pytest.mark.parametrize(
    "raw",
    [
        '<atem:function_calls><atem:invoke name="edit.edit"><atem:parameter name="content">unfinished',
        _atem("unknown", [("content", "123")]),
    ],
)
def test_malformed_and_unknown_calls_are_explicit_errors(raw):
    client, _ = client_for(raw)
    response = client.post("/v1/chat/completions", json=request({}))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_tool_output"


@pytest.mark.parametrize(
    "schema",
    [
        {
            "$schema": "https://example.invalid/unsupported-dialect",  # @lint-ignore
        },
        {
            "properties": {
                "content": {
                    "$ref": "https://example.invalid/schema",  # @lint-ignore
                }
            }
        },
        {"properties": {"content": {"$ref": "#/$defs/missing"}}},
        {"properties": {"content": {"type": "not-a-type"}}},
    ],
)
def test_invalid_schemas_fail_before_generation(schema):
    client, serving = client_for("unused")
    response = client.post("/v1/chat/completions", json=request(schema))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert not serving._runtime.prompts


def test_strict_summary_tool_returns_schema_valid_arguments():
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
        "additionalProperties": False,
    }
    client, _ = client_for(_atem("edit.edit", [("content", "retained facts")]))
    response = client.post("/v1/chat/completions", json=request(schema, strict=True))
    assert response.status_code == 200
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"content": "retained facts"}


def test_recursive_schema_validates_nested_values_without_remote_resolution():
    schema = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "next": {"$ref": "#/$defs/node"},
                },
            }
        },
        "properties": {"content": {"$ref": "#/$defs/node"}},
    }
    tool = ToolSchema(schema)
    args = {"content": tool.coerce("content", '{"value":"a","next":{"value":"b"}}')}
    tool.validate("edit", args)
    with pytest.raises(ToolOutputError):
        tool.validate("edit", {"content": {"next": {"value": 1}}})


@pytest.mark.parametrize(
    "schema",
    [
        {"additionalProperties": {"type": "string"}},
        {"patternProperties": {"^content$": {"type": "string"}}},
        {
            "allOf": [
                {"properties": {"content": {"type": "string"}}},
                {"required": ["content"]},
            ]
        },
    ],
)
def test_composed_and_map_schemas_preserve_strings(schema):
    tool = ToolSchema(schema)
    args = {"content": tool.coerce("content", "123")}
    tool.validate("edit", args)
    assert args == {"content": "123"}


@pytest.mark.parametrize("stream", [False, True])
def test_tool_result_names_survive_token_replay(stream):
    import asyncio
    from pathlib import Path

    from executorch.examples.llm_server.python.protocol import (
        ChatCompletionRequest,
        ChatMessage,
    )
    from executorch.examples.models.muse_glimmer.serving import chat_template
    from executorch.examples.models.muse_glimmer.tests.test_serve import _create_message
    from jinja2 import Environment

    jinja = Environment().from_string(
        Path(chat_template.__file__).with_suffix(".jinja").read_text()
    )

    class Tokenizer:
        all_special_tokens = []
        eos_token = None

        def apply_chat_template(self, messages, **kwargs):
            return jinja.render(messages=messages, bos_token="", **kwargs)

    template = chat_template.MuseGlimmerChatTemplate(
        allow_fallback=True, assistant_header="<|start|>assistant"
    )
    template._hf = Tokenizer()
    serving, runtime = _glimmer_serving(
        [
            {"text": _atem("read_file.read_file", []), "gen_ids": [777]},
            {"text": " to=user<|message|>Done"},
        ],
        template=template,
    )
    serving._tool_detector_cls = AtemToolCallDetector
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    async def scenario():
        user = ChatMessage(role="user", content="Read the file")
        first = await _create_message(
            serving,
            ChatCompletionRequest(
                messages=[user], tools=tools, session_id="s", stream=stream
            ),
        )
        call_id = first["tool_calls"][0]["id"]
        await _create_message(
            serving,
            ChatCompletionRequest(
                messages=[
                    user,
                    ChatMessage(**first),
                    ChatMessage(
                        role="tool", tool_call_id=call_id, content="FILE_RESULT"
                    ),
                ],
                tools=tools,
                session_id="s",
                stream=stream,
            ),
        )
        prompt = runtime.prompts[1]
        assert any(segment.get("ids") == [777] for segment in prompt.segments or [])
        text = "".join(segment.get("text", "") for segment in prompt.segments)
        assert (
            '<|start|>tool read_file.read_file<|message|><tool_output name="read_file.read_file">'
            in text
        )
        assert f"<|start|>tool {call_id}" not in text

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("keyword", ["oneOf", "anyOf", "if", "dependentSchemas"])
def test_schema_coercion_respects_whole_object_constraints(stream, keyword):
    number = {
        "properties": {"kind": {"const": "number"}, "content": {"type": "integer"}}
    }
    string = {"properties": {"kind": {"const": "text"}, "content": {"type": "string"}}}
    if keyword in ("oneOf", "anyOf"):
        schema = {keyword: [number, string]}
        kind, expected = "number", 123
    elif keyword == "if":
        schema = {
            "if": {"properties": {"kind": {"const": "text"}}},
            "then": string,
            "else": number,
        }
        kind, expected = "text", "123"
    else:
        schema = {"dependentSchemas": {"kind": string}}
        kind, expected = "text", "123"
    raw = _atem("edit.edit", [("kind", kind), ("content", "123")])
    client, _ = client_for(raw)
    response = client.post(
        "/v1/chat/completions", json=request(schema, stream, strict=True)
    )
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert not any("error" in chunk for chunk in chunks)
        calls = [
            call
            for chunk in chunks
            for choice in chunk.get("choices", [])
            for call in choice["delta"].get("tool_calls", [])
        ]
    else:
        calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "kind": kind,
        "content": expected,
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("constraint", [{"enum": ["ok"]}, {"pattern": "^ok$"}])
@pytest.mark.parametrize("conditional", [False, True])
def test_invalid_literal_string_is_not_unquoted_to_satisfy_schema(
    stream, constraint, conditional
):
    client, _ = client_for(_atem("edit.edit", [("content", '"ok"')]))
    schema = {"properties": {"content": {"type": "string", **constraint}}}
    if conditional:
        schema = {"if": {"required": ["content"]}, "then": schema}
    response = client.post(
        "/v1/chat/completions", json=request(schema, stream, strict=True)
    )
    if stream:
        assert '"invalid_tool_output"' in response.text
        assert '"tool_calls"' not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_tool_output"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 7, 10000])
@pytest.mark.parametrize("example_first", [False, True])
@pytest.mark.parametrize("fence", ["none", "balanced", "unclosed"])
def test_mixed_recipient_examples_never_become_calls(
    stream, split, example_first, fence
):
    example = _atem("edit.edit", [("q", "EXAMPLE")])
    if fence != "none":
        example = "```xml\n" + example
        if fence == "balanced":
            example += "\n```"
    user = "<|start|>assistant to=user<|message|>Example:\n" + example + "<|eom|>"
    real = (
        "<|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("q", "REAL")])
        + "<|eom|>"
    )
    raw = user + real if example_first else real + user
    client, _ = client_for(raw, split)
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    if example_first and fence == "unclosed":
        assert finish == "stop"
        assert not message.get("tool_calls")
        assert "REAL" in message["content"]
        return
    assert finish == "tool_calls"
    assert [
        json.loads(call["function"]["arguments"]) for call in message["tool_calls"]
    ] == [{"q": "REAL"}]
    assert "EXAMPLE" in message["content"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("split", [1, 7, 10000])
def test_request_stop_inside_parameter_suppresses_entire_call_set(
    stream, reasoning, split
):
    raw = (
        _atem("edit.edit", [("q", "complete")])
        + "\n"
        + _atem("edit.edit", [("q", "before\n\nafter")])
    )
    client, serving = client_for(raw, split)
    serving._runtime._outputs[0]["gen_ids"] = [1, 2, 3]
    body = request({"properties": {"q": {"type": "string"}}}, stream)
    body.update(stop="\n\n", session_id="stopped", reasoning={"exclude": not reasoning})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert not message.get("tool_calls")
    assert "atem:" not in (message.get("content") or "")
    assert serving._transcript._turns["stopped"][0]["ids"] is None


def _reference_dag(depth):
    definitions = {"n0": {"properties": {"q": {"type": "string"}}}}
    for index in range(1, depth + 1):
        definitions[f"n{index}"] = {"allOf": [{"$ref": f"#/$defs/n{index - 1}"}] * 2}
    return {"$defs": definitions, "$ref": f"#/$defs/n{depth}"}


def test_exponential_reference_expansion_rejected_before_coercion_or_generation(
    monkeypatch,
):
    def unexpected(*args):
        pytest.fail(
            "reference expansion must be bounded before recursive coercion analysis"
        )

    monkeypatch.setattr(ToolSchema, "_variant_count", unexpected)
    client, serving = client_for(_atem("edit.edit", [("q", "hello")]))
    response = client.post("/v1/chat/completions", json=request(_reference_dag(40)))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert "expanded schema nodes" in response.json()["error"]["message"]
    assert not serving._runtime.prompts


def test_bounded_shared_reference_schema_still_validates():
    schema = ToolSchema(_reference_dag(4))
    assert schema.coerce_arguments("edit", {"q": "123"}) == {"q": "123"}


@pytest.mark.parametrize("stream", [False, True])
def test_excessive_coercion_work_fails_before_generation(stream):
    schema = {
        "allOf": [
            {
                "anyOf": [
                    {"properties": {f"q{i}": {"const": value}}} for value in ("a", "b")
                ]
            }
            for i in range(9)
        ]
    }
    client, serving = client_for(_atem("edit.edit", [("q0", "a")]))
    response = client.post("/v1/chat/completions", json=request(schema, stream))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert "coercion work" in response.json()["error"]["message"]
    assert not serving._runtime.prompts


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cutoff", ["length", "request-stop", "native-stop"])
@pytest.mark.parametrize(
    "block",
    [
        '<atem:function_calls><atem:invoke name="edit.edit"></atem:function_calls>',
        "<atem:function_calls></atem:function_calls>",
        _atem("edit.edit", [("q", "ok")]).replace(
            "</atem:function_calls>", '<atem:parameter name="q"></atem:function_calls>'
        ),
    ],
)
def test_closed_malformed_tool_blocks_remain_errors_after_cutoff(stream, cutoff, block):
    client, serving = client_for(block + "\nTrailing prose STOP tail")
    body = request({"properties": {"q": {"type": "string"}}}, stream)
    if cutoff == "length":
        serving._runtime._outputs[0]["finish_reason"] = "length"
    else:
        body["stop"] = "STOP"
        if cutoff == "native-stop":
            serving._runtime._outputs[0].update(
                text=block + "\nTrailing prose ", matched_stop="STOP"
            )
    response = client.post("/v1/chat/completions", json=body)
    if stream:
        assert '"invalid_tool_output"' in response.text
        assert '"tool_calls"' not in response.text
        assert '"finish_reason"' not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_tool_output"


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("metadata", ["Reasoning strength", "Reasoning effort"])
@pytest.mark.parametrize("explicit", [False, True])
def test_developer_and_system_metadata_share_one_reasoning_control(
    role, metadata, explicit
):
    from pathlib import Path

    from executorch.examples.models.muse_glimmer.serving import chat_template
    from jinja2 import Environment

    template = Environment().from_string(
        Path(chat_template.__file__).with_suffix(".jinja").read_text()
    )
    prose = "Keep your reasoning effort brief."
    kwargs = {"reasoning_strength": "high"} if explicit else {}
    prompt = template.render(
        messages=[
            {"role": role, "content": prose + "\n" + metadata + ": low."},
            {"role": "user", "content": "hi"},
        ],
        tools=[{"function": {"name": "edit.edit"}}],
        add_generation_prompt=True,
        bos_token="<s>",
        **kwargs,
    )
    assert prompt.count("<|start|>system<|message|>") == 1
    assert prose in prompt
    assert prompt.count("Reasoning strength:") == 1
    assert "Reasoning strength: " + ("high." if explicit else "low.") in prompt
    assert "Reasoning effort:" not in prompt
    assert prompt.count("// Function schemas") == 1
    assert prompt.count("# Valid recipients:") == 1


@pytest.mark.parametrize(
    "kind,raw,expected",
    [
        ("integer", "+5", 5),
        ("integer", "007", 7),
        ("number", ".5", 0.5),
        ("number", "5.", 5.0),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_declared_numeric_spellings_remain_compatible(kind, raw, expected, stream):
    client, _ = client_for(_atem("edit.edit", [("q", raw)]))
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": kind}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "q": expected
    }
    assert ToolSchema({"properties": {"q": {"type": "string"}}}).coerce_arguments(
        "edit", {"q": raw}
    ) == {"q": raw}


@pytest.mark.parametrize("stream", [False, True])
def test_null_parameters_are_normalized_before_schema_compilation(stream):
    client, _ = client_for(_atem("edit.edit", []))
    response = client.post("/v1/chat/completions", json=request(None, stream))
    assert response.status_code == 200
    message, _ = response_message(response, stream)
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {}


@pytest.mark.parametrize(
    "system",
    [
        "Explain how reasoning strength changes answers.",
        "Reasoning strength: high.",
        "Use high reasoning effort for proofs",
    ],
)
def test_explicit_reasoning_effort_survives_system_prose(system):
    from pathlib import Path

    from executorch.examples.models.muse_glimmer.serving import chat_template
    from jinja2 import Environment

    template = Environment().from_string(
        Path(chat_template.__file__).with_suffix(".jinja").read_text()
    )
    prompt = template.render(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "hi"},
        ],
        reasoning_strength="low",
        add_generation_prompt=True,
        bos_token="<s>",
    )
    assert "Reasoning strength: low." in prompt
    assert "Reasoning strength: high." not in prompt
    if system.startswith("Explain"):
        assert system in prompt


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("matched_stop", [None, "", "<|eot|>", "\n\n"])
def test_native_stop_metadata_distinguishes_truncation_from_malformed_eos(
    stream, matched_stop
):
    raw = '<atem:function_calls><atem:invoke name="edit.edit"><atem:parameter name="q">unfinished'
    client, serving = client_for(raw)
    serving._runtime._outputs[0]["matched_stop"] = matched_stop
    body = request({"properties": {"q": {"type": "string"}}}, stream)
    body["stop"] = "\n\n"
    response = client.post("/v1/chat/completions", json=body)
    if matched_stop in (None, "\n\n"):
        assert response.status_code == 200
        message, finish = response_message(response, stream)
        assert finish == "stop"
        assert not message.get("tool_calls")
    elif stream:
        assert '"invalid_tool_output"' in response.text
    else:
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_tool_output"


def test_compiled_schemas_are_not_copied_with_request_normalization(monkeypatch):
    def unexpected(*args):
        pytest.fail("compiled schemas must not be deep-copied with request data")

    monkeypatch.setattr(ToolSchema, "__deepcopy__", unexpected, raising=False)
    client, _ = client_for(_atem("edit.edit", [("q", "hello")]))
    response = client.post(
        "/v1/chat/completions", json=request({"properties": {"q": {"type": "string"}}})
    )
    assert response.status_code == 200


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"q": {"type": "string", "required": True}}},
        {"$schema": "http://json-schema.org/draft-04/schema#"},
        {"$schema": "http://json-schema.org/draft-06/schema#"},
        {"$schema": "https://example.invalid/unknown-draft"},  # @lint-ignore
    ],
)
def test_invalid_active_schemas_fail_before_generation_regardless_of_strict(
    schema, strict
):
    client, serving = client_for(" to=user<|message|>No tool needed")
    response = client.post("/v1/chat/completions", json=request(schema, strict=strict))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert not serving._runtime.prompts


def test_packaged_chat_template_starts_outside_source_tree(tmp_path, monkeypatch):
    import importlib.util
    import tomllib
    from pathlib import Path
    from types import SimpleNamespace

    from setuptools import Distribution
    from setuptools.command.build_py import build_py

    root = Path(__file__).resolve().parents[4]
    config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]
    package = "executorch.examples.models.muse_glimmer.serving"
    monkeypatch.chdir(tmp_path)
    distribution = Distribution(
        {
            "packages": [package],
            "package_dir": {
                name: str(root / path) for name, path in config["package-dir"].items()
            },
            "package_data": config["package-data"],
        }
    )
    distribution.script_name = "setup.py"
    build = build_py(distribution)
    build.build_lib = str(tmp_path / "installed")
    build.ensure_finalized()
    build.run()
    installed = Path(build.build_lib) / package.replace(".", "/") / "chat_template.py"
    spec = importlib.util.spec_from_file_location(
        package + "._installed_template", installed
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = SimpleNamespace(chat_template=None)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer
    )
    module.MuseGlimmerChatTemplate("unused")
    assert "<atem:function_calls>" in tokenizer.chat_template


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_bounded_alternatives_report_argument_errors_not_work_limits(stream, valid):
    schema = {
        "allOf": [
            {"anyOf": [{"properties": {f"q{i}": {"const": v}}} for v in ("a", "b")]}
            for i in range(4)
        ]
    }
    raw = _atem("edit.edit", [(f"q{i}", "b" if valid else "x") for i in range(4)])
    client, _ = client_for(raw)
    response = client.post("/v1/chat/completions", json=request(schema, stream))
    if valid:
        assert response.status_code == 200
        message, finish = response_message(response, stream)
        assert finish == "tool_calls"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            f"q{i}": "b" for i in range(4)
        }
    else:
        assert "arguments violate" in response.text
        assert "budget" not in response.text
        assert '"tool_calls"' not in response.text
        assert response.status_code == (200 if stream else 502)


def test_nested_argument_validation_is_charged_before_validator_runs(monkeypatch):
    schema = ToolSchema(
        {"properties": {"q": {"type": "array", "items": {"type": "string"}}}}
    )

    def unexpected(*args, **kwargs):
        pytest.fail("oversized nested output must exhaust the budget before validation")

    monkeypatch.setattr(type(schema.validator), "iter_errors", unexpected)
    with pytest.raises(ToolOutputError, match="traversal budget"):
        schema.validate("edit", {"q": ["ok"] * 70000})


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separate_blocks", [False, True])
def test_tool_output_budget_is_shared_across_invocations(stream, separate_blocks):
    schema = {"properties": {"q": {"type": "array", "items": {"type": "string"}}}}
    one = _atem("edit.edit", [("q", json.dumps(["ok"] * 9000))])
    client, _ = client_for(one, split=10000)
    response = client.post("/v1/chat/completions", json=request(schema, stream))
    assert response.status_code == 200
    assert response_message(response, stream)[1] == "tool_calls"
    raw = "\n".join([one] * 4)
    if not separate_blocks:
        raw = raw.replace("</atem:function_calls>\n<atem:function_calls>", "")
    client, _ = client_for(raw, split=10000)
    response = client.post("/v1/chat/completions", json=request(schema, stream))
    assert response.status_code == (200 if stream else 502)
    assert "traversal budget" in response.text
    assert '"invalid_tool_output"' in response.text
    assert '"tool_calls"' not in response.text


def test_schema_work_estimate_does_not_execute_parameter_patterns(monkeypatch):
    from executorch.examples.models.muse_glimmer.serving.tool_parsers import schema

    def unexpected(*args, **kwargs):
        pytest.fail(
            "preflight must count every pattern without executing regex searches"
        )

    monkeypatch.setattr(schema.re, "search", unexpected)
    ToolSchema(
        {
            "properties": {"q": {"type": "string"}},
            "patternProperties": {"^q$": {"type": "string"}},
        }
    )
