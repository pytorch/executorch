# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.models.muse_glimmer.serving.stream_parser import (
    MuseGlimmerToolReasoningParser,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers.atem import (
    AtemToolCallDetector,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import (
    ToolOutputError,
    ToolSchema,
)
from executorch.examples.models.muse_glimmer.tests.test_atem_tool_parser import _atem
from executorch.examples.models.muse_glimmer.tests.test_serve import (
    _FakeRuntime,
    _glimmer_serving,
)
from fastapi.testclient import TestClient


def client_for(raw, split=1):
    serving, _ = _glimmer_serving([])

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
        .detect_and_parse(_atem("edit", [("content", "123")]), {"edit": schema})
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
        {"$schema": "http://json-schema.org/draft-07/schema#"},
        {"properties": {"content": {"$ref": "https://example.invalid/schema"}}},
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
