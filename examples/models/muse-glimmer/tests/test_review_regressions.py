# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Acceptance cases from the serving review, using the actual HTTP adapter."""

import hashlib
import json
import subprocess
import sys

import pytest

pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")

from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import (  # noqa: E402
    ToolOutputBudget,
    ToolOutputError,
    ToolSchema,
)
from executorch.examples.models.muse_glimmer.tests.test_atem_tool_parser import (  # noqa: E402
    _atem,
)
from executorch.examples.models.muse_glimmer.tests.test_migration_contracts import (  # noqa: E402
    client_for,
    request,
    response_message,
)
from pydantic import create_model  # noqa: E402


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("shape", ["flat", "nested", "pydantic"])
def test_realistic_tool_schemas_are_accepted(shape, stream):
    if shape == "flat":
        schema = {"properties": {f"q{i}": {"type": "string"} for i in range(100)}}
        arguments = {"q99": "ok"}
    elif shape == "nested":
        schema = {
            "properties": {
                f"obj{i}": {
                    "type": "object",
                    "properties": {f"q{i}_{j}": {"type": "string"} for j in range(8)},
                }
                for i in range(8)
            }
        }
        arguments = {"obj7": {"q7_7": "ok"}}
    else:
        schema = create_model(
            "Tool", **{f"q{i}": (str | None, None) for i in range(50)}
        ).model_json_schema()
        arguments = {"q49": "ok"}
    raw = _atem(
        "edit.edit",
        [(k, v if isinstance(v, str) else json.dumps(v)) for k, v in arguments.items()],
    )
    client, _ = client_for(raw)
    response = client.post("/v1/chat/completions", json=request(schema, stream))
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize("stream", [False, True])
def test_eight_valid_calls_with_composed_properties(stream):
    schema = {
        "properties": {
            f"q{i}": {
                "allOf": [
                    {"type": "string"},
                    {"minLength": 0},
                    {"maxLength": 8},
                    {"pattern": ".*"},
                ]
            }
            for i in range(20)
        }
    }
    arguments = {f"q{i}": "ok" for i in range(20)}
    client, _ = client_for("\n".join([_atem("edit.edit", list(arguments.items()))] * 8))
    response = client.post("/v1/chat/completions", json=request(schema, stream))
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert len(message["tool_calls"]) == 8
    assert all(
        json.loads(call["function"]["arguments"]) == arguments
        for call in message["tool_calls"]
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("terminator", ["", "<|eom|>", "<|eot|>"])
def test_bare_invocation_is_complete(stream, terminator):
    raw = (
        ' to=edit.edit<|message|><atem:invoke name="edit.edit"><atem:parameter name="q">ok</atem:parameter></atem:invoke>'
        + terminator
    )
    client, _ = client_for(raw)
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"q": "ok"}
    assert not message.get("content")


@pytest.mark.parametrize(
    "draft",
    [
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
def test_all_regex_validation_paths_have_a_deadline(draft):
    code = """
from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import ToolSchema, ToolOutputError
import sys
pattern = "^(a|a)*$"
value = "a" * 100 + "b"
for schema, args in [
    ({"properties": {"q": {"pattern": pattern}}}, {"q": value}),
    ({"patternProperties": {pattern: True}}, {value: "ok"}),
    ({"patternProperties": {pattern: True}, "additionalProperties": False}, {value: "ok"}),
    ({"allOf": [{"patternProperties": {pattern: True}}], "unevaluatedProperties": False}, {value: "ok"}),
]:
    tool = ToolSchema({"$schema": sys.argv[1], **schema})
    try:
        tool.validate("edit", args)
    except ToolOutputError as error:
        assert "regex validation" in str(error), error
    else:
        raise AssertionError("regex deadline was not enforced")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, draft], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_scalar_projection_uses_regex_deadline():
    budget = ToolOutputBudget()
    budget.regex_seconds = 0
    schema = ToolSchema({"properties": {"q": {"type": "string", "pattern": "x"}}})
    with pytest.raises(ToolOutputError, match="regex validation"):
        schema.coerce_arguments("edit", {"q": "x"}, budget=budget)


@pytest.mark.parametrize("stream", [False, True])
def test_overlapping_native_stop_is_request_cutoff(stream):
    client, serving = client_for(
        '<atem:function_calls><atem:invoke name="edit.edit">unfinished'
    )
    serving._runtime._outputs[0]["matched_stop"] = "<|eom|>"
    body = request({}, stream)
    body["stop"] = "<|"
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert not message.get("tool_calls")


def test_schema_work_is_bounded_across_all_tools():
    client, serving = client_for("ok")
    body = request({})
    schema = {"properties": {f"q{i}": {"type": "string"} for i in range(100)}}
    body["tools"] = [
        {"type": "function", "function": {"name": f"f{i}", "parameters": schema}}
        for i in range(100)
    ]
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert not serving._runtime.prompts


@pytest.mark.parametrize("stream", [False, True])
def test_closed_malformed_bare_invocation_is_not_hidden_by_cutoff(stream):
    raw = '<atem:invoke bad="edit.edit"></atem:invoke>'
    client, serving = client_for(raw)
    serving._runtime._outputs[0]["finish_reason"] = "length"
    response = client.post("/v1/chat/completions", json=request({}, stream))
    assert response.status_code == (200 if stream else 502)
    assert '"invalid_tool_output"' in response.text
    assert '"tool_calls"' not in response.text


def test_detector_subclass_receives_cutoff_context():
    from executorch.examples.models.muse_glimmer.serving.tool_parsers.atem import (
        AtemToolCallDetector,
    )

    class Detector(AtemToolCallDetector):
        pass

    client, serving = client_for('<atem:function_calls><atem:invoke name="edit.edit">')
    serving._tool_detector_cls = Detector
    serving._runtime._outputs[0]["finish_reason"] = "length"
    response = client.post("/v1/chat/completions", json=request({}))
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_streamed_reasoning_disagreement_is_logged(caplog):
    from executorch.examples.llm_server.python.protocol import DeltaMessage
    from executorch.examples.models.muse_glimmer.tests.test_serve import (
        _HarmonyTemplate,
    )

    class Parser:
        def feed(self, text):
            return []

        def finish(self):
            return [DeltaMessage(reasoning_content="different")]

    client, serving = client_for(
        " to=self<|message|>expected<|eom|> to=user<|message|>answer"
    )
    serving._tool_reasoning_parser_factory = Parser
    serving._template = serving._transcript._template = _HarmonyTemplate()
    serving._runtime._outputs[0]["gen_ids"] = [1, 2, 3]
    body = request({}, True)
    body["session_id"] = "disagreement"
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert "Streamed reasoning differs" in caplog.text
    message, _ = response_message(response, True)
    assert message["reasoning_content"] == "different"
    assert (
        serving._transcript._turns["disagreement"][0]["reasoning_fp"]
        == hashlib.sha256(b"different").digest()
    )
    body["messages"].extend(
        [
            {
                "role": "assistant",
                "content": message["content"],
                "reasoning_content": message["reasoning_content"],
            },
            {"role": "user", "content": "next"},
        ]
    )
    serving._runtime._outputs.append({"text": " to=user<|message|>next answer"})
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert [
        segment["ids"]
        for segment in serving._runtime.prompts[-1].segments or []
        if "ids" in segment
    ] == [[1, 2, 3]]
