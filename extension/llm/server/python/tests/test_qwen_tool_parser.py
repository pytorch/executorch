# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for QwenFunctionCallDetector (Qwen XML <function=…> tool format)."""

import json

from executorch.extension.llm.server.python.tool_parsers import QwenFunctionCallDetector

# name -> JSON-schema `parameters` (as the server passes it to the detector).
_TOOLS = {
    "get_weather": {"type": "object", "properties": {"city": {"type": "string"}}},
    "add": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    },
}


def _parse(text, tools=_TOOLS):
    return QwenFunctionCallDetector().detect_and_parse(text, tools)


def test_basic_call():
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {"city": "Paris"}
    assert r.normal_text == ""


def test_observed_model_output():
    # The exact shape seen from Qwen3.5-MoE during the live smoke.
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    r = _parse(text)
    assert [c.name for c in r.calls] == ["get_weather"]


def test_numeric_and_multi_param_coercion():
    text = (
        "<function=add><parameter=a>2</parameter>"
        "<parameter=b>3</parameter></function>"
    )
    r = _parse(text)
    assert json.loads(r.calls[0].arguments) == {"a": 2, "b": 3}


def test_multiple_calls():
    text = (
        "<function=get_weather><parameter=city>Paris</parameter></function>"
        "<function=add><parameter=a>1</parameter></function>"
    )
    r = _parse(text)
    assert [c.name for c in r.calls] == ["get_weather", "add"]
    assert [c.tool_index for c in r.calls] == [0, 1]


def test_leading_text_preserved():
    text = "Let me check.<function=get_weather><parameter=city>Paris</parameter></function>"
    r = _parse(text)
    assert r.normal_text == "Let me check."
    assert len(r.calls) == 1


def test_no_tool_call_is_plain_text():
    text = "The capital of France is Paris."
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


def test_undefined_tool_degrades_to_text():
    # A call to a tool not in the request -> whole response kept as visible text.
    text = "<function=delete_everything><parameter=x>1</parameter></function>"
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


def test_missing_tool_call_wrapper_still_parses():
    # Tolerate a truncated/absent <tool_call> wrapper as long as the function
    # block is complete.
    text = "<function=get_weather><parameter=city>Paris</parameter></function>"
    r = _parse(text)
    assert len(r.calls) == 1
    assert json.loads(r.calls[0].arguments) == {"city": "Paris"}


# Schema-aware coercion: the XML format is stringly-typed, so values must be cast
# to the declared schema type (the cause of several BFCL function-calling misses).
def test_boolean_value_coerced_by_schema():
    tools = {"f": {"properties": {"flag": {"type": "boolean"}}}}
    # The model writes a non-JSON capitalized "True"; the schema says boolean.
    text = "<function=f><parameter=flag>True</parameter></function>"
    r = _parse(text, tools)
    assert json.loads(r.calls[0].arguments) == {"flag": True}


def test_string_schema_keeps_numeric_literal_as_string():
    tools = {"f": {"properties": {"id": {"type": "string"}}}}
    # A numeric-looking value the schema declares as a string must NOT become int.
    text = "<function=f><parameter=id>1234</parameter></function>"
    r = _parse(text, tools)
    args = json.loads(r.calls[0].arguments)
    assert args == {"id": "1234"} and isinstance(args["id"], str)


def test_untyped_param_falls_back_to_json_guess():
    # No declared type -> best-effort JSON guess (so loosely-typed tools still work).
    tools = {"f": {"properties": {}}}
    text = (
        "<function=f><parameter=n>42</parameter>"
        "<parameter=items>[1, 2]</parameter></function>"
    )
    r = _parse(text, tools)
    assert json.loads(r.calls[0].arguments) == {"n": 42, "items": [1, 2]}
