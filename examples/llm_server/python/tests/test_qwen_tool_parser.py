# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for QwenFunctionCallDetector (Qwen XML <function=…> tool format)."""

import json

from executorch.examples.llm_server.python.tool_parsers import QwenFunctionCallDetector

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


_TYPED = {
    "code_tool": {"type": "object", "properties": {"code": {"type": "string"}}},
    "calc": {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "x": {"type": "number"},
            "flag": {"type": "boolean"},
        },
    },
}


def test_param_value_with_literal_parameter_close():
    # A value containing literal </parameter> must be preserved, not truncated.
    text = "<function=code_tool><parameter=code>a </parameter> b</parameter></function>"
    r = _parse(text, _TYPED)
    assert json.loads(r.calls[0].arguments) == {"code": "a </parameter> b"}


def test_param_value_with_function_markup():
    # A value containing <function=...> markup must stay in the value, not split.
    text = (
        "<function=code_tool><parameter=code>x = <function=foo></parameter></function>"
    )
    r = _parse(text, _TYPED)
    assert len(r.calls) == 1
    assert json.loads(r.calls[0].arguments) == {"code": "x = <function=foo>"}


def test_declared_integer_with_float_string_kept_raw():
    text = "<function=calc><parameter=n>10.0</parameter></function>"
    val = json.loads(_parse(text, _TYPED).calls[0].arguments)["n"]
    assert val == "10.0" and isinstance(val, str)  # not float 10.0


def test_declared_boolean_with_one_kept_raw():
    text = "<function=calc><parameter=flag>1</parameter></function>"
    val = json.loads(_parse(text, _TYPED).calls[0].arguments)["flag"]
    assert val == "1" and isinstance(val, str)  # not int 1


def test_declared_integer_with_underscores_kept_raw():
    text = "<function=calc><parameter=n>1_000</parameter></function>"
    val = json.loads(_parse(text, _TYPED).calls[0].arguments)["n"]
    assert val == "1_000" and isinstance(val, str)  # not int 1000


def _reject_bare_constant(c):
    # json.loads parse_constant hook: fires only for bare NaN/Infinity/-Infinity.
    raise AssertionError(f"emitted bare non-finite constant: {c}")


def test_declared_number_non_finite_never_emitted():
    for bad in ("NaN", "Infinity", "-Infinity", "1e999"):
        text = f"<function=calc><parameter=x>{bad}</parameter></function>"
        args = _parse(text, _TYPED).calls[0].arguments
        # Strict-client safe: no bare NaN/Infinity constant in the emitted JSON.
        json.loads(args, parse_constant=_reject_bare_constant)
        assert json.loads(args)["x"] == bad  # kept as the raw string


def test_multiple_valid_calls_still_parse():
    text = (
        "<function=add><parameter=a>1</parameter><parameter=b>2</parameter></function>"
        "<function=add><parameter=a>3</parameter><parameter=b>4</parameter></function>"
    )
    r = _parse(text)
    assert [json.loads(c.arguments) for c in r.calls] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_truncated_call_degrades_without_leaking_markup():
    # A call cut off by max_tokens (no closing tags) must NOT leak the partial
    # <function=...> markup -- only the leading text survives (mirrors Hermes).
    text = "Sure! <function=get_weather><parameter=city>Paris"
    r = _parse(text, _TYPED)
    assert not r.calls
    assert "<function=" not in r.normal_text
    assert r.normal_text == "Sure!"


def test_truncated_tool_call_wrapper_no_leak():
    text = "ok <tool_call>\n<function=get_weather><parameter=city>Par"
    r = _parse(text, _TYPED)
    assert not r.calls
    assert "<tool_call>" not in r.normal_text and "<function=" not in r.normal_text
    assert r.normal_text == "ok"
