# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for AtemToolCallDetector (Muse Glimmer ATEM XML tool format).

The Muse Glimmer tool-call surface the model emits is a harmony assistant message
whose body is an ATEM XML block, byte-for-byte:

    [<|start|>assistant] to=NAME<|message|><atem:function_calls>
    <atem:invoke name="NAME">
    <atem:parameter name="k">VALUE</atem:parameter>
    </atem:invoke>
    </atem:function_calls>[<|eom|>|<|eot|>]

Test inputs are hand-built from that spec (the fragment constants / _atem builder
below), NOT copied from parser output, so the assertions independently pin the
format, including the exact newline placement used by the reference renderer.
"""

import json

from executorch.examples.models.muse_glimmer.serving.tool_parsers import (
    AtemToolCallDetector,
)

# Harmony framing fragments, assembled by hand into model-output strings.
_START = "<|start|>"
_MESSAGE = "<|message|>"
_EOM = "<|eom|>"
_EOT = "<|eot|>"


def _atem(name, params):
    """Byte-exact ATEM block for one <atem:invoke>, mirroring the golden renderer:
    newline after <atem:function_calls>, after <atem:invoke ...>, and after EVERY
    </atem:parameter>; newline after </atem:invoke>; NO trailing newline after
    </atem:function_calls>."""
    s = '<atem:function_calls>\n<atem:invoke name="' + name + '">\n'
    for k, v in params:
        s += '<atem:parameter name="' + k + '">' + v + "</atem:parameter>\n"
    s += "</atem:invoke>\n</atem:function_calls>"
    return s


_TOOLS = {
    "get_weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string"},
        },
    },
    "add": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    },
}


def _parse(text, tools=_TOOLS):
    return AtemToolCallDetector().detect_and_parse(text, tools)


def test_basic_call_terminated():
    # A tool call as the worker would surface it with the terminator intact.
    text = (
        f" to=get_weather{_MESSAGE}" + _atem("get_weather", [("city", "Paris")]) + _EOM
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {"city": "Paris"}
    assert r.normal_text == ""


def test_basic_call_without_terminator():
    # The worker registers <|eom|>/<|eot|> as EOS and stops WITHOUT emitting them,
    # so a complete call commonly arrives with no trailing terminator.
    text = f" to=get_weather{_MESSAGE}" + _atem("get_weather", [("city", "Paris")])
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {"city": "Paris"}


def test_call_with_start_prefix():
    # A non-first message in a turn carries its own <|start|>assistant prefix.
    text = (
        f"{_START}assistant to=add{_MESSAGE}"
        + _atem("add", [("a", "1"), ("b", "2")])
        + _EOT
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "add"
    assert json.loads(r.calls[0].arguments) == {"a": 1, "b": 2}


def test_call_with_constrain_field():
    # constrain=json is an optional header field after to=; it must not break
    # detection or leak into the surfaced content.
    text = (
        f" to=get_weather constrain=json{_MESSAGE}"
        + _atem("get_weather", [("city", "SF")])
        + _EOM
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {"city": "SF"}
    assert r.normal_text == ""


def test_call_without_harmony_header():
    # The block alone (no to= header) still parses -- the ATEM block is the
    # trigger, not the harmony header.
    text = _atem("get_weather", [("city", "Paris")])
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {"city": "Paris"}


def test_multiple_invokes_in_one_block():
    # Parallel calls as multiple <atem:invoke> inside a single block (the ts_xml
    # canonical shape).
    block = (
        '<atem:function_calls>\n<atem:invoke name="add">\n'
        '<atem:parameter name="a">1</atem:parameter>\n'
        '<atem:parameter name="b">2</atem:parameter>\n'
        "</atem:invoke>\n"
        '<atem:invoke name="add">\n'
        '<atem:parameter name="a">3</atem:parameter>\n'
        '<atem:parameter name="b">4</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    text = f" to=add{_MESSAGE}" + block + _EOT
    r = _parse(text)
    assert [c.tool_index for c in r.calls] == [0, 1]
    assert [json.loads(c.arguments) for c in r.calls] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_multiple_parallel_calls_across_messages():
    # Parallel calls as one invoke per block across separate assistant messages
    # joined by <|eom|> (the muse_glimmer HF chat-template shape).
    text = (
        f" to=add{_MESSAGE}"
        + _atem("add", [("a", "1"), ("b", "2")])
        + _EOM
        + f"{_START}assistant to=add{_MESSAGE}"
        + _atem("add", [("a", "3"), ("b", "4")])
        + _EOT
    )
    r = _parse(text)
    assert [c.tool_index for c in r.calls] == [0, 1]
    assert [json.loads(c.arguments) for c in r.calls] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_multi_param_call():
    text = f" to=get_weather{_MESSAGE}" + _atem(
        "get_weather", [("city", "Paris"), ("units", "celsius")]
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert json.loads(r.calls[0].arguments) == {"city": "Paris", "units": "celsius"}


def test_leading_prose_before_call_is_preserved():
    text = (
        f"Checking now. to=add{_MESSAGE}"
        + _atem("add", [("a", "5"), ("b", "6")])
        + _EOM
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "add"
    assert r.normal_text == "Checking now."


def test_cot_then_tool_call_leading_text_preserved():
    # A private-CoT message (to=self) precedes the tool call; the CoT body is
    # returned as leading text and only the ATEM block is a call.
    text = (
        f" to=self{_MESSAGE}Let me check the weather.{_EOM}"
        + f"{_START}assistant to=get_weather{_MESSAGE}"
        + _atem("get_weather", [("city", "Paris")])
        + _EOM
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert "Let me check the weather." in r.normal_text


def test_plain_text_passes_through():
    text = "The weather is nice today."
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


def test_undefined_tool_degrades_to_full_text():
    # A well-formed call to a tool not in the request degrades the WHOLE response
    # to the raw text (surface intent, never a partial set).
    text = (
        f" to=delete_everything{_MESSAGE}"
        + _atem("delete_everything", [("target", "all")])
        + _EOM
    )
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


def test_truncated_call_degrades_without_leaking_markup():
    # A call cut off by max_tokens (no </atem:invoke>) must NOT leak the partial
    # markup -- only the leading text survives.
    text = (
        'Sure! <atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Paris'
    )
    r = _parse(text)
    assert not r.calls
    assert "<atem:" not in r.normal_text
    assert r.normal_text == "Sure!"


def test_no_calls_when_block_absent_but_prose_only():
    text = "The capital of France is Paris."
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


# Schema-aware coercion: the ATEM format is stringly-typed, so values must be cast
# to the declared schema type (the cause of several function-calling misses).
def test_integer_value_coerced_by_schema():
    text = f" to=add{_MESSAGE}" + _atem("add", [("a", "2"), ("b", "3")])
    r = _parse(text)
    args = json.loads(r.calls[0].arguments)
    assert args == {"a": 2, "b": 3}
    assert isinstance(args["a"], int) and isinstance(args["b"], int)


def test_number_value_coerced_by_schema():
    tools = {"calc": {"properties": {"x": {"type": "number"}}}}
    text = _atem("calc", [("x", "3.14")])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"x": 3.14} and isinstance(args["x"], float)


def test_boolean_value_coerced_by_schema():
    tools = {"f": {"properties": {"flag": {"type": "boolean"}}}}
    # The renderer emits lowercase true/false for booleans.
    text = _atem("f", [("flag", "true")])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"flag": True} and isinstance(args["flag"], bool)


def test_boolean_capitalized_value_coerced():
    tools = {"f": {"properties": {"flag": {"type": "boolean"}}}}
    text = _atem("f", [("flag", "False")])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"flag": False} and isinstance(args["flag"], bool)


def test_string_schema_keeps_numeric_literal_as_string():
    tools = {"f": {"properties": {"id": {"type": "string"}}}}
    # A numeric-looking value the schema declares as a string must NOT become int.
    text = _atem("f", [("id", "1234")])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"id": "1234"} and isinstance(args["id"], str)


def test_string_value_is_not_trimmed():
    # The renderer contract: string values are emitted verbatim, spaces not
    # stripped.
    tools = {"f": {"properties": {"s": {"type": "string"}}}}
    text = _atem("f", [("s", "  hi  ")])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"s": "  hi  "}


def test_untyped_param_falls_back_to_json_guess():
    # No declared type -> best-effort JSON guess (so loosely-typed tools work).
    tools = {"f": {"properties": {}}}
    text = _atem("f", [("n", "42"), ("items", "[1, 2]")])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"n": 42, "items": [1, 2]}


def test_array_and_object_values_json_decoded():
    # List/object params are rendered as compact JSON; coercion JSON-decodes them.
    tools = {
        "f": {
            "properties": {
                "lst": {"type": "array"},
                "d": {"type": "object"},
            }
        }
    }
    text = _atem("f", [("lst", "[1, 2]"), ("d", '{"k": "v"}')])
    args = json.loads(_parse(text, tools).calls[0].arguments)
    assert args == {"lst": [1, 2], "d": {"k": "v"}}


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


def test_declared_integer_with_float_string_kept_raw():
    text = _atem("calc", [("n", "10.0")])
    val = json.loads(_parse(text, _TYPED).calls[0].arguments)["n"]
    assert val == "10.0" and isinstance(val, str)  # not float 10.0


def test_declared_boolean_with_one_kept_raw():
    text = _atem("calc", [("flag", "1")])
    val = json.loads(_parse(text, _TYPED).calls[0].arguments)["flag"]
    assert val == "1" and isinstance(val, str)  # not int 1


def test_declared_integer_with_underscores_kept_raw():
    text = _atem("calc", [("n", "1_000")])
    val = json.loads(_parse(text, _TYPED).calls[0].arguments)["n"]
    assert val == "1_000" and isinstance(val, str)  # not int 1000


def _reject_bare_constant(c):
    # json.loads parse_constant hook: fires only for bare NaN/Infinity/-Infinity.
    raise AssertionError(f"emitted bare non-finite constant: {c}")


def test_declared_number_non_finite_never_emitted():
    for bad in ("NaN", "Infinity", "-Infinity", "1e999"):
        text = _atem("calc", [("x", bad)])
        args = _parse(text, _TYPED).calls[0].arguments
        # Strict-client safe: no bare NaN/Infinity constant in the emitted JSON.
        json.loads(args, parse_constant=_reject_bare_constant)
        assert json.loads(args)["x"] == bad  # kept as the raw string


def test_param_value_with_literal_parameter_close_preserved():
    # A value containing literal </atem:parameter> must be preserved, not
    # truncated at the first delimiter.
    body = (
        '<atem:function_calls>\n<atem:invoke name="code_tool">\n'
        '<atem:parameter name="code">a </atem:parameter> b</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    r = _parse(body, _TYPED)
    assert json.loads(r.calls[0].arguments) == {"code": "a </atem:parameter> b"}


def test_param_value_with_xmlish_markup_not_truncated():
    # A value containing <atem:...>-like markup must stay in the value, not split
    # the call or truncate at the first tag.
    body = (
        '<atem:function_calls>\n<atem:invoke name="code_tool">\n'
        '<atem:parameter name="code">x = <atem:invoke name="foo"></atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    r = _parse(body, _TYPED)
    assert len(r.calls) == 1
    assert json.loads(r.calls[0].arguments) == {"code": 'x = <atem:invoke name="foo">'}


def test_value_with_angle_brackets_kept_as_string():
    # A string value containing < and > must survive intact (no truncation).
    tools = {"f": {"properties": {"expr": {"type": "string"}}}}
    body = (
        '<atem:function_calls>\n<atem:invoke name="f">\n'
        '<atem:parameter name="expr">a < b && c > d</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    r = _parse(body, tools)
    assert json.loads(r.calls[0].arguments) == {"expr": "a < b && c > d"}


def test_per_request_index_is_isolated():
    # A fresh detector per request restarts tool_index at 0.
    text = f" to=add{_MESSAGE}" + _atem("add", [("a", "1"), ("b", "2")])
    first = AtemToolCallDetector().detect_and_parse(text, _TOOLS)
    second = AtemToolCallDetector().detect_and_parse(text, _TOOLS)
    assert first.calls[0].tool_index == 0
    assert second.calls[0].tool_index == 0
