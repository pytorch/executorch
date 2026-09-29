# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for GemmaToolCallDetector."""

import json

from executorch.examples.llm_server.python.tool_parsers import GemmaToolCallDetector

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
    "set_alarm": {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "meta": {
                "type": "object",
                "properties": {"priority": {"type": "integer"}},
            },
        },
    },
}


def _parse(text, tools=_TOOLS):
    return GemmaToolCallDetector().detect_and_parse(text, tools)


def test_basic_call():
    text = (
        '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,'
        'units:<|"|>celsius<|"|>}<tool_call|>'
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {
        "city": "Paris",
        "units": "celsius",
    }
    assert r.normal_text == ""


def test_multiple_calls_and_indices():
    text = (
        "<|tool_call>call:add{a:1,b:2}<tool_call|>"
        "<|tool_call>call:add{a:3,b:4}<tool_call|>"
    )
    r = _parse(text)
    assert [c.tool_index for c in r.calls] == [0, 1]
    assert [json.loads(c.arguments) for c in r.calls] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_nested_values():
    text = (
        '<|tool_call>call:set_alarm{enabled:true,labels:[<|"|>wake<|"|>,'
        '<|"|>work<|"|>],meta:{<|"|>priority<|"|>:5}}<tool_call|>'
    )
    r = _parse(text)
    assert json.loads(r.calls[0].arguments) == {
        "enabled": True,
        "labels": ["wake", "work"],
        "meta": {"priority": 5},
    }


def test_leading_text_preserved():
    r = _parse("Checking.<|tool_call>call:add{a:1,b:2}<tool_call|>")
    assert r.normal_text == "Checking."
    assert len(r.calls) == 1


def test_no_tool_call_is_plain_text():
    text = "The weather is nice."
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


def test_undefined_tool_degrades_to_full_text():
    text = "<|tool_call>call:delete_everything{x:1}<tool_call|>"
    r = _parse(text)
    assert r.calls == []
    assert r.normal_text == text


def test_truncated_call_degrades_without_leaking_markup():
    r = _parse('Sure <|tool_call>call:get_weather{city:<|"|>Par')
    assert r.calls == []
    assert r.normal_text == "Sure"


def test_malformed_call_degrades_without_leaking_markup():
    r = _parse('Lead <|tool_call>call:get_weather{city:<|"|>Paris<tool_call|>')
    assert r.calls == []
    assert r.normal_text == "Lead"


def test_string_typed_bare_value_preserves_raw():
    tools = {"f": {"type": "object", "properties": {"code": {"type": "string"}}}}
    r = _parse("<|tool_call>call:f{code:007}<tool_call|>", tools)
    assert json.loads(r.calls[0].arguments) == {"code": "007"}


def test_integer_typed_bare_value_is_int():
    tools = {"f": {"type": "object", "properties": {"n": {"type": "integer"}}}}
    r = _parse("<|tool_call>call:f{n:007}<tool_call|>", tools)
    assert json.loads(r.calls[0].arguments) == {"n": 7}


def test_untyped_bare_vs_quoted_distinction():
    tools = {"f": {"type": "object", "properties": {}}}
    r = _parse('<|tool_call>call:f{a:5,b:<|"|>5<|"|>}<tool_call|>', tools)
    assert json.loads(r.calls[0].arguments) == {"a": 5, "b": "5"}
