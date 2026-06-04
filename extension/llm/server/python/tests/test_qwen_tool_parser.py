# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for QwenFunctionCallDetector (Qwen XML <function=…> tool format)."""

import json

from executorch.extension.llm.server.python.tool_parsers import QwenFunctionCallDetector

_TOOLS = {"get_weather", "add"}


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
