# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for HermesDetector (Hermes/Qwen JSON <tool_call> format).

Covers the explicit all-or-nothing malformed-call policy and the no-markup-leak
guarantee: an undefined/malformed/truncated call degrades to the leading text
with the <tool_call> markup stripped, never surfaced to the client.
"""

import json

from executorch.examples.llm_server.python.tool_parsers import HermesDetector

_TOOLS = {
    "get_weather": {"type": "object", "properties": {"city": {"type": "string"}}},
    "echo": {"type": "object", "properties": {"text": {"type": "string"}}},
}


def _parse(text, tools=_TOOLS):
    return HermesDetector().detect_and_parse(text, tools)


def test_basic_call():
    text = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )
    r = _parse(text)
    assert len(r.calls) == 1 and r.calls[0].name == "get_weather"
    assert json.loads(r.calls[0].arguments) == {"city": "Paris"}


def test_multiple_calls_still_parse():
    text = (
        '<tool_call>{"name": "echo", "arguments": {"text": "a"}}</tool_call>'
        '<tool_call>{"name": "echo", "arguments": {"text": "b"}}</tool_call>'
    )
    r = _parse(text)
    assert [json.loads(c.arguments)["text"] for c in r.calls] == ["a", "b"]


def test_no_tool_call_is_passthrough():
    r = _parse("just some text")
    assert not r.calls and r.normal_text == "just some text"


def test_malformed_block_with_valid_sibling_degrades_no_leak():
    # All-or-nothing: one malformed block degrades the WHOLE response (the valid
    # sibling is NOT emitted in isolation), and no <tool_call> markup leaks.
    text = (
        'lead<tool_call>{"name": "echo", "arguments": {"text": "ok"}}</tool_call>'
        "<tool_call>{bad json}</tool_call>"
    )
    r = _parse(text)
    assert not r.calls
    assert "<tool_call>" not in r.normal_text
    assert r.normal_text == "lead"


def test_unclosed_marker_degrades_no_leak():
    text = 'lead<tool_call>{"name": "echo", "arguments": {"text": "x"}}'
    r = _parse(text)
    assert not r.calls
    assert "<tool_call>" not in r.normal_text
    assert r.normal_text == "lead"


def test_string_value_containing_close_marker_not_truncated():
    # A JSON string value containing literal </tool_call> must not truncate the
    # captured JSON (raw_decode parses the whole object regardless).
    text = (
        '<tool_call>{"name": "echo", "arguments": '
        '{"text": "a </tool_call> b"}}</tool_call>'
    )
    r = _parse(text)
    assert len(r.calls) == 1
    assert json.loads(r.calls[0].arguments) == {"text": "a </tool_call> b"}


def test_arguments_null_falls_back_to_parameters():
    text = (
        '<tool_call>{"name": "echo", "arguments": null, '
        '"parameters": {"text": "p"}}</tool_call>'
    )
    r = _parse(text)
    assert json.loads(r.calls[0].arguments) == {"text": "p"}


def test_undefined_tool_degrades_to_full_text():
    # A WELL-FORMED call to an undefined tool degrades the whole response to
    # visible text (unchanged policy: surface the model's intent, never a partial
    # set). This differs from the malformed/truncated case, which strips markup.
    text = 'hi<tool_call>{"name": "nope", "arguments": {}}</tool_call>'
    r = _parse(text)
    assert not r.calls
    assert "<tool_call>" in r.normal_text  # full text, markup visible
