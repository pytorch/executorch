# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Executable reproductions of the ATEM scanner and runtime review findings."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")

from executorch.examples.llm_server.python.errors import APIError  # noqa: E402
from executorch.examples.llm_server.python.session_runtime import (  # noqa: E402
    _copy_stats,
    GenStats,
)
from executorch.examples.llm_server.python.worker_client import (  # noqa: E402
    WorkerError,
    WorkerStats,
)
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", ["edit", "3d_render", "-render"])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_complete_calls_allow_prose_and_all_openai_name_prefixes(
    stream, name, compact, wrapped
):
    block = _atem(f"{name}.{name}", [("q", "ok")])
    if not wrapped:
        block = block.removeprefix("<atem:function_calls>\n").removesuffix(
            "\n</atem:function_calls>"
        )
    if compact:
        block = block.replace("\n", "")
    raw = f" to={name}.{name}<|message|>" + block + "\nDone."
    client, _ = client_for(raw)
    body = request({"properties": {"q": {"type": "string"}}}, stream)
    body["tools"][0]["function"]["name"] = name
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert message["tool_calls"][0]["function"]["name"] == name
    assert message["content"] == "Done."


def test_large_unclosed_and_quoted_scans_complete_without_quadratic_work():
    code = """
from executorch.examples.models.muse_glimmer.serving.serve import _extract_muse_glimmer_reasoning
for prefix, item in [
    ('<atem:function_calls>', '\\n<atem:invoke name="x">'),
    (' to=user<|message|>', '\\nExample: <atem:invoke name="x">'),
    ('', 'quoted <atem:invoke name="x">'),
]:
    _extract_muse_glimmer_reasoning(prefix + item * 16000)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


@pytest.mark.parametrize(
    "keyword", ["allOf", "anyOf", "oneOf", "not", "if", "dependentSchemas"]
)
def test_same_instance_reference_cycles_rejected_before_generation(keyword):
    ref = {"$ref": "#/$defs/a"}
    value = (
        [ref]
        if keyword in ("allOf", "anyOf", "oneOf")
        else ({"q": ref} if keyword == "dependentSchemas" else ref)
    )
    schema = {"$defs": {"a": {keyword: value}}, "$ref": "#/$defs/a"}
    client, serving = client_for("unused")
    response = client.post("/v1/chat/completions", json=request(schema))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert not serving._runtime.prompts


def test_unevaluated_reference_annotation_uses_public_resolver():
    schema = ToolSchema(
        {
            "$defs": {"a": {"properties": {"q": {"type": "string"}}}},
            "$ref": "#/$defs/a",
            "unevaluatedProperties": False,
        }
    )
    schema.validate("edit", {"q": "ok"})
    with pytest.raises(APIError):
        schema.validate("edit", {"extra": "bad"})


def test_only_legacy_worker_stats_report_unknown_string_stop():
    stats = GenStats()
    _copy_stats(
        stats,
        SimpleNamespace(
            num_prompt_tokens=1, num_generated_tokens=2, finish_reason="stop"
        ),
    )
    assert stats.matched_stop == ""
    _copy_stats(stats, WorkerStats.from_message({"finish_reason": "stop"}))
    assert stats.matched_stop is None
    _copy_stats(
        stats, WorkerStats.from_message({"finish_reason": "stop", "matched_stop": None})
    )
    assert stats.matched_stop == ""


@pytest.mark.parametrize(
    "value,finish",
    [
        (1, "stop"),
        (True, "stop"),
        ({"stop": "x"}, "stop"),
        ([], "stop"),
        ("", "stop"),
        ("x", "length"),
        ("x", None),
    ],
)
def test_legacy_worker_rejects_invalid_stop_metadata(value, finish):
    with pytest.raises(WorkerError, match="matched_stop"):
        WorkerStats.from_message({"matched_stop": value, "finish_reason": finish})


@pytest.mark.parametrize("recipient", ["self", "user"])
def test_ended_recipient_does_not_own_following_headerless_call(recipient):
    client, _ = client_for(
        f" to={recipient}<|message|>before<|eom|>\n" + _atem("edit.edit", [("q", "ok")])
    )
    response = client.post("/v1/chat/completions", json=request({}))
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("role", ["system", "developer"])
def test_late_system_instructions_do_not_delay_tool_metadata(role):
    from pathlib import Path

    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = Path(__file__).parent.parent / "serving" / "chat_template.jinja"
    rendered = _compile_jinja_template(template.read_text()).render(
        messages=[
            {"role": "user", "content": "question"},
            {"role": role, "content": "late instruction"},
        ],
        tools=request({})["tools"],
        bos_token="",
        current_date="2026-11-05",
    )
    assert rendered.index("edit") < rendered.index("<|start|>user")
    assert rendered.index("late instruction") > rendered.index("<|start|>user")


def test_user_addressed_parameter_example_does_not_change_recipient():
    from executorch.examples.models.muse_glimmer.serving.serve import (
        _extract_muse_glimmer_reasoning,
    )

    example = _atem(
        "edit.edit", [("q", "<|eom|><|start|>assistant to=self<|message|>quoted")]
    )
    reasoning, content = _extract_muse_glimmer_reasoning(
        " to=user<|message|>" + example
    )
    assert reasoning is None
    assert content == example


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("placement", ["alone", "before", "after", "between"])
@pytest.mark.parametrize("prefix", ["For example: ", "```xml\nFor example: "])
def test_skipped_wrapper_owns_invocations(stream, placement, prefix):
    example = prefix + _atem("edit.edit", [("q", "example")])
    if prefix.startswith("```"):
        example += "\n```"
    real = _atem("edit.edit", [("q", "real")])
    parts = {
        "alone": [example],
        "before": [example, real],
        "after": [real, example],
        "between": [real, example, real],
    }[placement]
    client, _ = client_for("\n".join(parts))
    response = client.post("/v1/chat/completions", json=request({}, stream))
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    calls = message.get("tool_calls") or []
    assert len(calls) == parts.count(real)
    assert all(
        json.loads(call["function"]["arguments"]) == {"q": "real"} for call in calls
    )
    assert message["content"] == example
    assert finish == ("tool_calls" if calls else "stop")


@pytest.mark.parametrize("stream", [False, True])
def test_skipping_user_wrapper_keeps_following_user_invocation_literal(stream):
    example = _atem("edit.edit", [("q", "example")])
    bare = example.removeprefix("<atem:function_calls>\n").removesuffix(
        "\n</atem:function_calls>"
    )
    content = example + "\n" + bare
    client, _ = client_for(" to=user<|message|>" + content)
    response = client.post("/v1/chat/completions", json=request({}, stream))
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert not message.get("tool_calls")
    assert message["content"] == content


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("keyword", ["pattern", "patternProperties"])
@pytest.mark.parametrize(
    "draft",
    [
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
@pytest.mark.parametrize(
    "pattern,value", [(r"^\p{L}+$", "é東京"), (r"^(?<year>\d{4})$", "2026")]
)
def test_ecma_patterns_use_same_engine_at_preflight_and_validation(
    stream, keyword, draft, pattern, value
):
    schema = (
        {"properties": {"q": {"type": "string", "pattern": pattern}}}
        if keyword == "pattern"
        else {
            "patternProperties": {pattern: {"type": "string"}},
            "additionalProperties": False,
        }
    )
    key, argument = ("q", value) if keyword == "pattern" else (value, "ok")
    client, _ = client_for(_atem("edit.edit", [(key, argument)]))
    response = client.post(
        "/v1/chat/completions", json=request({"$schema": draft, **schema}, stream)
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        key: argument
    }


@pytest.mark.parametrize("keyword", ["pattern", "patternProperties"])
def test_invalid_regex_still_rejected_before_generation(keyword):
    schema = {keyword: "[" if keyword == "pattern" else {"[": {}}}
    client, serving = client_for("unused")
    response = client.post("/v1/chat/completions", json=request(schema))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_schema"
    assert not serving._runtime.prompts


@pytest.mark.parametrize("keyword", ["uniqueItems", "enum", "const"])
def test_bounded_comparison_matches_jsonschema(keyword):
    from jsonschema import Draft202012Validator

    values = [
        None,
        True,
        False,
        0,
        1,
        1.0,
        "1",
        [1, True],
        [1.0, True],
        {"x": [1], "y": True},
        {"y": True, "x": [1.0]},
        {"x": [True], "y": 1},
    ]
    for left in values:
        for right in values:
            if keyword == "uniqueItems":
                schema, value = {keyword: True}, [left, right]
            else:
                schema, value = {keyword: [left] if keyword == "enum" else left}, right
            valid = Draft202012Validator(schema).is_valid(value)
            tool = ToolSchema(schema)
            if valid:
                tool.validate("edit", value)
            else:
                with pytest.raises(ToolOutputError):
                    tool.validate("edit", value)


def test_large_unique_object_array_does_not_do_pairwise_comparisons():
    code = """
from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import ToolSchema
schema = ToolSchema({'type': 'array', 'uniqueItems': True, 'items': {'type': 'object'}})
schema.validate('edit', [{'value': i} for i in range(5000)])
schema.validate('edit', [{'value': i * (2**61 - 1)} for i in range(5000)])
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=8)


@pytest.mark.parametrize("keyword", ["uniqueItems", "enum", "const"])
def test_deep_comparison_spends_budget(keyword):
    value = {"data": list(range(100))}
    schema = {
        keyword: (
            True
            if keyword == "uniqueItems"
            else ([value] if keyword == "enum" else value)
        )
    }
    instance = [value] if keyword == "uniqueItems" else value
    tool = ToolSchema(schema)
    budget = ToolOutputBudget()
    budget.remaining = 150
    with pytest.raises(ToolOutputError, match="budget"):
        tool.validate("edit", instance, budget=budget)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cutoff", ["length", "request", "eos"])
@pytest.mark.parametrize(
    "value",
    [
        "</atem:invoke></atem:function_calls>",
        "```\n</atem:invoke>\n</atem:function_calls>",
    ],
)
def test_unfinished_parameter_owns_literal_closing_tags(stream, cutoff, value):
    raw = (
        '<atem:function_calls><atem:invoke name="edit.edit"><atem:parameter name="q">'
        + value
    )
    client, serving = client_for(raw)
    body = request({}, stream)
    serving._runtime._outputs[0]["finish_reason"] = (
        "length" if cutoff == "length" else "stop"
    )
    if cutoff == "request":
        body["stop"] = "CUT"
        serving._runtime._outputs[0]["matched_stop"] = "CUT"
    response = client.post("/v1/chat/completions", json=body)
    if cutoff == "eos":
        assert response.status_code == (200 if stream else 502)
        assert '"invalid_tool_output"' in response.text
    else:
        assert response.status_code == 200
        message, finish = response_message(response, stream)
        assert finish == ("length" if cutoff == "length" else "stop")
        assert not message.get("tool_calls")


def test_hundred_mcp_style_tools_fit_request_budget():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to read or edit."},
            "mode": {"type": "string", "enum": ["read", "write", "append"]},
            "options": {
                "type": "object",
                "properties": {
                    "recursive": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["path"],
    }
    body = request(schema)
    body["tools"] = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "A workspace tool.",
                "parameters": schema,
            },
        }
        for i in range(100)
    ]
    client, serving = client_for("ok")
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert serving._runtime.prompts


@pytest.mark.parametrize("parameters", [True, False])
def test_http_tool_parameters_require_schema_objects(parameters):
    client, serving = client_for("unused")
    assert (
        client.post("/v1/chat/completions", json=request(parameters)).status_code == 422
    )
    assert not serving._runtime.prompts


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("quoted", [False, True])
def test_nested_wrapper_ownership_is_preserved(stream, quoted):
    inner = _atem("edit.edit", [("q", "inner")])
    example_call = '<atem:invoke name="edit.edit"><atem:parameter name="q">example</atem:parameter></atem:invoke>'
    nested = (
        "<atem:function_calls>\n"
        + inner
        + "\n"
        + example_call
        + "\n</atem:function_calls>"
    )
    prefix = "Example: " if quoted else ""
    client, _ = client_for(prefix + nested + "\n" + _atem("edit.edit", [("q", "real")]))
    response = client.post("/v1/chat/completions", json=request({}, stream))
    if not quoted:
        assert response.status_code == (200 if stream else 502)
        assert '"invalid_tool_output"' in response.text
        return
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert message["content"] == prefix + nested
    assert [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]] == [
        {"q": "real"}
    ]
