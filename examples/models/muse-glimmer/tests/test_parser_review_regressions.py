# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Executable reproductions of the ATEM scanner and runtime review findings."""

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
