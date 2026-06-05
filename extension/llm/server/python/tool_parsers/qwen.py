# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen XML-style tool calls: <function=NAME><parameter=K>V</parameter></function>.

Emitted by Qwen3.5-MoE / Qwen3-Coder (typically wrapped in <tool_call>…
</tool_call>), e.g.:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

This is a DIFFERENT format from HermesDetector (JSON inside <tool_call>); pick the
detector that matches your model. Detection triggers only on the unambiguous
`<function=…>` marker so ordinary prose is not misclassified. Parse failures fall
back to visible text — never a crash or a silent drop.
"""

import json
import logging
import re
from typing import Any, Optional

from .types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)

_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


class _UndefinedToolCall(Exception):
    """A call named a tool not in the request's `tools`. v1 degrades the WHOLE
    response to visible text rather than emitting a partial set (spec)."""


def _coerce(value: str, declared_type: Optional[str]) -> Any:
    """Cast a raw XML parameter string to the type declared in the tool's JSON
    schema.

    The Qwen XML format is stringly-typed (`<parameter=k>v</parameter>`), so
    without the schema we'd have to guess. A bare `json.loads` guess mistypes two
    common ways: a value the schema wants as a string but that looks numeric
    (`"1234"`) becomes an int, and a value the schema wants as a bool but that the
    model didn't write as valid JSON (`True`) stays a string. Coercing to the
    declared type keeps the emitted OpenAI tool_call schema-valid. Falls back to a
    JSON guess (then the raw string) when the type is unknown or coercion fails,
    so untyped/loosely-typed params keep working.
    """
    if declared_type == "string":
        return value
    if declared_type == "boolean":
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
    elif declared_type == "integer":
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            pass
    elif declared_type == "number":
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            pass
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


class QwenFunctionCallDetector:
    """Parses Qwen's XML tool-call format. Create a fresh instance per request
    (it holds the per-request tool-call index); never share across requests."""

    bot_token = "<tool_call>"

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(self, text: str, tools: dict[str, dict]) -> ParseResult:
        """Return leading text + any complete tool calls. On no call or a parse
        failure, return the original text unchanged (kept visible to the client).

        `tools` maps each defined tool name to its JSON-schema ``parameters``
        object; the schema is used to coerce stringly-typed XML values to their
        declared types (and the key set validates names)."""
        first = _FUNCTION_RE.search(text)
        if first is None:
            return ParseResult(normal_text=text)
        # Leading text ends at the <tool_call> wrapper if present, else at the
        # first <function=…> tag.
        cut = text.find(self.bot_token)
        if cut == -1 or cut > first.start():
            cut = first.start()
        normal = text[:cut].strip()
        try:
            calls = self._parse_calls(text, tools)
        except _UndefinedToolCall as e:
            logger.debug("undefined tool %s; returning raw text (no partial calls)", e)
            return ParseResult(normal_text=text)
        except Exception as e:  # noqa: BLE001 - never crash; fall back to visible text
            logger.debug("tool parse failed (%s); returning raw text", e)
            return ParseResult(normal_text=text)
        if not calls:
            return ParseResult(normal_text=text)
        return ParseResult(normal_text=normal, calls=calls)

    def _parse_calls(self, text: str, tools: dict[str, dict]) -> list[ToolCallItem]:
        calls = []
        for name, body in _FUNCTION_RE.findall(text):
            props = (tools.get(name) or {}).get("properties", {})
            args = {
                key: _coerce(value.strip(), props.get(key, {}).get("type"))
                for key, value in _PARAMETER_RE.findall(body)
            }
            calls.append(self._make_item(name, args, tools))
        return calls

    def _make_item(
        self, name: Optional[str], arguments: dict, tools: dict[str, dict]
    ) -> ToolCallItem:
        if not name or name not in tools:
            raise _UndefinedToolCall(repr(name))
        item = ToolCallItem(
            tool_index=self._next_index,
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
        self._next_index += 1
        return item
