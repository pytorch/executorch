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
import math
import re
from typing import Any, Optional

from .types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)

# Structural matching, NOT first-close-wins. A parameter value runs to the
# </parameter> that is followed by the next <parameter= or the end of the
# function body; a function body runs to the </function> that is followed by the
# next <function=, the tool-call wrapper, or end. This preserves a value that
# itself contains literal </parameter>, </function>, or <function=...> markup
# instead of silently truncating the call at the first delimiter (Qwen3-Coder
# emitting code/markup is a realistic trigger).
# Bound: a value containing the exact sequence "</parameter><parameter=" (or
# "</function><function=") can still mis-split. That adversarial case would need a
# full structural scanner; it is out of scope here (the realistic cases are fixed).
_FUNCTION_RE = re.compile(
    r"<function=([^>\s]+)\s*>(.*?)</function>\s*(?=<function=|</tool_call>|<tool_call>|\Z)",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter>\s*(?=<parameter=|\Z)",
    re.DOTALL,
)
_INT_RE = re.compile(r"[+-]?[0-9]+$")
_NUM_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


class _UndefinedToolCall(Exception):
    """A call named a tool not in the request's `tools`. Degrades the WHOLE
    response to visible text rather than emitting a partial set (spec)."""


def _coerce(value: str, declared_type: Optional[str]) -> Any:
    """Cast a raw XML parameter string to the type declared in the tool's JSON
    schema.

    The Qwen XML format is stringly-typed (`<parameter=k>v</parameter>`), so
    without the schema we'd have to guess. Coercing to the declared type keeps the
    emitted OpenAI tool_call schema-valid.

    On a DECLARED type whose strict cast fails, keep the raw string rather than
    falling through to a JSON guess that would emit a *different* JSON type (the
    bug this guards): `integer` + "10.0" must not become float 10.0; `boolean` +
    "1" must not become int 1; underscore numerics ("1_000") are not accepted for
    numeric types. Non-finite floats (NaN/Infinity/1e999) are never emitted -- they
    are kept as the raw string -- so `arguments` is always valid JSON. Only when
    the type is unknown do we make a JSON guess (then raw string), so
    untyped/loosely-typed params keep working.
    """
    v = value.strip()
    if declared_type == "string":
        return value
    if declared_type == "boolean":
        low = v.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        return value  # not a valid bool literal -> keep raw, don't mistype
    if declared_type == "integer":
        # strict: digits only (no float, no underscores)
        return int(v) if _INT_RE.match(v) else value
    if declared_type == "number":
        if _NUM_RE.match(v):
            f = float(v)
            if math.isfinite(f):
                return f
        return value  # non-numeric / non-finite -> keep raw, never emit NaN/Inf
    # Unknown declared type: a JSON guess, but reject non-finite (json.loads
    # parses NaN/Infinity by default, which json.dumps would then re-emit).
    try:
        guess = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(guess, float) and not math.isfinite(guess):
        return value
    return guess


class QwenFunctionCallDetector:
    """Parses Qwen's XML tool-call format. Create a fresh instance per request
    (it holds the per-request tool-call index); never share across requests."""

    bot_token = "<tool_call>"

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(self, text: str, tools: dict[str, dict]) -> ParseResult:
        """Return leading text + any complete tool calls.

        Degrade policy (mirrors HermesDetector):
         * No tool marker at all -> return the text unchanged.
         * A WELL-FORMED call to an undefined tool -> degrade the whole response
           to the full visible text (surface the model's intent; never a partial
           set).
         * A TRUNCATED/partial call (a <function=/<tool_call> marker present but
           no complete <function=…></function>, e.g. cut by max_tokens) or other
           malformed markup -> degrade to the LEADING text before the first
           marker, so the raw markup is never leaked to the client as content.

        `tools` maps each defined tool name to its JSON-schema ``parameters``
        object; the schema is used to coerce stringly-typed XML values to their
        declared types (and the key set validates names)."""
        first = _FUNCTION_RE.search(text)
        if first is None:
            # No complete call. If a tool marker is present the call was
            # truncated/partial -> strip it; otherwise there is no tool intent.
            markers = [
                i
                for i in (text.find(self.bot_token), text.find("<function="))
                if i != -1
            ]
            if markers:
                return ParseResult(normal_text=text[: min(markers)].strip())
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
            # well-formed call to an undefined tool: surface full text (no partial set)
            logger.debug("undefined tool %s; returning raw text (no partial calls)", e)
            return ParseResult(normal_text=text)
        except Exception as e:  # noqa: BLE001 - never crash
            # malformed markup: degrade to leading text (don't leak partial markup)
            logger.debug("malformed tool call (%s); degrading to leading text", e)
            return ParseResult(normal_text=normal)
        if not calls:
            return ParseResult(normal_text=text)
        return ParseResult(normal_text=normal, calls=calls)

    def _parse_calls(self, text: str, tools: dict[str, dict]) -> list[ToolCallItem]:
        calls = []
        for fm in _FUNCTION_RE.finditer(text):
            name, body = fm.group(1), fm.group(2)
            props = (tools.get(name) or {}).get("properties", {})
            args = {}
            for pm in _PARAMETER_RE.finditer(body):
                key = pm.group(1)
                args[key] = _coerce(pm.group(2).strip(), props.get(key, {}).get("type"))
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
            # allow_nan=False: belt-and-suspenders so a non-finite that escaped
            # _coerce raises (-> caught -> text fallback) rather than emitting
            # invalid JSON (NaN/Infinity) a strict client would reject.
            arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False),
        )
        self._next_index += 1
        return item
