# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parse Muse Glimmer ATEM text into OpenAI-compatible tool calls.

Blocks may contain multiple invocations. Parameters are coerced using the
declared tool schema. Malformed blocks are hidden, while calls to undefined
tools remain visible text.
"""

import json
import logging
import math
import re
from typing import Any, Optional

from executorch.examples.llm_server.python.tool_parsers.types import (
    ParseResult,
    ToolCallItem,
)

logger = logging.getLogger(__name__)

_START = "<|start|>"
_MESSAGE = "<|message|>"
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_INVOKE_OPEN = "<atem:invoke"

# Require a following structural boundary so delimiter-like text inside a
# parameter value does not end the match early.
_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="([^"]*)"\s*>(.*?)</atem:invoke>\s*'
    r"(?=<atem:invoke|</atem:function_calls>|<atem:function_calls>|\Z)",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r'<atem:parameter\s+name="([^"]*)"\s*>(.*?)</atem:parameter>\s*'
    r"(?=<atem:parameter|</atem:invoke>|\Z)",
    re.DOTALL,
)

# Match the optional assistant prefix and recipient header before an ATEM block.
_IDENT = r"[a-zA-Z_][a-zA-Z0-9_-]*"
_RECIPIENT = rf"{_IDENT}(?:\.{_IDENT})*"
_HEADER_RE = re.compile(
    rf"(?:{re.escape(_START)}assistant)?"
    rf" to={_RECIPIENT}"
    rf"(?: constrain={_IDENT})?"
    rf"{re.escape(_MESSAGE)}"
)

_INT_RE = re.compile(r"[+-]?[0-9]+$")
_NUM_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


class _UndefinedToolCall(Exception):
    """Raised when a call names a tool absent from the request."""


def _coerce(value: str, declared_type: Optional[str]) -> Any:
    """Coerce a raw parameter using its declared JSON-schema type.

    Failed declared conversions remain strings. Untyped values use JSON parsing,
    and non-finite numbers remain strings so serialized arguments stay valid.
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
    # Unknown/absent declared type (also array/object): a JSON guess, but reject
    # non-finite (json.loads parses NaN/Infinity by default, which json.dumps
    # would then re-emit).
    try:
        guess = json.loads(v)
    except (ValueError, TypeError):
        return value
    if isinstance(guess, float) and not math.isfinite(guess):
        return value
    return guess


class AtemToolCallDetector:
    """Parse one request's Muse Glimmer ATEM tool calls."""

    # The block open is the unambiguous trigger. serving_chat only reads
    # ``bot_token`` for presence checks / leading-text cut points, so the ATEM
    # block opener is the right marker here.
    bot_token = _FUNCTION_CALLS_OPEN

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(self, text: str, tools: dict[str, dict]) -> ParseResult:
        """Return leading text + any complete tool calls.

        Missing markers preserve the text. Truncated or malformed blocks are
        removed from the first marker; undefined tools preserve the full text.
        ``tools`` maps tool names to their parameter schemas.
        """
        first = _INVOKE_RE.search(text)
        if first is None:
            # No complete call. If a tool marker is present the call was
            # truncated/partial -> strip it; otherwise there is no tool intent.
            markers = [
                i
                for i in (text.find(self.bot_token), text.find(_INVOKE_OPEN))
                if i != -1
            ]
            if markers:
                return ParseResult(normal_text=text[: min(markers)].strip())
            return ParseResult(normal_text=text)
        # Leading text ends at the block open if present, else at the first
        # <atem:invoke>; a harmony "to=NAME<|message|>" header that immediately
        # precedes it is excluded so the header does not leak into content.
        block = text.find(self.bot_token)
        cut = block if (block != -1 and block <= first.start()) else first.start()
        header = self._preceding_header(text, cut)
        if header is not None:
            cut = header
        normal = text[:cut].strip()
        try:
            calls = self._parse_calls(text, tools)
        except _UndefinedToolCall as e:
            logger.debug("undefined tool %s; returning raw text (no partial calls)", e)
            return ParseResult(normal_text=text)
        except Exception as e:  # noqa: BLE001 - never crash
            logger.debug("malformed ATEM tool call (%s); degrading to leading text", e)
            return ParseResult(normal_text=normal)
        if not calls:
            return ParseResult(normal_text=text)
        return ParseResult(normal_text=normal, calls=calls)

    @staticmethod
    def _preceding_header(text: str, cut: int) -> Optional[int]:
        """Return the start of a recipient header adjacent to ``cut``."""
        last: Optional[int] = None
        for m in _HEADER_RE.finditer(text):
            if m.end() <= cut and text[m.end() : cut].strip() == "":
                last = m.start()
        return last

    def _parse_calls(self, text: str, tools: dict[str, dict]) -> list[ToolCallItem]:
        """Parse every invocation and coerce its parameters."""
        calls = []
        for im in _INVOKE_RE.finditer(text):
            name, body = im.group(1), im.group(2)
            props = (tools.get(name) or {}).get("properties", {})
            args = {}
            for pm in _PARAMETER_RE.finditer(body):
                key = pm.group(1)
                args[key] = _coerce(pm.group(2), props.get(key, {}).get("type"))
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
            # Reject non-finite values that escaped coercion.
            arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False),
        )
        self._next_index += 1
        return item
