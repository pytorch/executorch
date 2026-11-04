# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parse Muse Glimmer ATEM text into OpenAI-compatible tool calls.

Blocks may contain multiple invocations. Parameters are coerced using the
declared tool schema. Malformed or invalid calls fail the entire response.
"""

import json
import re
from typing import Optional

from executorch.examples.llm_server.python.tool_parsers.types import (
    ParseResult,
    ToolCallItem,
)

from ..chat_template import wire_tool_name
from .schema import ToolOutputError, ToolSchema

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
    rf" to=({_RECIPIENT})"
    rf"(?: constrain={_IDENT})?"
    rf"{re.escape(_MESSAGE)}"
)


def tool_start(text):
    headers = list(_HEADER_RE.finditer(text))
    for marker in re.finditer(r"<atem:function_calls>|<atem:invoke\b", text):
        prefix = text[: marker.start()]
        header = next((h for h in reversed(headers) if h.end() <= marker.start()), None)
        if header is not None:
            if header.group(1) in ("self", "user"):
                continue
            if not text[header.end() : marker.start()].strip():
                return marker.start()
        # Standalone blocks must begin a line outside quoted Markdown code.
        if not prefix.rsplit("\n", 1)[-1].strip() and prefix.count("```") % 2 == 0:
            return marker.start()
    return None


def parameter_value_spans(text):
    start = tool_start(text)
    if start is None:
        return []
    return [
        (invoke.start(2) + parameter.start(2), invoke.start(2) + parameter.end(2))
        for invoke in _INVOKE_RE.finditer(text, start)
        for parameter in _PARAMETER_RE.finditer(invoke.group(2))
    ]


class IncompleteToolOutputError(ToolOutputError):
    pass


class AtemToolCallDetector:
    """Parse one request's Muse Glimmer ATEM tool calls."""

    # The block open is the unambiguous trigger. serving_chat only reads
    # ``bot_token`` for presence checks / leading-text cut points, so the ATEM
    # block opener is the right marker here.
    bot_token = _FUNCTION_CALLS_OPEN

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(
        self, text: str, tools: dict, *, finish_reason=None
    ) -> ParseResult:
        start = tool_start(text)
        if start is None:
            return ParseResult(normal_text=text)
        header = self._preceding_header(text, start)
        normal = text[: header if header is not None else start].strip()
        try:
            if _INVOKE_RE.search(text, start) is None:
                raise IncompleteToolOutputError("Incomplete ATEM tool call")
            calls = self._parse_calls(text[start:], tools)
        except IncompleteToolOutputError:
            if finish_reason == "length":
                return ParseResult(normal_text=normal, incomplete=True)
            raise
        except (ValueError, RecursionError) as error:
            raise ToolOutputError("Malformed ATEM tool call") from error
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
        remainder = _INVOKE_RE.sub("", text)
        if re.search(r"</?atem:(?:invoke|parameter)\b", remainder):
            raise IncompleteToolOutputError("Incomplete invocation")
        calls = []
        for im in _INVOKE_RE.finditer(text):
            name, body = im.group(1), im.group(2)
            if name not in tools:
                matches = [key for key in tools if wire_tool_name(key) == name]
                if len(matches) == 1:
                    name = matches[0]
            if name not in tools:
                raise ToolOutputError(f"Model called undefined tool {name!r}")
            schema = tools[name]
            if not isinstance(schema, ToolSchema):
                schema = ToolSchema(schema)
            args = {}
            end = 0
            for pm in _PARAMETER_RE.finditer(body):
                if body[end : pm.start()].strip():
                    raise ValueError("Malformed parameter body")
                key = pm.group(1)
                if key in args:
                    raise ValueError("Duplicate parameter")
                args[key] = pm.group(2)
                end = pm.end()
            if body[end:].strip():
                raise ValueError("Unclosed or malformed parameter")
            args = schema.coerce_arguments(name, args)
            calls.append(self._make_item(name, args, tools))
        return calls

    def _make_item(
        self, name: Optional[str], arguments: dict, tools: dict[str, dict]
    ) -> ToolCallItem:
        if not name or name not in tools:
            raise ToolOutputError(f"Model called undefined tool {name!r}")
        item = ToolCallItem(
            tool_index=self._next_index,
            name=name,
            # Reject non-finite values that escaped coercion.
            arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False),
        )
        self._next_index += 1
        return item
