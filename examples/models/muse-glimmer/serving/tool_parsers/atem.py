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
    rf" to={_RECIPIENT}"
    rf"(?: constrain={_IDENT})?"
    rf"{re.escape(_MESSAGE)}"
)


def parameter_value_spans(text):
    return [match.span(2) for match in _PARAMETER_RE.finditer(text)]


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

        Missing markers preserve the text. Invalid tool intent raises an error.
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
                raise ToolOutputError("Incomplete ATEM tool call")
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
        except ToolOutputError:
            raise
        except ValueError as error:
            raise ToolOutputError("Malformed ATEM tool call") from error
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
        remainder = _INVOKE_RE.sub("", text)
        if re.search(r"</?atem:(?:invoke|parameter)\b", remainder):
            raise ValueError("Incomplete invocation")
        calls = []
        for im in _INVOKE_RE.finditer(text):
            name, body = im.group(1), im.group(2)
            if name not in tools:
                matches = [key for key in tools if wire_tool_name(key) == name]
                if len(matches) == 1:
                    name = matches[0]
            if name not in tools:
                raise ToolOutputError(f"Model called undefined tool {name!r}")
            schema = ToolSchema(tools[name])
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
