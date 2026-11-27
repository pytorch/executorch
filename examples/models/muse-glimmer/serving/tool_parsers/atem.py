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
from dataclasses import dataclass, field
from typing import Optional

from executorch.examples.llm_server.python.tool_parsers.types import (
    ParseResult,
    ToolCallItem,
)

from ..framing import _masked_text, FenceTracker, recipient_headers, wire_tool_name
from .schema import ToolOutputBudget, ToolOutputError, ToolSchema

_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_INVOKE_OPEN = "<atem:invoke"

# Tag discovery never scans an invocation body with a backtracking regex.
_TAG_RE = re.compile(r"</?atem:(?:function_calls|invoke|parameter)\b")
_OPEN_RE = re.compile(r'<atem:(invoke|parameter)\s+name="([^"]*)"\s*>')
_PARAMETER_END_RE = re.compile(
    r"</atem:parameter>\s*(?=<atem:parameter|</atem:invoke>|\Z)"
)
_TERMINATOR_RE = re.compile(r"\s*<\|eo[mt]\|>")


@dataclass
class _Invocation:
    start: int
    name: str | None
    body: int
    end: int | None = None
    parameters: list = field(default_factory=list)
    malformed: bool = False


class _Syntax:
    def __init__(self, text):
        self.invocations = {}
        self.markers = []
        self.wrapper_ends = {}
        self.nested_wrappers = set()
        self._wrappers = []
        self.parameters = []
        self.text = text
        self._parameter_ends = list(_PARAMETER_END_RE.finditer(text))
        self._close_index = 0
        self._active = None
        self._cursor = 0
        skip = 0
        for tag in _TAG_RE.finditer(text):
            if tag.start() < skip:
                continue
            skip = self._parameter(tag)
            if not skip:
                self._record_tag(tag)

    def _parameter(self, tag):
        active = self._active
        if tag.group() != "<atem:parameter" or active is None:
            return 0
        opening = _OPEN_RE.match(self.text, tag.start())
        closes = self._parameter_ends
        while (
            self._close_index < len(closes)
            and closes[self._close_index].start() < tag.start()
        ):
            self._close_index += 1
        if opening is None:
            return 0
        if self._close_index == len(closes):
            self.parameters.append((opening.end(), len(self.text)))
            return len(self.text)
        close = closes[self._close_index]
        active.malformed |= bool(self.text[self._cursor : tag.start()].strip())
        active.parameters.append((opening.group(2), opening.end(), close.start()))
        self.parameters.append((opening.end(), close.start()))
        self._cursor = close.end()
        return close.end()

    def _record_tag(self, tag):
        start = tag.start()
        if tag.group() == "<atem:invoke":
            self.markers.append(start)
            if self._active is not None:
                self._active.malformed = True
            opening = _OPEN_RE.match(self.text, start)
            self._active = _Invocation(
                start,
                opening.group(2) if opening else None,
                opening.end() if opening else tag.end(),
            )
            self.invocations[start] = self._active
            self._cursor = self._active.body
        elif self.text.startswith("</atem:invoke>", start) and self._active is not None:
            self._active.malformed |= bool(self.text[self._cursor : start].strip())
            self._active.end = start + len("</atem:invoke>")
            self._active = None
        elif self.text.startswith("<atem:function_calls>", start):
            self.markers.append(start)
            if self._wrappers:
                self.nested_wrappers.add(self._wrappers[0])
            self._wrappers.append(start)
        elif self.text.startswith("</atem:function_calls>", start):
            if self._wrappers:
                self.wrapper_ends[self._wrappers.pop()] = start + len(
                    "</atem:function_calls>"
                )
            self._active = None

    def block_end(self, start):
        if self.text.startswith(_INVOKE_OPEN, start):
            return self.invocations[start].end
        return self.wrapper_ends.get(start)


def _tool_blocks(text, syntax=None):
    syntax = syntax if syntax is not None else _Syntax(text)
    visible = _masked_text(text, syntax.parameters)
    headers = iter(recipient_headers(visible))
    following = next(headers, None)
    header = None
    offset = scanned = 0
    excluded_until = 0
    fences = FenceTracker()
    line_text = header_text = False
    for start in syntax.markers:
        if start < max(offset, excluded_until):
            continue
        prefix = visible[scanned:start]
        line_text = (line_text if "\n" not in prefix else False) or bool(
            prefix.rsplit("\n", 1)[-1].strip()
        )
        prefix_start = scanned
        while following is not None and following.end() <= start:
            header = following
            header_text = False
            prefix_start = header.end()
            following = next(headers, None)
            fences = FenceTracker()
        header_text |= bool(visible[prefix_start:start].strip())
        for terminator in re.finditer(r"<\|eo[mt]\|>", visible[prefix_start:start]):
            cut = prefix_start + terminator.start()
            fences.scan(visible, max(scanned, prefix_start), cut)
            if not fences.in_fence:
                header = None
            scanned = prefix_start + terminator.end()
        fences.scan(visible, max(scanned, prefix_start), start)
        scanned = start
        if header is not None and header.start() < offset:
            header = None
        adjacent = header is not None and not header_text
        if (
            fences.in_fence
            or (header is not None and header.group(1) in ("self", "user"))
            or (not adjacent and line_text)
        ):
            # An ineligible wrapper owns its children, including bare invokes.
            if text.startswith(_FUNCTION_CALLS_OPEN, start):
                excluded_until = syntax.block_end(start) or len(text)
            continue
        end = syntax.block_end(start)
        yield start, end, header.start() if adjacent else start
        if end is None:
            return
        offset = scanned = end
        fences = FenceTracker()
        line_text = header_text = False
        header = None


def tool_start(text):
    block = next(_tool_blocks(text), None)
    return block[0] if block is not None else None


def parameter_value_spans(text):
    if "<atem:" not in text:
        return []
    return _Syntax(text).parameters


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
        self, text: str, tools: dict, *, finish_reason=None, truncated=False
    ) -> ParseResult:
        normal, calls = [], []
        offset = 0
        incomplete = False
        budget = ToolOutputBudget()
        syntax = _Syntax(text)
        invocations = iter(syntax.invocations.values())
        invocation = next(invocations, None)
        try:
            for start, end, header in _tool_blocks(text, syntax):
                normal.append(text[offset:header])
                offset = end or len(text)
                # Consume only framing belonging to this call block.
                terminator = _TERMINATOR_RE.match(text, offset)
                if terminator:
                    offset = terminator.end()
                if end is None:
                    raise IncompleteToolOutputError("Incomplete ATEM tool call")
                if start in syntax.nested_wrappers:
                    raise ToolOutputError("Nested ATEM call wrappers are malformed")
                selected = []
                while invocation is not None and invocation.start < end:
                    if invocation.start >= start:
                        selected.append(invocation)
                    invocation = next(invocations, None)
                calls.extend(
                    self._parse_calls(text, start, end, selected, tools, budget=budget)
                )
        except IncompleteToolOutputError:
            if finish_reason != "length" and not truncated:
                raise
            calls = []
            incomplete = True
        except (ValueError, RecursionError) as error:
            raise ToolOutputError("Malformed ATEM tool call") from error
        normal.append(text[offset:])
        return ParseResult(
            normal_text="".join(normal).strip() if calls or incomplete else text,
            calls=calls,
            incomplete=incomplete,
        )

    def _parse_calls(self, text, start, end, invocations, tools, *, budget):
        if not invocations:
            raise ToolOutputError("Closed ATEM block has no complete invocation")
        calls = []
        cursor = start
        for invocation in invocations:
            gap = text[cursor : invocation.start].replace(_FUNCTION_CALLS_OPEN, "")
            if "atem:" in gap or invocation.end is None or invocation.malformed:
                raise ToolOutputError("Malformed invocation in closed ATEM block")
            cursor = invocation.end
            name = invocation.name
            if name not in tools:
                matches = [key for key in tools if wire_tool_name(key) == name]
                if len(matches) == 1:
                    name = matches[0]
            if name not in tools:
                raise ToolOutputError(f"Model called undefined tool {name!r}")
            schema = tools[name]
            if not isinstance(schema, ToolSchema):
                raise TypeError("ATEM requires precompiled ToolSchema instances")
            args = {}
            for key, begin, stop in invocation.parameters:
                if key in args:
                    raise ValueError("Duplicate parameter")
                args[key] = text[begin:stop]
            args = schema.coerce_arguments(name, args, budget=budget)
            calls.append(self._make_item(name, args, tools))
        if "atem:" in text[cursor:end].replace("</atem:function_calls>", ""):
            raise ToolOutputError("Malformed invocation in closed ATEM block")
        return calls

    def _make_item(
        self, name: Optional[str], arguments: dict, tools: dict[str, ToolSchema]
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
