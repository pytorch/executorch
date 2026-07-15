# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermes-style tool calls: <tool_call>{"name": ..., "arguments": {...}}</tool_call>.

Used by Qwen2.5/Qwen3 (and Hermes models); the Qwen XML format is handled
separately by QwenFunctionCallDetector. The server buffers a model's full output
and parses it once into complete OpenAI tool_calls (no partial-fragment
streaming).

Malformed-call policy (explicit): ALL-OR-NOTHING. If any <tool_call> block is
malformed, names an undefined tool, or is truncated/unclosed, the WHOLE response
degrades -- no partial call set is emitted. When it degrades, the raw
<tool_call>...</tool_call> markup is NOT leaked into visible content: only the
leading text before the first marker is returned. Never crashes.
"""

import json
import logging
from typing import Any, Optional

from .types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)

_BOT = "<tool_call>"
_EOT = "</tool_call>"
_DECODER = json.JSONDecoder()


class _UndefinedToolCall(Exception):
    """A <tool_call> named a tool not in the request's `tools`. Degrades the
    WHOLE response to visible text rather than emitting a partial set — never
    silently drop an undefined call while keeping its siblings (spec)."""


class HermesDetector:
    """Parses Hermes/Qwen tool calls. Create a fresh instance per request (it
    holds the per-request tool-call index); never share across requests."""

    bot_token = "<tool_call>"

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(self, text: str, tools: dict[str, dict]) -> ParseResult:
        """Return leading text + any complete tool calls. On no/undefined/
        malformed call, degrade to the leading text BEFORE the first marker --
        never leak the raw <tool_call> markup into visible content."""
        if _BOT not in text:
            return ParseResult(normal_text=text)
        # Leading text before the first marker; this is what we show if parsing
        # degrades, so the structural markup is never surfaced to the client.
        normal = text[: text.find(_BOT)].strip()
        try:
            calls = self._parse_calls(text, tools)
        except _UndefinedToolCall as e:
            # Well-formed call to an undefined tool: degrade the WHOLE response to
            # visible text (surface the model's intent; never emit a partial set).
            logger.debug("undefined tool %s; returning raw text (no partial calls)", e)
            return ParseResult(normal_text=text)
        except Exception as e:  # noqa: BLE001 - never crash
            # Genuinely malformed / truncated / unclosed markup: degrade to the
            # leading text so the partial <tool_call> garbage is NOT surfaced.
            logger.debug("malformed tool call (%s); degrading to leading text", e)
            return ParseResult(normal_text=normal)
        if not calls:
            return ParseResult(normal_text=text)
        return ParseResult(normal_text=normal, calls=calls)

    def _parse_calls(self, text: str, tools: dict[str, dict]) -> list[ToolCallItem]:
        """All-or-nothing: any malformed/unclosed block raises (caller degrades).

        Each block's JSON is parsed with raw_decode rather than a non-greedy
        regex, so a string value that itself contains '</tool_call>' does not
        truncate the captured JSON. The block must be closed by '</tool_call>'.
        """
        calls = []
        pos = 0
        while True:
            start = text.find(_BOT, pos)
            if start == -1:
                break
            s = start + len(_BOT)
            while s < len(text) and text[s].isspace():
                s += 1
            obj, end = _DECODER.raw_decode(text, s)  # JSONDecodeError -> degrade
            close = text.find(_EOT, end)
            if close == -1 or text[end:close].strip():
                raise ValueError("unclosed or trailing-garbage <tool_call> block")
            pos = close + len(_EOT)
            for entry in obj if isinstance(obj, list) else [obj]:
                if not isinstance(entry, dict):
                    raise ValueError("tool call entry is not an object")
                # `parameters` is the fallback ONLY when `arguments` is absent or
                # explicitly null (get-with-default misses the explicit-null case).
                args = entry.get("arguments")
                if args is None:
                    args = entry.get("parameters")
                calls.append(self._make_item(entry.get("name"), args, tools))
        return calls

    def _make_item(
        self, name: Optional[str], arguments: Any, tools: dict[str, dict]
    ) -> ToolCallItem:
        if not name or name not in tools:
            raise _UndefinedToolCall(repr(name))
        item = ToolCallItem(
            tool_index=self._next_index,
            name=name,
            arguments=json.dumps(
                arguments if arguments is not None else {}, ensure_ascii=False
            ),
        )
        self._next_index += 1
        return item
