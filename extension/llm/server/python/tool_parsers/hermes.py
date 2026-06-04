# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hermes-style tool calls: <tool_call>{"name": ..., "arguments": {...}}</tool_call>.

Used by Qwen2.5/Qwen3 (and Hermes models); the Qwen XML format is handled
separately by QwenFunctionCallDetector. The server buffers a model's full output
and parses it once into complete OpenAI tool_calls (no partial-fragment
streaming). Parse failures fall back to visible text — never a crash or a silent
drop.
"""

import json
import logging
import re
from typing import Any, Optional

from .types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class _UndefinedToolCall(Exception):
    """A <tool_call> named a tool not in the request's `tools`. v1 degrades the
    WHOLE response to visible text rather than emitting a partial set — never
    silently drop an undefined call while keeping its siblings (spec)."""


class HermesDetector:
    """Parses Hermes/Qwen tool calls. Create a fresh instance per request (it
    holds the per-request tool-call index); never share across requests."""

    bot_token = "<tool_call>"

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(self, text: str, tool_names: set[str]) -> ParseResult:
        """Return leading text + any complete tool calls. On no call or a parse
        failure, return the original text unchanged (kept visible to the client)."""
        if self.bot_token not in text:
            return ParseResult(normal_text=text)
        normal = text[: text.find(self.bot_token)].strip()
        try:
            calls = self._parse_calls(text, tool_names)
        except _UndefinedToolCall as e:
            # Degrade the whole response to visible text so the undefined call
            # isn't silently dropped (and its valid siblings aren't executed in
            # isolation, losing the model's full intent).
            logger.debug("undefined tool %s; returning raw text (no partial calls)", e)
            return ParseResult(normal_text=text)
        except Exception as e:  # noqa: BLE001 - never crash; fall back to visible text
            logger.debug("tool parse failed (%s); returning raw text", e)
            return ParseResult(normal_text=text)
        if not calls:
            return ParseResult(normal_text=text)
        return ParseResult(normal_text=normal, calls=calls)

    def _parse_calls(self, text: str, tool_names: set[str]) -> list[ToolCallItem]:
        calls = []
        for raw in _CALL_RE.findall(text):
            if not raw.strip():
                continue
            obj = json.loads(raw.strip())
            for entry in obj if isinstance(obj, list) else [obj]:
                calls.append(
                    self._make_item(
                        entry.get("name"),
                        entry.get("arguments", entry.get("parameters")),
                        tool_names,
                    )
                )
        return calls

    def _make_item(
        self, name: Optional[str], arguments: Any, tool_names: set[str]
    ) -> ToolCallItem:
        if not name or name not in tool_names:
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
