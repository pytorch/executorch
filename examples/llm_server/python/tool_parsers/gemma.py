# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma tool calls: <|tool_call>call:NAME{key:value}<tool_call|>."""

import json
import logging
import math
import re
from typing import Any, Optional

from .types import ParseResult, ToolCallItem

logger = logging.getLogger(__name__)

_BOT = "<|tool_call>"
_EOT = "<tool_call|>"
_QUOTE = '<|"|>'
_INT_RE = re.compile(r"[+-]?[0-9]+$")
_NUM_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


class _UndefinedToolCall(Exception):
    pass


class _Bare(str):
    __slots__ = ()


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def _skip_ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def _expect(self, token: str) -> None:
        self._skip_ws()
        if not self.text.startswith(token, self.i):
            raise ValueError(f"expected {token!r}")
        self.i += len(token)

    def parse_call(self) -> tuple[str, dict[str, Any]]:
        self._expect("call:")
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in "{ \t\r\n":
            self.i += 1
        name = self.text[start : self.i].strip()
        if not name:
            raise ValueError("missing tool name")
        args = self._parse_object()
        self._skip_ws()
        if self.i != len(self.text):
            raise ValueError("trailing garbage")
        return name, args

    def _parse_object(self) -> dict[str, Any]:
        self._expect("{")
        out: dict[str, Any] = {}
        self._skip_ws()
        if self.i < len(self.text) and self.text[self.i] == "}":
            self.i += 1
            return out
        while True:
            key = self._parse_key()
            self._expect(":")
            out[key] = self._parse_value()
            self._skip_ws()
            if self.i >= len(self.text):
                raise ValueError("unclosed object")
            if self.text[self.i] == "}":
                self.i += 1
                return out
            self._expect(",")

    def _parse_key(self) -> str:
        self._skip_ws()
        if self.text.startswith(_QUOTE, self.i):
            return self._parse_string()
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ": \t\r\n":
            self.i += 1
        key = self.text[start : self.i].strip()
        if not key:
            raise ValueError("missing key")
        return key

    def _parse_value(self) -> Any:
        self._skip_ws()
        if self.text.startswith(_QUOTE, self.i):
            return self._parse_string()
        if self.i < len(self.text) and self.text[self.i] == "{":
            return self._parse_object()
        if self.i < len(self.text) and self.text[self.i] == "[":
            return self._parse_array()
        return self._parse_bare()

    def _parse_string(self) -> str:
        self._expect(_QUOTE)
        end = self.text.find(_QUOTE, self.i)
        if end == -1:
            raise ValueError("unclosed string")
        value = self.text[self.i : end]
        self.i = end + len(_QUOTE)
        return value

    def _parse_array(self) -> list[Any]:
        self._expect("[")
        out = []
        self._skip_ws()
        if self.i < len(self.text) and self.text[self.i] == "]":
            self.i += 1
            return out
        while True:
            out.append(self._parse_value())
            self._skip_ws()
            if self.i >= len(self.text):
                raise ValueError("unclosed array")
            if self.text[self.i] == "]":
                self.i += 1
                return out
            self._expect(",")

    def _parse_bare(self) -> _Bare:
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ",]}":
            self.i += 1
        return _Bare(self.text[start : self.i].strip())


def _guess_scalar(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    if _INT_RE.match(raw):
        return int(raw)
    if _NUM_RE.match(raw):
        value = float(raw)
        if math.isfinite(value):
            return value
    return str(raw)


def _schema_type(schema: dict[str, Any]) -> Optional[str]:
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), typ[0] if typ else None)
    return typ


def _coerce_string(value: Any) -> Any:
    return str(value)


def _coerce_bool(value: Any) -> Any:
    low = str(value).strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return str(value)


def _coerce_int(value: Any) -> Any:
    s = str(value).strip()
    return int(s) if _INT_RE.match(s) else str(value)


def _coerce_number(value: Any) -> Any:
    s = str(value).strip()
    if _NUM_RE.match(s):
        parsed = float(s)
        if math.isfinite(parsed):
            return parsed
    return str(value)


_SCALAR_COERCERS = {
    "string": _coerce_string,
    "boolean": _coerce_bool,
    "integer": _coerce_int,
    "number": _coerce_number,
}


def _coerce(value: Any, schema: Optional[dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        props = (schema or {}).get("properties") or {}
        return {k: _coerce(v, props.get(k)) for k, v in value.items()}
    if isinstance(value, list):
        items = (schema or {}).get("items")
        item_schema = items if isinstance(items, dict) else None
        return [_coerce(v, item_schema) for v in value]
    typ = _schema_type(schema) if schema else None
    coercer = _SCALAR_COERCERS.get(typ)
    if coercer is not None:
        return coercer(value)
    return _guess_scalar(value) if isinstance(value, _Bare) else value


class GemmaToolCallDetector:
    bot_token = _BOT

    def __init__(self):
        self._next_index = 0

    def detect_and_parse(self, text: str, tools: dict[str, dict]) -> ParseResult:
        if _BOT not in text:
            return ParseResult(normal_text=text)
        normal = text[: text.find(_BOT)].strip()
        try:
            calls = self._parse_calls(text, tools)
        except _UndefinedToolCall as e:
            logger.debug("undefined tool %s; returning raw text (no partial calls)", e)
            return ParseResult(normal_text=text)
        except Exception as e:  # noqa: BLE001 - never crash
            logger.debug("malformed Gemma tool call (%s); degrading to leading text", e)
            return ParseResult(normal_text=normal)
        return (
            ParseResult(normal_text=normal, calls=calls) if calls else ParseResult(text)
        )

    def _parse_calls(self, text: str, tools: dict[str, dict]) -> list[ToolCallItem]:
        calls = []
        pos = 0
        while True:
            start = text.find(_BOT, pos)
            if start == -1:
                break
            body_start = start + len(_BOT)
            end = text.find(_EOT, body_start)
            if end == -1:
                raise ValueError("unclosed Gemma tool call")
            name, args = _Parser(text[body_start:end]).parse_call()
            if name not in tools:
                raise _UndefinedToolCall(repr(name))
            schema = tools.get(name) or {}
            item = ToolCallItem(
                tool_index=self._next_index,
                name=name,
                arguments=json.dumps(
                    _coerce(args, schema), ensure_ascii=False, allow_nan=False
                ),
            )
            self._next_index += 1
            calls.append(item)
            pos = end + len(_EOT)
        return calls
