# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incremental MG framing. Truncated <| headers are suppressed at EOF,
unlike the legacy whole-text fallback; ambiguous bare to= suffixes remain text.
Known standalone controls are suppressed, including in reasoning; unknown tokens
remain body text. Control-only bodies do not add separators between blocks.
No model stop is inferred.
"""

import re
from collections.abc import Iterable

from executorch.examples.llm_server.python.protocol import DeltaMessage

from .serve import (
    _ASSISTANT_HEADER,
    _MUSE_GLIMMER_ADDRESSED_HEADER_RE,
    _MUSE_GLIMMER_IDENT,
    _MUSE_GLIMMER_RECIPIENT,
    _MUSE_GLIMMER_STRAY_SPECIAL_RE,
)

_IDENT = re.compile(_MUSE_GLIMMER_IDENT)
_RECIPIENT = re.compile(_MUSE_GLIMMER_RECIPIENT)
_BOUNDARY = re.compile(r"(?<![A-Za-z0-9_])")
_CONTROLS = tuple(
    f"<|{name}|>"
    for name in ("start", "message", "channel", "constrain", "end", "eom", "eot")
)


def _header_prefix(text: str, pos: int) -> bool:
    """Recognize only unfinished prefixes of the existing header grammar."""
    if text[pos : pos + 1] == "<":
        if len(text) - pos < len(_ASSISTANT_HEADER):
            return _ASSISTANT_HEADER.startswith(text[pos:])
        if not text.startswith(_ASSISTANT_HEADER, pos):
            return False
        pos += len(_ASSISTANT_HEADER)
    if text[pos : pos + 1].isspace():
        pos += 1
    for literal, pattern in (
        ("to=", _RECIPIENT),
        (" constrain=", _IDENT),
        ("<|message|>", None),
    ):
        if literal == " constrain=" and text[pos : pos + 1] != " ":
            continue
        if len(text) - pos < len(literal):
            return literal.startswith(text[pos:])
        if not text.startswith(literal, pos):
            return False
        pos += len(literal)
        if pattern is not None:
            match = pattern.match(text, pos)
            if match is None:
                return pos == len(text)
            pos = match.end()
            if pattern is _RECIPIENT and pos == len(text) - 1 and text[pos] == ".":
                return True
    return pos == len(text)


class MuseGlimmerStreamParser:
    """Keep only unresolved framing and boundary whitespace, never prior bodies."""

    def __init__(self) -> None:
        self._pending = ""
        self._previous = ""
        self._field = "content"
        self._seen: set[str] = set()
        self._in_block = False
        self._space = ""
        self._after_end = False
        self._end_space = ""

    def _body(self, text: str) -> list[DeltaMessage]:
        # Post-end whitespace is framing only if a header or EOF follows it.
        if self._after_end:
            text = self._end_space + text
            if not text.strip():
                self._end_space = text
                return []
            self._after_end = False
            self._end_space = ""
        text = self._space + text
        stable = text.rstrip()
        self._space = text[len(stable) :]
        if not stable:
            return []
        if not self._in_block:
            if self._field == "content":
                stable = stable.lstrip()
            if self._field in self._seen:
                stable = (
                    "\n" if self._field == "reasoning_content" else "\n\n"
                ) + stable
            self._seen.add(self._field)
            self._in_block = True
        return [DeltaMessage(**{self._field: stable})]

    def _close_block(self) -> list[DeltaMessage]:
        result = []
        if self._field == "reasoning_content" and self._in_block and self._space:
            result.append(DeltaMessage(reasoning_content=self._space))
        self._space = self._end_space = ""
        self._after_end = self._in_block = False
        return result

    def feed(self, text: str) -> Iterable[DeltaMessage]:
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> Iterable[DeltaMessage]:
        return self._drain(final=True) + self._close_block()

    def _drain(self, *, final: bool) -> list[DeltaMessage]:
        # Retain one source character for the grammar's negative lookbehind.
        text = self._previous + self._pending
        pos = body_start = len(self._previous)
        result = []
        while pos < len(text):
            header = _MUSE_GLIMMER_ADDRESSED_HEADER_RE.match(text, pos)
            if header:
                result.extend(self._body(text[body_start:pos]))
                result.extend(self._close_block())
                self._field = (
                    "reasoning_content" if header.group(1) == "self" else "content"
                )
                pos = body_start = header.end()
                continue
            potential = _BOUNDARY.match(text, pos) and _header_prefix(text, pos)
            control = _MUSE_GLIMMER_STRAY_SPECIAL_RE.match(text, pos)
            partial_control = (
                control is None
                and text[pos] == "<"
                and any(
                    len(text) - pos < len(token) and token.startswith(text[pos:])
                    for token in _CONTROLS
                )
            )
            if potential or partial_control:
                if not final:
                    break
                suffix = text[pos:].lstrip()
                if suffix.startswith("<|"):
                    result.extend(self._body(text[body_start:pos]))
                    pos = body_start = len(text)
                    break
            if control:
                result.extend(self._body(text[body_start:pos]))
                if control.group() in ("<|eom|>", "<|eot|>"):
                    self._after_end = True
                pos = body_start = control.end()
            else:
                pos += 1
        result.extend(self._body(text[body_start:pos]))
        self._pending = text[pos:]
        self._previous = text[pos - 1 : pos] if pos else ""
        return result
