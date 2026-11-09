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

from .framing import (
    ASSISTANT_HEADER,
    CONTROL_RE,
    CONTROLS,
    FenceTracker,
    HEADER_RE,
    IDENT,
    RECIPIENT,
)

from .tool_parsers.atem import _OPEN_RE, _PARAMETER_END_RE

_IDENT = re.compile(IDENT)
_RECIPIENT = re.compile(RECIPIENT)
_BOUNDARY = re.compile(r"(?<![A-Za-z0-9_])")


def _header_prefix(text: str, pos: int) -> bool:
    """Recognize only unfinished prefixes of the existing header grammar."""
    if text[pos : pos + 1] == "<":
        if len(text) - pos < len(ASSISTANT_HEADER):
            return ASSISTANT_HEADER.startswith(text[pos:])
        if not text.startswith(ASSISTANT_HEADER, pos):
            return False
        pos += len(ASSISTANT_HEADER)
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


class _OpeningTag:
    """Track an unfinished ATEM name attribute without rescanning its value."""

    def __init__(self):
        self.phase = "space"
        self.attribute_pos = 0

    def pending(self, text):
        for char in text:
            if self.phase == "space":
                if not char.isspace():
                    return False
                self.phase = "attribute"
            elif self.phase == "attribute":
                if self.attribute_pos == 0 and char.isspace():
                    continue
                if char != 'name="'[self.attribute_pos]:
                    return False
                self.attribute_pos += 1
                if self.attribute_pos == len('name="'):
                    self.phase = "value"
            elif self.phase == "value":
                if char == '"':
                    self.phase = "end"
            elif not char.isspace():
                return False
        return True


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
        self._has_header = False
        self._header_allowed = True
        self._fences = FenceTracker()
        self._in_invocation = False
        self._wait_for = None
        self._held = []
        self._wait_tail = ""
        self._opening = None

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

    def _ready(self, text):
        if self._wait_for == "opening":
            assert self._opening is not None
            return not self._opening.pending(text)
        tail = self._wait_tail + text
        if self._wait_for == "line":
            self._wait_tail = tail[-8:]
            return "\n" in tail or re.search(r"<\|eo[mt]\|>", tail) is not None
        if re.search(r"</atem:parameter>\s*(?=<atem:parameter|</atem:invoke>)", tail):
            return True
        marker = "</atem:parameter>"
        cut = tail.rfind(marker)
        suffix = tail[cut + len(marker) :].lstrip() if cut >= 0 else None
        if suffix is not None and any(
            token.startswith(suffix) for token in ("<atem:parameter", "</atem:invoke>")
        ):
            self._wait_tail = marker + suffix
        else:
            self._wait_tail = tail[-len(marker) :]
        return False

    def feed(self, text: str) -> Iterable[DeltaMessage]:
        if self._wait_for is not None:
            self._held.append(text)
            if not self._ready(text):
                return []
            self._pending = "".join(self._held)
            self._held = []
            self._wait_for = None
        else:
            self._pending += text
        result = self._drain(final=False)
        if self._wait_for is not None:
            self._held = [self._pending]
            self._pending = ""
            self._wait_tail = ""
            if self._wait_for != "opening":
                self._ready(self._held[0])
        return result

    def finish(self) -> Iterable[DeltaMessage]:
        if self._wait_for is not None:
            self._pending = "".join(self._held)
            self._held = []
            self._wait_for = None
        return self._drain(final=True) + self._close_block()

    def _atem(self, text, pos, final):
        if text[pos] != "<":
            return pos
        for marker in ("<atem:invoke", "<atem:parameter", "</atem:invoke>"):
            if (
                not final
                and len(text) - pos < len(marker)
                and marker.startswith(text[pos:])
            ):
                return None
        opening = _OPEN_RE.match(text, pos)
        if opening is not None:
            if opening.group(1) == "invoke":
                self._in_invocation = True
                self._fences.scan(text, pos, opening.end())
                return opening.end()
            if self._in_invocation:
                close = _PARAMETER_END_RE.search(text, opening.end())
                if close is not None and (final or close.end() < len(text)):
                    self._fences.scan(text, pos, opening.end())
                    self._fences.scan(text, close.start(), close.end())
                    return close.end()
                if not final:
                    self._wait_for = "parameter"
                    return None
                self._fences.scan(text, pos, opening.end())
                return len(text)
        elif not final and self._hold_opening(text, pos):
            return None
        if text.startswith("</atem:invoke>", pos):
            self._in_invocation = False
        return pos

    def _hold_opening(self, text, pos):
        for marker in ("<atem:invoke", "<atem:parameter"):
            if text.startswith(marker, pos):
                self._opening = _OpeningTag()
                if self._opening.pending(text[pos + len(marker) :]):
                    self._wait_for = "opening"
                    return True
        return False

    def _drain(self, *, final: bool) -> list[DeltaMessage]:
        # Retain one source character for the grammar's negative lookbehind.
        text = self._previous + self._pending
        pos = body_start = len(self._previous)
        result = []
        while pos < len(text):
            if self._fences.needs_line(text, pos, final):
                self._wait_for = "line"
                break
            atem = self._atem(text, pos, final)
            if atem is None:
                break
            if atem > pos:
                result.extend(self._body(text[body_start:atem]))
                self._header_allowed = False
                pos = body_start = atem
                continue
            header = HEADER_RE.match(text, pos)
            if header:
                if self._fences.in_fence or (
                    self._has_header and not self._header_allowed
                ):
                    self._fences.scan(text, pos, header.end())
                    pos = header.end()
                    continue
                result.extend(self._body(text[body_start:pos]))
                result.extend(self._close_block())
                self._field = (
                    "reasoning_content" if header.group(1) == "self" else "content"
                )
                self._fences.new_body()
                self._has_header = True
                self._header_allowed = False
                pos = body_start = header.end()
                continue
            potential = _BOUNDARY.match(text, pos) and _header_prefix(text, pos)
            control = CONTROL_RE.match(text, pos)
            partial_control = (
                control is None
                and text[pos] == "<"
                and any(
                    len(text) - pos < len(token) and token.startswith(text[pos:])
                    for token in CONTROLS
                )
            )
            if potential or partial_control:
                if not final:
                    break
                suffix = text[pos:].lstrip()
                if suffix.startswith("<|") and (
                    partial_control or not self._has_header or self._header_allowed
                ):
                    result.extend(self._body(text[body_start:pos]))
                    pos = body_start = len(text)
                    break
            if control:
                body_start, deltas = self._control(text, pos, body_start, control)
                result.extend(deltas)
                pos = control.end()
            else:
                self._fences.scan(text, pos, pos + 1)
                if not text[pos].isspace():
                    self._header_allowed = False
                pos += 1
        result.extend(self._body(text[body_start:pos]))
        self._pending = text[pos:]
        self._previous = text[pos - 1 : pos] if pos else ""
        return result

    def _control(self, text, pos, body_start, control):
        quoted = self._fences.in_fence
        self._fences.scan(text, pos, control.end())
        if quoted:
            return body_start, []
        result = self._body(text[body_start:pos])
        if control.group() in ("<|eom|>", "<|eot|>"):
            self._after_end = True
            self._header_allowed = True
        else:
            self._header_allowed = False
        return control.end(), result


class MuseGlimmerToolReasoningParser:
    """Stream only reasoning before the first ATEM block; never parse its values."""

    def __init__(self):
        self._parser = MuseGlimmerStreamParser()
        self._pending = ""
        self._finished = False
        self._marker = "<atem:"

    def feed(self, text):
        if self._finished:
            return []
        self._pending += text
        cut = self._pending.find(self._marker)
        if cut >= 0:
            result = list(self._parser.feed(self._pending[:cut]))
            self._finished = True
            self._pending = ""
            return result
        hold = len(self._marker) - 1
        if len(self._pending) <= hold:
            return []
        text, self._pending = self._pending[:-hold], self._pending[-hold:]
        return self._parser.feed(text)

    def finish(self):
        if self._finished:
            return []
        self._finished = True
        return list(self._parser.feed(self._pending)) + list(self._parser.finish())
