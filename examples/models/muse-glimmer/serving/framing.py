# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipient headers change channels only after the current message ends."""

import re
from bisect import bisect_left

IDENT = r"[a-zA-Z0-9_-]+"
RECIPIENT = rf"{IDENT}(?:\.{IDENT})*"
HEADER_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:<\|start\|>assistant)?\s?to=({RECIPIENT})"
    rf"(?: constrain={IDENT})?<\|message\|>"
)
ASSISTANT_HEADER = "<|start|>assistant"
CONTROLS = tuple(
    f"<|{name}|>"
    for name in ("start", "message", "channel", "constrain", "end", "eom", "eot")
)
CONTROL_RE = re.compile("|".join(map(re.escape, CONTROLS)))
END_RE = re.compile(r"<\|eo[mt]\|>\s*$")
_BODY_CONTROL_RE = re.compile(HEADER_RE.pattern + "|" + CONTROL_RE.pattern)


class FenceTracker:
    """Recognize whole Markdown fence lines before interpreting their controls."""

    def __init__(self):
        self.marker = None
        self.width = 3
        self.new_body()

    @property
    def in_fence(self):
        return self.marker is not None

    def new_body(self):
        self.indent = 0

    def needs_line(self, text, pos, final):
        if final or self.indent > 3 or text[pos] not in "`~":
            return False
        end = pos
        while end < len(text) and text[end] == text[pos]:
            end += 1
        if end - pos < 3 and end < len(text):
            return False
        return "\n" not in text[end:] and re.search(r"<\|eo[mt]\|>", text[end:]) is None

    def scan(self, text, start, end):
        for pos in range(start, end):
            char = text[pos]
            if char == "\n":
                self.new_body()
            elif char in " \t" and self.indent <= 3:
                self.indent += 1 if char == " " else 4 - self.indent % 4
            elif self.indent <= 3 and char in "`~":
                self._line(text, pos)
                self.indent = 4
            else:
                self.indent = 4

    def _line(self, text, pos):
        end = text.find("\n", pos)
        line = text[pos : end if end >= 0 else len(text)]
        run = len(line) - len(line.lstrip(line[0]))
        suffix = re.split(r"<\|eo[mt]\|>", line[run:], maxsplit=1)[0]
        if self.marker is None:
            if run >= 3 and (line[0] == "~" or "`" not in suffix):
                self.marker, self.width = line[0], run
        elif line[0] == self.marker and run >= self.width and not suffix.strip(" \t\r"):
            self.marker = None


def wire_tool_name(name: str) -> str:
    return name if "." in name else f"{name}.{name}"


def _masked_text(text, spans, start=0, end=None):
    end = len(text) if end is None else end
    pieces, offset = [], start
    index = max(0, bisect_left(spans, (start,)) - 1)
    while index < len(spans) and spans[index][0] < end:
        begin, stop = spans[index]
        begin, stop = max(begin, start), min(stop, end)
        if stop > begin:
            pieces.extend((text[offset:begin], " " * (stop - begin)))
            offset = stop
        index += 1
    pieces.append(text[offset:end])
    return "".join(pieces)


def body_spans(text, start=0, end=None, protected_spans=()):
    """Source ranges retained in a body, preserving quoted controls and headers."""
    visible = _masked_text(text, protected_spans, start, end)
    cursor = scanned = 0
    fences = FenceTracker()
    for match in _BODY_CONTROL_RE.finditer(visible):
        fences.scan(visible, scanned, match.start())
        scanned = match.end()
        quoted = match.group(1) or fences.in_fence
        fences.scan(visible, match.start(), match.end())
        if quoted:
            continue
        if cursor < match.start():
            yield start + cursor, start + match.start()
        cursor = match.end()
    if cursor < len(visible):
        yield start + cursor, start + len(visible)


def clean_body(text):
    return "".join(text[start:end] for start, end in body_spans(text))


def recipient_headers(text, protected_spans=()):
    visible = _masked_text(text, protected_spans)
    previous = None
    scanned = 0
    fences = FenceTracker()
    for header in HEADER_RE.finditer(visible):
        fences.scan(visible, scanned, header.start())
        scanned = header.end()
        if fences.in_fence:
            fences.scan(visible, header.start(), header.end())
            continue
        if previous is not None:
            end = header.start()
            while end > previous.end() and text[end - 1].isspace():
                end -= 1
            if not text.endswith(("<|eom|>", "<|eot|>"), previous.end(), end):
                fences.scan(visible, header.start(), header.end())
                continue
        fences.new_body()
        yield header
        previous = header


def recipient_bodies(text, protected_spans=()):
    headers = list(recipient_headers(text, protected_spans))
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        terminator = END_RE.search(text, header.end(), end)
        if terminator is not None:
            end = terminator.start()
        yield header, header.end(), end
