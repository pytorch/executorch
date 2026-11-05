# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipient headers change channels only after the current message ends."""

import re

IDENT = r"[a-zA-Z_][a-zA-Z0-9_-]*"
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
    """Track line-start Markdown fences across chunks and message boundaries."""

    def __init__(self):
        self.marker = None
        self.width = 3
        self.new_body()

    @property
    def in_fence(self):
        return self.marker is not None

    def new_body(self):
        self.indent = 0
        self.run = 0
        self.character = None
        self.opening = self.closed = False

    def feed(self, text):
        for char in text:
            if char == "\n":
                self.new_body()
            elif char in " \t" and self.run == 0:
                self.indent += 1
            elif self.indent <= 3 and char in "`~" and self.character in (None, char):
                self.character = char
                self.run += 1
                if self.opening:
                    self.width = self.run
                elif not self.closed and self.marker is None and self.run == 3:
                    self.marker, self.width, self.opening = char, 3, True
                elif self.marker == char and self.run >= self.width:
                    self.marker, self.closed = None, True
            else:
                self.indent = 4


def wire_tool_name(name: str) -> str:
    return name if "." in name else f"{name}.{name}"


def body_spans(text, start=0, end=None):
    """Source ranges retained in a body, preserving quoted controls and headers."""
    end = len(text) if end is None else end
    cursor = scanned = start
    fences = FenceTracker()
    for match in _BODY_CONTROL_RE.finditer(text, start, end):
        fences.feed(text[scanned : match.start()])
        scanned = match.end()
        quoted = match.group(1) or fences.in_fence
        fences.feed(match.group())
        if quoted:
            continue
        if cursor < match.start():
            yield cursor, match.start()
        cursor = match.end()
    if cursor < end:
        yield cursor, end


def clean_body(text):
    return "".join(text[start:end] for start, end in body_spans(text))


def recipient_headers(text, protected_spans=()):
    # Mask tool arguments/consumed blocks without shifting source coordinates.
    pieces, offset = [], 0
    for start, end in protected_spans:
        pieces.extend((text[offset:start], " " * (end - start)))
        offset = end
    pieces.append(text[offset:])
    visible = "".join(pieces)
    previous = None
    scanned = 0
    fences = FenceTracker()
    for header in HEADER_RE.finditer(visible):
        fences.feed(visible[scanned : header.start()])
        scanned = header.end()
        if fences.in_fence:
            fences.feed(header.group())
            continue
        if previous is not None:
            end = header.start()
            while end > previous.end() and text[end - 1].isspace():
                end -= 1
            if not text.endswith(("<|eom|>", "<|eot|>"), previous.end(), end):
                fences.feed(header.group())
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
