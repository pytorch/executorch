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
END_RE = re.compile(r"<\|eo[mt]\|>\s*$")
_BODY_CONTROL_RE = re.compile(
    HEADER_RE.pattern + r"|<\|(?:start|message|channel|constrain|end|eom|eot)\|>"
)


def clean_body(text):
    # Complete headers inside a body are quoted text, not stray controls.
    return _BODY_CONTROL_RE.sub(lambda m: m.group() if m.group(1) else "", text)


def recipient_headers(text, protected_spans=()):
    spans = iter(protected_spans)
    span = next(spans, None)
    previous = None
    for header in HEADER_RE.finditer(text):
        while span is not None and span[1] <= header.start():
            span = next(spans, None)
        if span is not None and span[0] <= header.start():
            continue
        if previous is None:
            if text[: header.start()].count("```") % 2:
                continue
        else:
            end = header.start()
            while end > previous.end() and text[end - 1].isspace():
                end -= 1
            if not text.endswith(("<|eom|>", "<|eot|>"), previous.end(), end):
                continue
        yield header
        previous = header
