# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protocol-agnostic tool-parsing types.

Kept independent of the OpenAI wire schema so the parser package is reusable;
serving_chat translates these into OpenAI tool_calls / deltas at the edge.
Design adapted from SGLang's core_types, with explicit per-request state.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCallItem:
    """A parsed tool call. `arguments` is a JSON string (the full arguments —
    this server emits complete calls, not fragments)."""

    tool_index: int
    name: Optional[str] = None
    arguments: str = ""


@dataclass
class ParseResult:
    """Outcome of a parse: free text plus any tool calls found."""

    normal_text: str = ""
    calls: list[ToolCallItem] = field(default_factory=list)
