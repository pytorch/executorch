# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tool-call parsing. Two formats, pick the one matching your model:

- HermesDetector: JSON inside <tool_call>…</tool_call> (Qwen2.5/3, Hermes).
- QwenFunctionCallDetector: Qwen XML <function=…><parameter=…> (Qwen3.5-MoE /
  Qwen3-Coder).

The server buffers the model's full output and parses it once into complete
OpenAI tool_calls; parse failures degrade to visible text.
"""

from .hermes import HermesDetector
from .qwen import QwenFunctionCallDetector
from .types import ParseResult, ToolCallItem

__all__ = [
    "HermesDetector",
    "QwenFunctionCallDetector",
    "ParseResult",
    "ToolCallItem",
]
