# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tool-call parsing. Hermes/Qwen format only (<tool_call>...</tool_call>).

The server buffers the model's full output and parses it once into complete
OpenAI tool_calls; parse failures degrade to visible text.
"""

from .hermes import HermesDetector
from .types import ParseResult, ToolCallItem

__all__ = ["HermesDetector", "ParseResult", "ToolCallItem"]
