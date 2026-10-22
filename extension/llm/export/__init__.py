# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from executorch.extension.llm.export.int4 import ExportableInt4Tensor
from executorch.extension.llm.export.mx import ExportableMXTensor
from executorch.extension.llm.export.nvfp4 import ExportableNVFP4Tensor

__all__ = [
    "ExportableGGUFTensor",
    "ExportableInt4Tensor",
    "ExportableMXTensor",
    "ExportableNVFP4Tensor",
    "LLMEdgeManager",
]


def __getattr__(name: str):
    # Lazy: importing the (lightweight) tensor subclasses above must not pull in
    # builder.py's heavy backend/quantizer dependencies. Consumers that want
    # ``LLMEdgeManager`` still get it via ``from ...export import LLMEdgeManager``.
    if name == "LLMEdgeManager":
        from executorch.extension.llm.export.builder import LLMEdgeManager

        return LLMEdgeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
