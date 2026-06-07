#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""MLX pattern handlers for GGUF-quantized weights.

``ExportableGGUFTensor`` (extension/llm/export/gguf.py) lowers a quantized
linear/embedding to::

    linear(x, torchao::gguf_dequantize(weight, ggml_type, out_dtype), bias)
    embedding(torchao::gguf_dequantize(weight, ggml_type, out_dtype), indices)

These handlers match that ``gguf_dequantize -> linear/embedding`` subgraph and
lower it without materializing the dequantized weight:

* **Q6_K** -> fused custom Metal kernels in :mod:`.q6k` (linear + embedding).
* **Q4_K** -> MLX's native 4-bit ``quantized_matmul`` via :mod:`.q4k` (linear);
  the GGUF blocks are repacked into MLX qparams at export time.

Other quant types are left unmatched (the caller is expected to convert them to a
torchao ``Int4Tensor`` / ``IntxUnpackedToInt8Tensor`` first).

Importing this module registers the patterns as a side effect.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from executorch.backends.mlx.builder.op_helpers import get_aten_target
from executorch.backends.mlx.builder.op_registry import PatternHandler, REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.pattern_utils import has_single_user, match_target
from torch.export.exported_program import ExportedProgram
from torch.fx.node import Node

# Quant types each pattern can lower (linear has both a custom Q6_K kernel and an
# MLX-native Q4_K path; embedding only has the Q6_K gather kernel).
_LINEAR_TYPES = {"q4_k", "q6_k"}
_EMBEDDING_TYPES = {"q4_k", "q6_k"}


def parse_gguf_dequantize_node(
    node: Node,
) -> Optional[Tuple[Node, str, torch.dtype]]:
    """Parse a ``torchao::gguf_dequantize`` node.

    Returns ``(weight_node, ggml_type, output_dtype)`` or ``None`` if ``node`` is
    not a ``gguf_dequantize`` node (or the op isn't registered).
    """
    try:
        import executorch.extension.llm.export.gguf  # noqa: F401  registers the op
    except ImportError:
        return None

    if get_aten_target(node.target) is not torch.ops.torchao.gguf_dequantize.default:
        return None

    weight = node.args[0]
    ggml_type = node.args[1]
    output_dtype = torch.bfloat16
    if len(node.args) > 2:
        output_dtype = node.args[2]
    elif "output_dtype" in node.kwargs:
        output_dtype = node.kwargs["output_dtype"]
    return weight, ggml_type, output_dtype


@REGISTRY.register_pattern(name="GGUF_QUANTIZED_LINEAR")
class GGUFQuantizedLinearHandler(PatternHandler):
    """Lower ``gguf_dequantize + linear`` to a fused quantized matmul.

    Matches ``linear(x, gguf_dequantize(weight, ggml_type, out_dtype), bias)``
    and dispatches on ``ggml_type``: Q6_K -> custom Metal kernels, Q4_K -> MLX
    4-bit ``quantized_matmul``.
    """

    def __init__(self, head, body, weight, ggml_type, output_dtype):
        super().__init__(head, body)
        self.weight = weight
        self.ggml_type = ggml_type
        self.output_dtype = output_dtype

    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node):
        if not match_target(head, torch.ops.aten.linear.default):
            return None
        if len(head.args) < 2 or not isinstance(head.args[1], Node):
            return None
        dequant = head.args[1]
        if not has_single_user(dequant):
            return None
        parsed = parse_gguf_dequantize_node(dequant)
        if parsed is None:
            return None
        weight, ggml_type, output_dtype = parsed
        if ggml_type not in _LINEAR_TYPES:
            return None
        return cls(head, [dequant], weight, ggml_type, output_dtype)

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        x_node = n.args[0]
        bias_node = n.args[2] if len(n.args) > 2 else None
        if self.ggml_type == "q6_k":
            from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.linear import (
                emit_linear,
            )
        else:  # q4_k
            from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.linear import (
                emit_linear,
            )
        return emit_linear(P, n, x_node, self.weight, bias_node)


@REGISTRY.register_pattern(name="GGUF_QUANTIZED_EMBEDDING")
class GGUFQuantizedEmbeddingHandler(PatternHandler):
    """Fuse ``gguf_dequantize + embedding`` into the Q6_K gather kernel.

    Matches::

        embedding(gguf_dequantize(weight, "q6_k", out_dtype), indices)
    """

    def __init__(self, head, body, weight, ggml_type, output_dtype):
        super().__init__(head, body)
        self.weight = weight
        self.ggml_type = ggml_type
        self.output_dtype = output_dtype

    @classmethod
    def maybe_create(cls, ep: ExportedProgram, head: Node):
        if not match_target(head, torch.ops.aten.embedding.default):
            return None
        if len(head.args) < 2 or not isinstance(head.args[0], Node):
            return None
        dequant = head.args[0]
        if not has_single_user(dequant):
            return None
        parsed = parse_gguf_dequantize_node(dequant)
        if parsed is None:
            return None
        weight, ggml_type, output_dtype = parsed
        if ggml_type not in _EMBEDDING_TYPES:
            return None
        return cls(head, [dequant], weight, ggml_type, output_dtype)

    def __call__(self, P: MLXProgramBuilder, n: Node) -> Slot:
        assert n == self.head
        indices_node = n.args[1]
        if self.ggml_type == "q6_k":
            from executorch.backends.mlx.custom_kernel_ops.gguf.q6k.embedding import (
                emit_embedding,
            )
        else:  # q4_k
            from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.embedding import (
                emit_embedding,
            )
        return emit_embedding(P, n, self.weight, indices_node, self.output_dtype)
