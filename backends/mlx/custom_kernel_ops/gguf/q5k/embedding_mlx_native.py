#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q5_K** embedding lowering via MLX's native 5-bit quantized gather.

Lowers a ``dequantize_gguf -> embedding`` pattern to a quantized gather: gather
the packed quants / scales / biases by index, then dequantize the gathered rows
(``DequantizeNode``, mode "affine"). The GGUF blob is repacked into MLX qparams
at export time (see :mod:`.repack_mlx`).
"""

from __future__ import annotations

from executorch.backends.mlx.builder.op_helpers import emit_quantized_gather
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.custom_kernel_ops.gguf.q5k.repack_mlx import (
    _BITS,
    repack_mlx,
)
from torch.fx.node import Node


def emit_embedding(
    P: MLXProgramBuilder,
    head: Node,
    weight_node: Node,
    indices_node: Node,
    output_dtype,
) -> Slot:
    """Lower a Q5_K ``dequantize_gguf -> embedding`` pattern to a quantized gather.

    Gathers the packed quants / scales / biases by index, then dequantizes the
    gathered rows (MLX affine 5-bit) -- the same shape as MLX's generic quantized
    embedding.
    """
    w_slot, scales_slot, biases_slot, group_size = repack_mlx(P, weight_node)
    (indices_slot,) = P.slot_map([indices_node])

    out = P.make_or_get_slot(head)
    emit_quantized_gather(
        P,
        out,
        indices_slot,
        w_slot,
        scales_slot,
        biases_slot,
        group_size=group_size,
        bits=_BITS,
        mode="affine",
        out_dtype=output_dtype,
    )
    return out
