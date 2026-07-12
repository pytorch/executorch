#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q4_K** linear lowering via MLX's native 4-bit quantized matmul.

Lowers a ``dequantize_gguf -> linear`` pattern to a ``QuantizedMatmulNode``
(mode "affine"); the GGUF blob is repacked into MLX qparams at export time (see
:mod:`.repack_mlx`). The group size is 32 by default, or the merged 64/128 when
adjacent sub-blocks are identical.
"""

from __future__ import annotations

from typing import Optional

from executorch.backends.mlx.builder.op_helpers import torch_dtype_to_scalar_type
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.repack_mlx import (
    _BITS,
    repack_mlx,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AddNode,
    AsTypeNode,
    QuantizedMatmulNode,
)
from torch.fx.node import Node


def emit_linear(
    P: MLXProgramBuilder,
    head: Node,
    x_node: Node,
    weight_node: Node,
    bias_node: Optional[Node],
) -> Slot:
    """Lower a Q4_K ``dequantize_gguf -> linear`` pattern to MLX 4-bit matmul.

    ``weight_node`` is the raw GGUF blob constant; ``head`` is the ``aten.linear``
    node. The blob is repacked into MLX qparams at export time, so only the
    MLX-format constants are serialized.
    """
    w_slot, scales_slot, biases_slot, group_size = repack_mlx(P, weight_node)
    x_slot, bias_slot = P.slot_map([x_node, bias_node])

    out = P.make_or_get_slot(head)
    P.emit(
        QuantizedMatmulNode(
            x=P.slot_to_tid(x_slot),
            w=P.slot_to_tid(w_slot),
            scales=P.slot_to_tid(scales_slot),
            biases=P.slot_to_tid(biases_slot),
            out=P.slot_to_tid(out),
            group_size=group_size,
            bits=_BITS,
            mode="affine",
            transpose=True,
        )
    )

    if bias_node is not None:
        P.emit(
            AddNode(
                a=P.slot_to_tid(out),
                b=P.slot_to_tid(bias_slot),
                out=P.slot_to_tid(out),
            )
        )

    out_dtype = head.meta["val"].dtype
    if out_dtype != x_node.meta["val"].dtype:
        P.emit(
            AsTypeNode(
                x=P.slot_to_tid(out),
                out=P.slot_to_tid(out),
                scalar_type=torch_dtype_to_scalar_type(out_dtype),
            )
        )

    return out
