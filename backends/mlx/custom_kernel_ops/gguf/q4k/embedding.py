#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""GGUF **Q4_K** embedding lowering via MLX's native 4-bit quantized gather.

Lowers a ``gguf_dequantize -> embedding`` pattern to a quantized gather: gather
the packed quants / scales / biases by index (``TakeNode``), then dequantize the
gathered rows (``DequantizeNode``, mode "affine"). The GGUF blob is repacked into
MLX qparams at export time (see :mod:`.common`).
"""

from __future__ import annotations

from executorch.backends.mlx.builder.op_helpers import torch_dtype_to_scalar_type
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.custom_kernel_ops.gguf.q4k.common import (
    _BITS,
    _repack_mlx,
)
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    DequantizeNode,
    IntOrVidOrTid,
    TakeNode,
)
from torch.fx.node import Node


def emit_embedding(
    P: MLXProgramBuilder,
    head: Node,
    weight_node: Node,
    indices_node: Node,
    output_dtype,
) -> Slot:
    """Lower a Q4_K ``gguf_dequantize -> embedding`` pattern to a quantized gather.

    Gathers the packed quants / scales / biases by index, then dequantizes the
    gathered rows (MLX affine 4-bit) -- the same shape as MLX's generic quantized
    embedding.
    """
    w_slot, scales_slot, biases_slot, group_size = _repack_mlx(P, weight_node)
    (indices_slot,) = P.slot_map([indices_node])
    ids_index = IntOrVidOrTid.from_tid(P.slot_to_tid(indices_slot))

    _, wq_sel = P.make_tmp_slot()
    P.emit(
        TakeNode(
            x=P.slot_to_tid(w_slot),
            index=ids_index,
            out=P.slot_to_tid(wq_sel),
            axis=0,
        )
    )
    _, sc_sel = P.make_tmp_slot()
    P.emit(
        TakeNode(
            x=P.slot_to_tid(scales_slot),
            index=ids_index,
            out=P.slot_to_tid(sc_sel),
            axis=0,
        )
    )
    _, b_sel = P.make_tmp_slot()
    P.emit(
        TakeNode(
            x=P.slot_to_tid(biases_slot),
            index=ids_index,
            out=P.slot_to_tid(b_sel),
            axis=0,
        )
    )

    out = P.make_or_get_slot(head)
    P.emit(
        DequantizeNode(
            w=P.slot_to_tid(wq_sel),
            scales=P.slot_to_tid(sc_sel),
            out=P.slot_to_tid(out),
            biases=P.slot_to_tid(b_sel),
            group_size=group_size,
            bits=_BITS,
            mode="affine",
            dtype=torch_dtype_to_scalar_type(output_dtype),
        )
    )
    return out
