#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::tq4_compress``: TurboQuant TQ4 quantize + nibble-pack.

Maps ``(..., D)`` floats to ``(..., D/2)`` uint8 by:
    1. Bucketizing each value against ``boundaries`` (15 sorted thresholds).
    2. Packing pairs of 4-bit indices into one byte: high nibble holds
       the even-position index, low nibble holds the odd-position index.

Constraints:
    * ``boundaries`` must be 1-D length 15 (4-bit codebook).
    * Last dim of ``values`` must be even and statically known.

Usage::

    import executorch.backends.mlx.model_ops.tq4_compress  # noqa: F401

    packed = torch.ops.mlx.tq4_compress(rotated, boundaries)
    # rotated:    (..., D)   float
    # boundaries: (15,)      same dtype as rotated
    # packed:     (..., D/2) uint8
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.fx.node import Node


@torch.library.custom_op("mlx::tq4_compress", mutates_args=())
def tq4_compress(values: Tensor, boundaries: Tensor) -> Tensor:
    """TurboQuant TQ4 quantize + nibble-pack.

    Args:
        values: ``(..., D)`` float, last dim must be even.
        boundaries: ``(15,)`` 1-D sorted, same dtype as ``values``.

    Returns:
        ``(..., D/2)`` uint8. Each byte holds two 4-bit indices: high
        nibble is the even-position index, low nibble is the odd.
    """
    if boundaries.dim() != 1 or boundaries.shape[0] != 15:
        raise ValueError(
            f"mlx::tq4_compress: boundaries must be 1-D length 15; "
            f"got shape {tuple(boundaries.shape)}"
        )
    if values.shape[-1] % 2 != 0:
        raise ValueError(
            f"mlx::tq4_compress: input last dim must be even; got "
            f"{values.shape[-1]}"
        )

    indices = torch.bucketize(values, boundaries).to(torch.uint8)
    packed = (indices[..., 0::2] << 4) | indices[..., 1::2]
    return packed


@torch.library.register_fake("mlx::tq4_compress")
def tq4_compress_fake(values: Tensor, boundaries: Tensor) -> Tensor:
    out_shape = list(values.shape)
    out_shape[-1] = out_shape[-1] // 2
    return values.new_empty(out_shape, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# MLX handler
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_helpers import (
    emit_product,
    emit_shape,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
)


# One thread per output byte: reads ``values[2*gid]``, ``values[2*gid+1]``,
# bucketizes against the 15 boundaries (loop unrolled, ``B`` is a template
# constant), and packs the two 4-bit indices into one byte.
_TQ4_COMPRESS_SOURCE = """
    uint gid = thread_position_in_grid.x;
    float v_hi = float(values[2 * gid]);
    float v_lo = float(values[2 * gid + 1]);
    uchar idx_hi = 0;
    uchar idx_lo = 0;
    #pragma unroll
    for (uint i = 0; i < B; ++i) {
        float bnd = float(boundaries[i]);
        idx_hi += (uchar)(v_hi > bnd);
        idx_lo += (uchar)(v_lo > bnd);
    }
    out[gid] = (idx_hi << 4) | idx_lo;
"""


@REGISTRY.register(target=[torch.ops.mlx.tq4_compress.default])
def _tq4_compress_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Lower ``mlx::tq4_compress`` to a fused Metal kernel."""
    args = P.args(n)
    if len(args) != 2:
        raise ValueError(
            f"mlx::tq4_compress: expected 2 args (values, boundaries), "
            f"got {len(args)}"
        )

    values_slot, boundaries_slot = args
    values_node = n.args[0]
    boundaries_node = n.args[1]

    values_meta = values_node.meta["val"]
    boundaries_meta = boundaries_node.meta["val"]

    # Validate boundaries length: must be 15 for 4-bit nibble pack.
    bnd_shape = boundaries_meta.shape
    if (
        len(bnd_shape) != 1
        or not isinstance(bnd_shape[0], int)
        or int(bnd_shape[0]) != 15
    ):
        raise ValueError(
            f"mlx::tq4_compress: boundaries must be 1-D length 15; "
            f"got shape {tuple(bnd_shape)}"
        )

    last_dim = values_meta.shape[-1]
    if not isinstance(last_dim, int):
        raise NotImplementedError(
            "mlx::tq4_compress: last dim must be statically known"
        )
    if int(last_dim) % 2 != 0:
        raise ValueError(f"mlx::tq4_compress: last dim must be even; got {last_dim}")
    half_last = int(last_dim) // 2

    in_dtype_int = torch_dtype_to_scalar_type(values_meta.dtype)

    out = P.make_or_get_slot(n)
    leading = emit_shape(P, values_node, values_slot, end_dim=-1)
    half_last_iov = IntOrVid.from_literal(half_last)
    out_shape_flat = leading + [half_last_iov]

    # One thread per output byte, so the grid size is the output numel
    # (product of leading dims times the halved last dim).
    n_out_iov = emit_product(P, leading + [half_last_iov])

    P.emit(
        MetalKernelNode(
            name="tq4_compress",
            source=_TQ4_COMPRESS_SOURCE,
            inputs=[
                P.slot_to_tid(values_slot),
                P.slot_to_tid(boundaries_slot),
            ],
            outputs=[P.slot_to_tid(out)],
            grid=[n_out_iov, IntOrVid.from_literal(1), IntOrVid.from_literal(1)],
            # 32 threads per threadgroup so each TG fills one Apple-GPU SIMD group
            threadgroup=[
                IntOrVid.from_literal(32),
                IntOrVid.from_literal(1),
                IntOrVid.from_literal(1),
            ],
            input_names=["values", "boundaries"],
            output_names=["out"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[torch_dtype_to_scalar_type(torch.uint8)],
            template_arg_names=["InT", "B"],
            template_arg_kinds=[2, 0],  # 2=dtype, 0=int
            template_arg_values=[
                in_dtype_int,
                15,
            ],
        )
    )

    return out
