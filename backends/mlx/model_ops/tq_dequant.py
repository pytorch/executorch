#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::tq_dequant``: TurboQuant TQ4 unpack + centroid gather + multiply-by-norm.

    indices    = unpack 4-bit nibbles from packed bytes  (..., D)
    centvals   = centroids[indices]                       (..., D)
    out        = centvals * norms                         (..., D)

Output is in **rotated space** — the inverse rotation, if needed, is
left to the caller (typically MLX's tuned bf16 GEMM).

Constraints:
    * ``D`` (= ``packed.shape[-1] * 2``) must be a multiple of 32.
    * ``centroids`` must be a 1-D tensor of length 16.
    * Output dtype matches ``norms.dtype``.

Usage::

    import executorch.backends.mlx.model_ops.tq_dequant  # noqa: F401

    out = torch.ops.mlx.tq_dequant(packed, norms, centroids)
    # packed:    (..., D/2) uint8
    # norms:     (..., 1)   bf16
    # centroids: (16,)      bf16
    # out:       (..., D)   bf16  (in rotated space)
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Optional, Union

import torch
from torch import Tensor
from torch.fx.node import Node


# ---------------------------------------------------------------------------
# Custom op + eager fallback
# ---------------------------------------------------------------------------


@torch.library.custom_op("mlx::tq_dequant", mutates_args=())
def tq_dequant(
    packed: Tensor,
    norms: Tensor,
    centroids: Tensor,
) -> Tensor:
    """Fused unpack + centroid gather + multiply-by-norm.

    Args:
        packed: ``(..., D/2)`` uint8. High nibble = even-position index,
            low nibble = odd-position index.
        norms: ``(..., 1)`` of compute dtype, broadcasts over D.
        centroids: ``(16,)`` of compute dtype.

    Returns:
        ``(..., D)`` of compute dtype, in rotated space.
    """
    if centroids.dim() != 1 or centroids.shape[0] != 16:
        raise ValueError(
            f"mlx::tq_dequant: centroids must be 1-D length 16; got "
            f"shape {tuple(centroids.shape)}"
        )
    high = (packed >> 4).long()
    low = (packed & 0x0F).long()
    indices = torch.stack([high, low], dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    return centroids[indices] * norms


@torch.library.register_fake("mlx::tq_dequant")
def tq_dequant_fake(packed: Tensor, norms: Tensor, centroids: Tensor) -> Tensor:
    out_shape = list(packed.shape)
    out_shape[-1] = out_shape[-1] * 2
    return packed.new_empty(out_shape, dtype=norms.dtype)


# ---------------------------------------------------------------------------
# MLX handler
# ---------------------------------------------------------------------------

from executorch.backends.mlx.builder.op_helpers import torch_dtype_to_scalar_type
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
    MultiplyIntNode,
    SymSizeNode,
)


_TQ_DEQUANT_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""


# Per-vector decompress:
#   * Grid (32, 1, M), threadgroup (32, 1, 1): one simdgroup per vector.
#   * Each lane handles DIMS_PER_LANE = D/32 output values, sourced
#     from BYTES_PER_LANE = DIMS_PER_LANE/2 packed bytes.
#   * The 16-entry codebook is preloaded into per-lane registers.
_TQ_DEQUANT_SOURCE = """
    constexpr uint DIMS_PER_LANE  = D / 32;
    constexpr uint BYTES_PER_LANE = DIMS_PER_LANE / 2;

    uint vec_id  = thread_position_in_grid.z;
    uint lane_id = thread_position_in_threadgroup.x;

    InT cent[16];
    for (uint c = 0; c < 16; ++c) {
        cent[c] = centroids[c];
    }

    InT norm = norms[vec_id];

    uint packed_base = vec_id * (D / 2) + lane_id * BYTES_PER_LANE;
    uint out_base    = vec_id * D       + lane_id * DIMS_PER_LANE;

    for (uint i = 0; i < BYTES_PER_LANE; ++i) {
        uchar byte = packed[packed_base + i];
        uchar idx_hi = (byte >> 4) & 0x0F;
        uchar idx_lo = byte & 0x0F;
        out[out_base + 2 * i + 0] = cent[idx_hi] * norm;
        out[out_base + 2 * i + 1] = cent[idx_lo] * norm;
    }
"""


def _compute_M(P: MLXProgramBuilder, packed_node: Node) -> Union[int, IntOrVid]:
    """``M`` = numel(packed) / (D/2) = product of leading dims of ``packed``."""
    val = packed_node.meta.get("val")
    if val is None:
        raise ValueError("mlx::tq_dequant: input has no meta['val']")
    shape = val.shape

    if not isinstance(shape[-1], int):
        raise NotImplementedError(
            "mlx::tq_dequant: last dim of packed must be statically known"
        )

    leading = list(shape[:-1])
    if all(isinstance(s, int) for s in leading):
        return reduce(mul, [int(s) for s in leading], 1)

    in_slot = P.slot_map([packed_node])[0]
    in_tid = P.slot_to_tid(in_slot)

    acc_iov: Optional[IntOrVid] = None
    for dim_idx, s in enumerate(leading):
        if isinstance(s, int):
            d_iov = IntOrVid.from_literal(int(s))
        else:
            _, d_val = P.make_tmp_value_slot()
            P.emit(SymSizeNode(a=in_tid, dim=dim_idx, out=P.slot_to_vid(d_val)))
            d_iov = P.to_int_or_vid(d_val)

        if acc_iov is None:
            acc_iov = d_iov
        else:
            _, acc_val = P.make_tmp_value_slot()
            P.emit(MultiplyIntNode(a=acc_iov, b=d_iov, out=P.slot_to_vid(acc_val)))
            acc_iov = P.to_int_or_vid(acc_val)

    assert acc_iov is not None
    return acc_iov


def _output_shape_flat(
    P: MLXProgramBuilder, packed_node: Node, packed_slot: Slot
) -> list:
    """Output shape: same as packed but with last dim doubled."""
    val = packed_node.meta["val"]
    shape = val.shape
    last_idx = len(shape) - 1
    out: list = []
    for dim_idx, s in enumerate(shape):
        if isinstance(s, int):
            d = int(s) * 2 if dim_idx == last_idx else int(s)
            out.append(IntOrVid.from_literal(d))
        else:
            if dim_idx == last_idx:
                raise NotImplementedError(
                    "mlx::tq_dequant: dynamic last-dim is not supported"
                )
            _, d_val = P.make_tmp_value_slot()
            P.emit(
                SymSizeNode(
                    a=P.slot_to_tid(packed_slot),
                    dim=dim_idx,
                    out=P.slot_to_vid(d_val),
                )
            )
            out.append(P.to_int_or_vid(d_val))
    return out


@REGISTRY.register(target=[torch.ops.mlx.tq_dequant.default])
def _tq_dequant_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Lower ``mlx::tq_dequant`` to a single fused Metal kernel."""
    args = P.args(n)
    if len(args) != 3:
        raise ValueError(
            f"mlx::tq_dequant: expected 3 args (packed, norms, centroids); "
            f"got {len(args)}"
        )
    packed_slot, norms_slot, centroids_slot = args
    packed_node = n.args[0]
    norms_node = n.args[1]
    centroids_node = n.args[2]

    packed_meta = packed_node.meta["val"]
    norms_meta = norms_node.meta["val"]
    centroids_meta = centroids_node.meta["val"]

    if centroids_meta.dim() != 1 or int(centroids_meta.shape[0]) != 16:
        raise ValueError(
            f"mlx::tq_dequant: centroids must be 1-D length 16; got "
            f"shape {tuple(centroids_meta.shape)}"
        )

    last_dim_packed = packed_meta.shape[-1]
    if not isinstance(last_dim_packed, int):
        raise NotImplementedError(
            "mlx::tq_dequant: packed last dim must be statically known"
        )
    half_D = int(last_dim_packed)
    D = half_D * 2
    if D % 32 != 0:
        raise NotImplementedError(
            f"mlx::tq_dequant: unpacked dim must be a multiple of 32 (one "
            f"per SIMD lane); got D={D}"
        )

    out_dtype_int = torch_dtype_to_scalar_type(norms_meta.dtype)

    out = P.make_or_get_slot(n)
    out_shape_flat = _output_shape_flat(P, packed_node, packed_slot)
    M = _compute_M(P, packed_node)
    M_iov: IntOrVid = IntOrVid.from_literal(int(M)) if isinstance(M, int) else M

    P.emit(
        MetalKernelNode(
            name="tq_dequant",
            source=_TQ_DEQUANT_SOURCE,
            header=_TQ_DEQUANT_HEADER,
            inputs=[
                P.slot_to_tid(packed_slot),
                P.slot_to_tid(norms_slot),
                P.slot_to_tid(centroids_slot),
            ],
            outputs=[P.slot_to_tid(out)],
            grid=[
                IntOrVid.from_literal(32),
                IntOrVid.from_literal(1),
                M_iov,
            ],
            threadgroup=[
                IntOrVid.from_literal(32),
                IntOrVid.from_literal(1),
                IntOrVid.from_literal(1),
            ],
            input_names=["packed", "norms", "centroids"],
            output_names=["out"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[out_dtype_int],
            template_arg_names=["InT", "D"],
            template_arg_kinds=[2, 0],  # 2=dtype, 0=int
            template_arg_values=[out_dtype_int, D],
        )
    )

    return out


_registered = True
