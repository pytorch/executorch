#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
``mlx::tq_norm``: L2 norm along the last dim, lowered to a single Metal kernel.

    norms[..., 0] = sqrt(sum_i x[..., i]^2)

Reads / writes ``x.dtype`` directly (no graph-level dtype casts).
Reduces in fp32 inside Metal registers via ``simd_sum`` for precision
on large ``D`` (bf16 sum-of-squares loses too much for D>=128).

Constraints:
    * Last dim ``D`` must be statically known and a multiple of 32.

Usage::

    import executorch.backends.mlx.model_ops.tq_norm  # noqa: F401

    norms = torch.ops.mlx.tq_norm(x)
    # x:     (..., D) bf16
    # norms: (..., 1) bf16, equal to vector_norm(x, dim=-1, keepdim=True)
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


@torch.library.custom_op("mlx::tq_norm", mutates_args=())
def tq_norm(x: Tensor) -> Tensor:
    """L2 norm along last dim.

    Args:
        x: ``(..., D)``. For MLX lowering, ``D`` must be a multiple of 32.

    Returns:
        ``(..., 1)`` of the same dtype as ``x``.
    """
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True).to(x.dtype)


@torch.library.register_fake("mlx::tq_norm")
def tq_norm_fake(x: Tensor) -> Tensor:
    out_shape = list(x.shape)
    out_shape[-1] = 1
    return x.new_empty(out_shape, dtype=x.dtype)


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


_TQ_NORM_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""


# Per-vector reduction:
#   * Grid (32, 1, M), threadgroup (32, 1, 1): one simdgroup per vector.
#   * Each lane covers DIMS_PER_LANE = D/32 elements; partial sums are
#     accumulated in an fp32 register.
#   * ``simd_sum`` reduces across the 32 lanes; lane 0 sqrts and writes.
_TQ_NORM_SOURCE = """
    constexpr uint DIMS_PER_LANE = D / 32;

    uint vec_id = thread_position_in_grid.z;
    uint lane_id = thread_position_in_threadgroup.x;

    uint base = vec_id * D + lane_id * DIMS_PER_LANE;

    float local_sum_sq = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; ++i) {
        float v = float(x[base + i]);
        local_sum_sq += v * v;
    }

    float total_sum_sq = simd_sum(local_sum_sq);

    if (lane_id == 0) {
        norms[vec_id] = (InT)sqrt(total_sum_sq);
    }
"""


def _compute_M(P: MLXProgramBuilder, node: Node) -> Union[int, IntOrVid]:
    """``M = numel(x) / D`` (product of leading dims). Returns a static
    int when known, else an IntOrVid built from SymSize + MultiplyInt."""
    val = node.meta.get("val")
    if val is None:
        raise ValueError("mlx::tq_norm: input node has no meta['val']")
    shape = val.shape

    last_dim = shape[-1]
    if not isinstance(last_dim, int):
        raise NotImplementedError("mlx::tq_norm: last dim must be statically known")

    leading_shape = list(shape[:-1])

    if all(isinstance(s, int) for s in leading_shape):
        return reduce(mul, [int(s) for s in leading_shape], 1)

    in_slot = P.slot_map([node])[0]
    in_tid = P.slot_to_tid(in_slot)

    acc_iov: Optional[IntOrVid] = None
    for dim_idx, s in enumerate(leading_shape):
        if isinstance(s, int):
            d_iov = IntOrVid.from_literal(int(s))
        else:
            _, d_val = P.make_tmp_value_slot()
            P.emit(
                SymSizeNode(
                    a=in_tid,
                    dim=dim_idx,
                    out=P.slot_to_vid(d_val),
                )
            )
            d_iov = P.to_int_or_vid(d_val)

        if acc_iov is None:
            acc_iov = d_iov
        else:
            _, acc_val = P.make_tmp_value_slot()
            P.emit(
                MultiplyIntNode(
                    a=acc_iov,
                    b=d_iov,
                    out=P.slot_to_vid(acc_val),
                )
            )
            acc_iov = P.to_int_or_vid(acc_val)

    assert acc_iov is not None
    return acc_iov


def _output_shape_flat(P: MLXProgramBuilder, node: Node, in_slot: Slot) -> list:
    """Output shape: same as input but with last dim = 1."""
    val = node.meta["val"]
    shape = val.shape
    last_idx = len(shape) - 1
    out: list = []
    for dim_idx, s in enumerate(shape):
        if isinstance(s, int):
            d = 1 if dim_idx == last_idx else int(s)
            out.append(IntOrVid.from_literal(d))
        else:
            if dim_idx == last_idx:
                raise NotImplementedError(
                    "mlx::tq_norm: dynamic last-dim is not supported"
                )
            _, d_val = P.make_tmp_value_slot()
            P.emit(
                SymSizeNode(
                    a=P.slot_to_tid(in_slot),
                    dim=dim_idx,
                    out=P.slot_to_vid(d_val),
                )
            )
            out.append(P.to_int_or_vid(d_val))
    return out


@REGISTRY.register(target=[torch.ops.mlx.tq_norm.default])
def _tq_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Lower ``mlx::tq_norm`` to a single fused Metal kernel."""
    args = P.args(n)
    if len(args) != 1:
        raise ValueError(f"mlx::tq_norm: expected 1 arg (x), got {len(args)}")

    (x_slot,) = args
    x_node = n.args[0]

    x_meta = x_node.meta["val"]

    last_dim = x_meta.shape[-1]
    if not isinstance(last_dim, int):
        raise NotImplementedError("mlx::tq_norm: last dim must be statically known")
    D = int(last_dim)
    if D % 32 != 0:
        raise NotImplementedError(
            f"mlx::tq_norm: last dim must be a multiple of 32 (one per "
            f"SIMD lane); got D={D}"
        )

    in_dtype_int = torch_dtype_to_scalar_type(x_meta.dtype)

    out = P.make_or_get_slot(n)
    out_shape_flat = _output_shape_flat(P, x_node, x_slot)
    M = _compute_M(P, x_node)
    M_iov: IntOrVid = IntOrVid.from_literal(int(M)) if isinstance(M, int) else M

    P.emit(
        MetalKernelNode(
            name="tq_norm",
            source=_TQ_NORM_SOURCE,
            header=_TQ_NORM_HEADER,
            inputs=[P.slot_to_tid(x_slot)],
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
            input_names=["x"],
            output_names=["norms"],
            output_shapes_flat=out_shape_flat,
            output_shape_lengths=[len(out_shape_flat)],
            output_dtypes=[in_dtype_int],
            template_arg_names=["InT", "D"],
            template_arg_kinds=[2, 0],  # 2=dtype, 0=int
            template_arg_values=[in_dtype_int, D],
        )
    )

    return out


_registered = True
