# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse an FP16 weighted RMSNorm with its residual addition.

Reduction order and rounding follow MLX's rms_norm.metal (Copyright © 2024
Apple Inc., MIT license in backends/mlx/third-party/mlx/LICENSE).
"""

import math
import os

import torch
from executorch.backends.mlx.builder.op_helpers import (
    emit_if_else,
    emit_product,
    emit_shape,
    get_aten_target_normalized,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import PatternHandler, REGISTRY
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    IntOrVid,
    MetalKernelNode,
    ReshapeNode,
)
from torch.fx.node import Node

SOURCE = r"""
const uint row = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint threads = threads_per_threadgroup.x;
const uint axis_size = x_shape[x_ndim - 1];
threadgroup float sums[32];
threadgroup float inv[1];
float acc = 0;
for (uint r = 0; r < axis_size; r += threads * 4) {
    for (uint i = 0; i < 4; ++i) {
        const uint col = r + lid * 4 + i;
        if (col < axis_size) {
            const float value = x[size_t(row) * axis_size + col];
            acc += value * value;
        }
    }
}
acc = simd_sum(acc);
if (simd == 0) sums[lane] = 0;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) sums[simd] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0) {
    acc = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = precise::rsqrt(acc / axis_size + EPSILON);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = 0; r < axis_size; r += threads * 4) {
    for (uint i = 0; i < 4; ++i) {
        const uint col = r + lid * 4 + i;
        if (col < axis_size) {
            const size_t index = size_t(row) * axis_size + col;
            const half normalized = half(float(x[index]) * inv[0]);
            const half weighted = weight[col] * normalized;
            out[index] = RESIDUAL_FIRST ? residual[index] + weighted : weighted + residual[index];
        }
    }
}
"""


@REGISTRY.register_pattern(name="FP16_RMS_NORM_ADD")
class RMSNormAddHandler(PatternHandler):
    def __init__(self, head, norm, x, weight, residual, eps, residual_first):
        super().__init__(head, [norm])
        self.x, self.weight, self.residual = x, weight, residual
        self.eps, self.residual_first = eps, residual_first

    @classmethod
    def maybe_create(cls, ep, head):
        if (
            os.environ.get("ET_MLX_FUSE_NORM_ADD", "0") != "1"
            or get_aten_target_normalized(head.target)
            not in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor)
            or len(head.args) != 2
            or (head.kwargs and head.kwargs != {"alpha": 1})
        ):
            return None
        for residual_first in (True, False):
            residual, norm = head.args if residual_first else head.args[::-1]
            if (
                not isinstance(norm, Node)
                or not isinstance(residual, Node)
                or get_aten_target_normalized(norm.target)
                is not torch.ops.aten.rms_norm.default
                or len(norm.users) != 1
                or len(norm.args) != 4
                or norm.kwargs
            ):
                continue
            x, dims, weight, eps = norm.args
            if not isinstance(x, Node) or not isinstance(weight, Node):
                continue
            shape = x.meta["val"].shape
            if (
                not shape
                or not isinstance(shape[-1], int)
                or shape[-1] <= 0
                or list(dims) != [shape[-1]]
                or weight.meta["val"].shape != (shape[-1],)
                or residual.meta["val"].shape != shape
                or head.meta["val"].shape != shape
                or any(
                    n.meta["val"].dtype != torch.float16
                    for n in (x, weight, residual, norm, head)
                )
                or not isinstance(eps, float)
                or not math.isfinite(eps)
                or eps <= 0
            ):
                continue
            return cls(head, norm, x, weight, residual, eps, residual_first)
        return None

    def __call__(self, P, n):
        inputs = P.slot_map([self.x, self.weight, self.residual])
        out = P.make_or_get_slot(n)
        shape = emit_shape(P, self.x, inputs[0])
        count = emit_product(P, shape[:-1])
        literal = IntOrVid.from_literal
        dim = self.x.meta["val"].shape[-1]
        threads = min(1024, ((dim + 127) // 128) * 32)

        def nonempty():
            grid = emit_product(P, [count, literal(threads)])
            P.emit(
                MetalKernelNode(
                    name="mlx_fp16_rms_norm_add",
                    source=SOURCE.replace("EPSILON", f"float({self.eps!r})"),
                    header="using namespace metal;",
                    inputs=[P.slot_to_tid(s) for s in inputs],
                    outputs=[P.slot_to_tid(out)],
                    input_names=["x", "weight", "residual"],
                    output_names=["out"],
                    grid=[grid, literal(1), literal(1)],
                    threadgroup=[literal(threads), literal(1), literal(1)],
                    ensure_row_contiguous=True,
                    output_shapes_flat=shape,
                    output_shape_lengths=[len(shape)],
                    output_dtypes=[torch_dtype_to_scalar_type(torch.float16)],
                    template_arg_names=["RESIDUAL_FIRST"],
                    template_arg_kinds=[0],
                    template_arg_values=[int(self.residual_first)],
                )
            )

        emit_if_else(
            P,
            count,
            emit_then=nonempty,
            emit_else=lambda: P.emit(
                ReshapeNode(
                    x=P.slot_to_tid(inputs[0]), out=P.slot_to_tid(out), shape=shape
                )
            ),
        )
        return out
