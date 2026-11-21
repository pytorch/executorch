# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse the FP32 vocabulary soft cap without changing rounding boundaries."""

import os

import torch
from executorch.backends.mlx.builder.op_helpers import (
    emit_ceil_div,
    emit_if_else,
    emit_product,
    emit_shape,
    get_aten_target_normalized,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import PatternHandler, REGISTRY
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AsTypeNode,
    IntOrVid,
    MetalKernelNode,
)
from torch.fx.node import Node

SOURCE = r"""
#pragma clang fp contract(off)
uint count = 1;
for (int d = 0; d < x_ndim; ++d) count *= x_shape[d];
const uint index = thread_position_in_grid.x;
if (index >= count) return;
const float scaled = float(x[index]) * multiplier[0];
const float divided = scaled / cap[0];
const float capped = precise::tanh(divided);
out[index] = capped * cap[0];
"""


@REGISTRY.register_pattern(name="FP32_SOFT_CAP")
class SoftCapHandler(PatternHandler):
    def __init__(self, head, body, x, multiplier, cap):
        super().__init__(head, body)
        self.x, self.multiplier, self.cap = x, multiplier, cap

    @classmethod
    def maybe_create(cls, ep, head):
        if os.environ.get("ET_MLX_FUSE_SOFT_CAP", "0") != "1":
            return None
        aten = torch.ops.aten
        node = head
        chain = []
        for targets, nargs in (
            ((aten.mul.Tensor, aten.mul_.Tensor), 2),
            ((aten.tanh.default, aten.tanh_.default), 1),
            ((aten.div.Tensor, aten.div_.Tensor), 2),
            ((aten.mul.Tensor, aten.mul_.Tensor), 2),
            (
                (
                    aten._to_copy.default,
                    torch.ops.dim_order_ops._to_dim_order_copy.default,
                ),
                1,
            ),
        ):
            if (
                not isinstance(node, Node)
                or get_aten_target_normalized(node.target) not in targets
                or len(node.args) != nargs
                or node.meta["val"].dtype != torch.float32
                or (chain and len(node.users) != 1)
                or (len(chain) < 4 and node.kwargs)
            ):
                return None
            chain.append(node)
            node = node.args[0]
        cap, multiplier = head.args[1], chain[3].args[1]
        if chain[2].args[1] is not cap or not isinstance(node, Node):
            return None
        if (
            node.meta["val"].dtype != torch.float16
            or not node.meta["val"].ndim
            or node.meta["val"].shape != head.meta["val"].shape
        ):
            return None
        if any(
            not isinstance(n, Node)
            or n.meta["val"].dtype != torch.float32
            or n.meta["val"].numel() != 1
            for n in (multiplier, cap)
        ):
            return None
        return cls(head, chain[1:], node, multiplier, cap)

    def __call__(self, P, n):
        inputs = P.slot_map([self.x, self.multiplier, self.cap])
        out = P.make_or_get_slot(n)
        shape = emit_shape(P, self.x, inputs[0])
        count = emit_product(P, shape)
        literal = IntOrVid.from_literal
        dtype = torch_dtype_to_scalar_type(torch.float32)
        source = SOURCE
        for node, name in ((self.multiplier, "multiplier"), (self.cap, "cap")):
            if node.meta["val"].ndim == 0:
                source = source.replace(f"{name}[0]", name)

        def nonempty():
            grid = emit_product(P, [emit_ceil_div(P, count, 256), literal(256)])
            P.emit(
                MetalKernelNode(
                    name="mlx_fp32_soft_cap",
                    source=source,
                    header="using namespace metal;",
                    inputs=[P.slot_to_tid(s) for s in inputs],
                    outputs=[P.slot_to_tid(out)],
                    input_names=["x", "multiplier", "cap"],
                    output_names=["out"],
                    grid=[grid, literal(1), literal(1)],
                    threadgroup=[literal(256), literal(1), literal(1)],
                    ensure_row_contiguous=True,
                    output_shapes_flat=shape,
                    output_shape_lengths=[len(shape)],
                    output_dtypes=[dtype],
                )
            )

        emit_if_else(
            P,
            count,
            emit_then=nonempty,
            emit_else=lambda: P.emit(
                AsTypeNode(
                    x=P.slot_to_tid(inputs[0]),
                    out=P.slot_to_tid(out),
                    scalar_type=dtype,
                )
            ),
        )
        return out
