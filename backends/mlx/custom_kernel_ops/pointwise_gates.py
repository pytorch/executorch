# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP16 SwiGLU and sigmoid gating without intermediate tensors."""

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
    IntOrVid,
    MetalKernelNode,
    ReshapeNode,
)
from torch.fx.node import Node

_SOURCE = r"""
uint count = 1;
for (int d = 0; d < gate_ndim; ++d) count *= gate_shape[d];
const uint index = thread_position_in_grid.x;
if (index >= count) return;
INDEXING
const half g = gate[gi];
auto y = 1 / (1 + precise::exp(abs(g)));
const half sigmoid = (g < 0) ? y : 1 - y;
// Match the rounding boundaries of native MLX's separate operations.
const half value = SWIGLU ? half(g * sigmoid) : sigmoid;
out[index] = value * up[ui];
"""


def _source(shape):
    axes = [i for i, size in enumerate(shape) if not isinstance(size, int) or size != 1]
    indexing = ["uint remaining = index;", "size_t gi = 0, ui = 0;"]
    for axis in reversed(axes):
        coordinate = "remaining"
        if axis != axes[0]:
            size = shape[axis]
            divisor = (
                str(size)
                if isinstance(size, int) and size > 0
                else f"gate_shape[{axis}]"
            )
            coordinate = f"c{axis}"
            indexing.extend(
                [
                    f"const uint {coordinate} = remaining % {divisor};",
                    f"remaining /= {divisor};",
                ]
            )
        indexing.extend(
            [
                f"gi += size_t({coordinate}) * gate_strides[{axis}];",
                f"ui += size_t({coordinate}) * up_strides[{axis}];",
            ]
        )
    return _SOURCE.replace("INDEXING", "\n".join(indexing))


@REGISTRY.register_pattern(name="FP16_POINTWISE_GATE")
class PointwiseGateHandler(PatternHandler):
    def __init__(self, head, activation, gate, up, swiglu):
        super().__init__(head, [activation])
        self.gate = gate
        self.up = up
        self.swiglu = swiglu

    @classmethod
    def maybe_create(cls, ep, head):
        if (
            os.environ.get("ET_MLX_FUSE_GATES", "0") != "1"
            or get_aten_target_normalized(head.target)
            not in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul_.Tensor)
            or len(head.args) != 2
        ):
            return None
        if get_aten_target_normalized(head.target) is torch.ops.aten.mul_.Tensor:
            if not isinstance(head.args[0], Node) or len(head.args[0].users) != 1:
                return None
        for activation, up in (head.args, head.args[::-1]):
            if not isinstance(activation, Node) or not isinstance(up, Node):
                continue
            target = get_aten_target_normalized(activation.target)
            if target not in (
                torch.ops.aten.silu.default,
                torch.ops.aten.silu_.default,
                torch.ops.aten.sigmoid.default,
                torch.ops.aten.sigmoid_.default,
            ):
                continue
            if len(activation.users) != 1 or len(activation.args) != 1:
                continue
            gate = activation.args[0]
            if not isinstance(gate, Node):
                continue
            if (
                target
                in (torch.ops.aten.silu_.default, torch.ops.aten.sigmoid_.default)
                and len(gate.users) != 1
            ):
                continue
            gv, uv = gate.meta["val"], up.meta["val"]
            if gv.dtype != torch.float16 or uv.dtype != torch.float16:
                continue
            if gv.shape != uv.shape:
                continue
            return cls(
                head,
                activation,
                gate,
                up,
                target in (torch.ops.aten.silu.default, torch.ops.aten.silu_.default),
            )
        return None

    def __call__(self, P, n):
        gate, up = P.slot_map([self.gate, self.up])
        out = P.make_or_get_slot(n)
        shape = emit_shape(P, self.gate, gate)
        count = emit_product(P, shape)
        literal = IntOrVid.from_literal

        def nonempty():
            grid = emit_product(P, [emit_ceil_div(P, count, 256), literal(256)])
            P.emit(
                MetalKernelNode(
                    name="mlx_fp16_gate",
                    source=_source(self.gate.meta["val"].shape),
                    header="using namespace metal;",
                    inputs=[P.slot_to_tid(gate), P.slot_to_tid(up)],
                    outputs=[P.slot_to_tid(out)],
                    input_names=["gate", "up"],
                    output_names=["out"],
                    grid=[grid, literal(1), literal(1)],
                    threadgroup=[literal(256), literal(1), literal(1)],
                    ensure_row_contiguous=False,
                    output_shapes_flat=shape,
                    output_shape_lengths=[len(shape)],
                    output_dtypes=[torch_dtype_to_scalar_type(torch.float16)],
                    template_arg_names=["SWIGLU"],
                    template_arg_kinds=[0],
                    template_arg_values=[int(self.swiglu)],
                )
            )

        emit_if_else(
            P,
            count,
            emit_then=nonempty,
            emit_else=lambda: P.emit(
                ReshapeNode(x=P.slot_to_tid(gate), out=P.slot_to_tid(out), shape=shape)
            ),
        )
        return out
