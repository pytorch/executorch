# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse FP32 rotary arithmetic around FP16 activations."""

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

_CASTS = (
    torch.ops.aten._to_copy.default,
    torch.ops.dim_order_ops._to_dim_order_copy.default,
)

SOURCE = r"""
#pragma clang fp contract(off)
const uint pair = thread_position_in_grid.x;
const uint half_dim = D / 2;
const uint tokens = x_shape[2];
if (pair >= H * tokens * half_dim) return;
const uint d = pair % half_dim;
const uint t = (pair / half_dim) % tokens;
const uint h = pair / (half_dim * tokens);
const uint first = INTERLEAVED ? 2 * d : d;
const uint second = INTERLEAVED ? first + 1 : d + half_dim;
const size_t base = size_t(h) * x_strides[1] + size_t(t) * x_strides[2];
const float a = x[base + first * x_strides[3]];
const float b = x[base + second * x_strides[3]];
const float c = cosine[size_t(t) * cosine_strides[2] + d * cosine_strides[3]];
const float s = sine[size_t(t) * sine_strides[2] + d * sine_strides[3]];
const float ac = a * c;
const float bs = b * s;
const float as = a * s;
const float bc = b * c;
const uint out_base = (h * tokens + t) * D;
out[out_base + first] = half(ac - bs);
out[out_base + second] = half(as + bc);
"""


def _matches(node, *targets):
    return isinstance(node, Node) and get_aten_target_normalized(node.target) in targets


def _sliced_input(first, second, interleaved):
    if not all(_matches(n, torch.ops.aten.slice.Tensor) for n in (first, second)):
        return None
    cast = first.args[0]
    if not _matches(cast, *_CASTS) or second.args[0] is not cast:
        return None
    x = cast.args[0]
    if not isinstance(x, Node) or cast.meta["val"].dtype != torch.float32:
        return None
    shape = x.meta["val"].shape
    if (
        len(shape) != 4
        or shape[0] != 1
        or not isinstance(shape[1], int)
        or not isinstance(shape[3], int)
        or shape[3] <= 0
        or shape[3] % 2
        or x.meta["val"].dtype != torch.float16
    ):
        return None
    half = shape[3] // 2
    slices = (
        ((0, shape[3], 2), (1, shape[3], 2))
        if interleaved
        else (
            (0, half, 1),
            (half, shape[3], 1),
        )
    )
    for node, (start, stop, step) in zip((first, second), slices):
        args = node.args
        if len(args) not in (4, 5) or args[1] not in (3, -1) or args[2] != start:
            return None
        if args[3] not in (stop, 9223372036854775807) or (
            args[3] == 9223372036854775807 and stop != shape[3]
        ):
            return None
        if (args[4] if len(args) == 5 else 1) != step or node.kwargs:
            return None
    return x, cast


def _rotation_inputs(join, interleaved):
    aten = torch.ops.aten
    if len(join.args) != 2 or join.args[1] != -1 or len(join.args[0]) != 2:
        return None
    sub, add = join.args[0]
    for node, targets in (
        (sub, (aten.sub.Tensor, aten.sub_.Tensor)),
        (add, (aten.add.Tensor, aten.add_.Tensor)),
    ):
        if (
            not _matches(node, *targets)
            or len(node.args) != 2
            or (node.kwargs and node.kwargs != {"alpha": 1})
        ):
            return None
    products = [*sub.args, *add.args]
    if any(
        not _matches(p, aten.mul.Tensor, aten.mul_.Tensor)
        or len(p.args) != 2
        or p.kwargs
        for p in products
    ):
        return None
    first, cosine = products[0].args
    second, sine = products[1].args
    if products[2].args != (first, sine) or products[3].args != (second, cosine):
        return None
    sliced = _sliced_input(first, second, interleaved)
    if sliced is None:
        return None
    x, cast = sliced
    shape = x.meta["val"].shape
    for phase in (cosine, sine):
        if (
            not isinstance(phase, Node)
            or phase.meta["val"].dtype != torch.float32
            or phase.meta["val"].shape != (1, 1, shape[2], shape[3] // 2)
        ):
            return None
    return x, cosine, sine, [join, sub, add, *products, first, second, cast]


@REGISTRY.register_pattern(name="FP16_ROTARY")
class RotaryHandler(PatternHandler):
    def __init__(self, head, body, x, cosine, sine, interleaved):
        super().__init__(head, body)
        self.x, self.cosine, self.sine = x, cosine, sine
        self.interleaved = interleaved

    @classmethod
    def maybe_create(cls, ep, head):
        aten = torch.ops.aten
        if os.environ.get("ET_MLX_FUSE_ROPE", "0") != "1" or not _matches(
            head, *_CASTS
        ):
            return None
        if head.meta["val"].dtype != torch.float16:
            return None
        join = head.args[0]
        body = []
        interleaved = _matches(join, aten.view.default, aten.reshape.default)
        if interleaved:
            body.append(join)
            join = join.args[0]
        if not _matches(join, aten.stack.default if interleaved else aten.cat.default):
            return None
        match = _rotation_inputs(join, interleaved)
        if match is None:
            return None
        x, cosine, sine, rotation = match
        body += rotation
        members = {*body, head}
        if head.meta["val"].shape != x.meta["val"].shape or any(
            set(n.users) - members for n in body
        ):
            return None
        return cls(head, body, x, cosine, sine, interleaved)

    def __call__(self, P, n):
        inputs = P.slot_map([self.x, self.cosine, self.sine])
        out = P.make_or_get_slot(n)
        shape = emit_shape(P, self.x, inputs[0])
        literal = IntOrVid.from_literal
        H, D = self.x.meta["val"].shape[1::2]
        count = emit_product(P, [shape[2], literal(H * D // 2)])

        def nonempty():
            grid = emit_product(P, [emit_ceil_div(P, count, 256), literal(256)])
            P.emit(
                MetalKernelNode(
                    name="mlx_fp16_rotary",
                    source=SOURCE,
                    header="using namespace metal;",
                    inputs=[P.slot_to_tid(s) for s in inputs],
                    outputs=[P.slot_to_tid(out)],
                    input_names=["x", "cosine", "sine"],
                    output_names=["out"],
                    grid=[grid, literal(1), literal(1)],
                    threadgroup=[literal(256), literal(1), literal(1)],
                    ensure_row_contiguous=False,
                    output_shapes_flat=shape,
                    output_shape_lengths=[4],
                    output_dtypes=[torch_dtype_to_scalar_type(torch.float16)],
                    template_arg_names=["H", "D", "INTERLEAVED"],
                    template_arg_kinds=[0, 0, 0],
                    template_arg_values=[H, D, int(self.interleaved)],
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
