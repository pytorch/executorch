# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import Any

import torch
from executorch.backends.cadence.aot.pass_utils import get_arg, replace_with_op
from executorch.backends.cadence.aot.quantizer.utils import (
    copy_node_metadata,
    create_zero_bias_int32,
    quantize_tensor_multiplier,
)
from executorch.backends.cadence.aot.utils import is_depthwise_conv
from torch import fx
from torch._ops import OpOverload

DQ_PER_TENSOR: OpOverload = torch.ops.quantized_decomposed.dequantize_per_tensor.default
Q_PER_TENSOR: OpOverload = torch.ops.quantized_decomposed.quantize_per_tensor.default


def insert_node_with_meta(
    gm: fx.GraphModule,
    op: OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    insert_before: fx.Node,
    like_node: fx.Node,
) -> fx.Node:
    """Create a new node and populate its FakeTensor metadata.

    Inserts ``op(*args, **kwargs)`` before ``insert_before``, runs the op
    under ``like_node``'s fake_mode to compute ``meta["val"]``, and copies
    remaining metadata from ``like_node``.
    """
    with gm.graph.inserting_before(insert_before):
        node = gm.graph.call_function(op, args, kwargs or {})
    assert "val" in like_node.meta
    fake_mode = like_node.meta["val"].fake_mode
    assert fake_mode is not None

    def _resolve(x: Any) -> Any:
        return x.meta["val"] if isinstance(x, fx.Node) else x

    fake_args = tuple(_resolve(a) for a in args)
    fake_kwargs = {k: _resolve(v) for k, v in (kwargs or {}).items()}
    with fake_mode:
        node.meta["val"] = op(*fake_args, **fake_kwargs)
    copy_node_metadata(node, like_node)
    return node


def find_quant_user(node: fx.Node) -> fx.Node | None:
    """Find the first quantize_per_tensor user of ``node``, traversing through getitem."""
    users = list(node.users)
    if not users:
        return None
    user = users[0]
    if user.target is operator.getitem:
        if user.args[1] == 0:
            users = list(user.users)
            if not users:
                return None
            user = users[0]
        else:
            return None
    if user.target == Q_PER_TENSOR:
        return user
    return None


def fuse_conv(
    pattern: object,
    gm: fx.GraphModule,
    conv_node: fx.Node,
    dq_input: fx.Node,
    dq_weight: fx.Node,
    quant_node: fx.Node,
) -> fx.Node:
    """Fuse a dq->conv->q chain into a single quantized conv op."""
    dq_bias = None
    if len(conv_node.args) > 2 and conv_node.args[2] is not None:
        bias_arg = conv_node.args[2]
        assert isinstance(bias_arg, fx.Node)
        dq_bias = bias_arg if bias_arg.target == DQ_PER_TENSOR else None
    weight_scale = get_arg(dq_weight, "scale", float)
    input_scale = get_arg(dq_input, "scale", float)
    bias_scale = input_scale * weight_scale
    if dq_bias is not None:
        bias_q = get_arg(dq_bias, "input", fx.Node)
    else:
        # Cadence quantized conv ops require a non-optional bias argument.
        weight_node = get_arg(dq_weight, "input", fx.Node)
        with gm.graph.inserting_before(conv_node):
            bias_q = create_zero_bias_int32(gm, weight_node, bias_scale)
    requantize_scale = bias_scale / get_arg(quant_node, "scale", float)
    requantize_scale_t = torch.tensor([requantize_scale])
    out_multiplier, out_shift = quantize_tensor_multiplier(requantize_scale_t)
    args = (
        get_arg(dq_input, "input", fx.Node),
        get_arg(dq_weight, "input", fx.Node),
        bias_q,
    )
    groups = get_arg(conv_node, "groups", int)
    kwargs = {
        "stride": get_arg(conv_node, "stride", list[int]),
        "padding": get_arg(conv_node, "padding", list[int]),
        "dilation": get_arg(conv_node, "dilation", list[int]),
        "groups": groups,
        "input_zero_point": get_arg(dq_input, "zero_point", int),
        "weight_zero_point": get_arg(dq_weight, "zero_point", int),
        "bias_scale": bias_scale,
        "out_scale": get_arg(quant_node, "scale", float),
        "out_zero_point": get_arg(quant_node, "zero_point", int),
        "out_multiplier": out_multiplier[0].item(),
        "out_shift": out_shift[0].item(),
    }
    replacement_op = pattern.replacement_op()  # pyre-ignore[16]
    if replacement_op == torch.ops.cadence.quantized_conv1d_ncl.per_tensor:
        input_node = get_arg(dq_input, "input", fx.Node)
        assert len(input_node.meta["val"].shape) >= 2
        in_channels = input_node.meta["val"].shape[1]
        if is_depthwise_conv(groups, in_channels):
            replacement_op = torch.ops.cadence.quantized_depthwise_conv1d_ncl.per_tensor
    return replace_with_op(gm, conv_node, replacement_op, args, kwargs, quant_node)


def fuse_linear(
    gm: fx.GraphModule,
    dq_input: fx.Node,
    dq_weight: fx.Node,
    dq_bias: fx.Node | None,
    quant_node: fx.Node,
    op_node: fx.Node,
    replacement_op: OpOverload,
    weight_q: fx.Node | None = None,
) -> fx.Node:
    """Fuse a dq->linear->q chain into a single quantized linear op."""
    assert op_node.target in (
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
    ), f"Expected linear/addmm, got {op_node.target}"
    weight_scale = get_arg(dq_weight, "scale", float)
    input_scale = get_arg(dq_input, "scale", float)
    bias_scale = input_scale * weight_scale
    requantize_scale = bias_scale / get_arg(quant_node, "scale", float)
    requantize_scale_t = torch.tensor([requantize_scale])
    out_multiplier, out_shift = quantize_tensor_multiplier(requantize_scale_t)
    if dq_bias is not None:
        bias_q = get_arg(dq_bias, "input", fx.Node)
    else:
        # Cadence quantized linear ops require a non-optional bias argument.
        weight_node = get_arg(dq_weight, "input", fx.Node)
        with gm.graph.inserting_before(op_node):
            bias_q = create_zero_bias_int32(gm, weight_node, bias_scale)
    final_weight = (
        weight_q if weight_q is not None else get_arg(dq_weight, "input", fx.Node)
    )
    args = (get_arg(dq_input, "input", fx.Node), final_weight, bias_q)
    kwargs = {
        "src_zero_point": get_arg(dq_input, "zero_point", int),
        "weight_zero_point": get_arg(dq_weight, "zero_point", int),
        "out_multiplier": out_multiplier[0].item(),
        "out_shift": out_shift[0].item(),
        "out_zero_point": get_arg(quant_node, "zero_point", int),
        "offset": None,
    }
    return replace_with_op(gm, op_node, replacement_op, args, kwargs, quant_node)


def fuse_matmul(
    gm: fx.GraphModule,
    anchor_node: fx.Node,
    dq0: fx.Node,
    dq1: fx.Node,
    quant_node: fx.Node,
    replacement_op: OpOverload,
) -> fx.Node:
    """Fuse a dq->matmul->q chain into a single quantized matmul op."""
    assert anchor_node.target in (
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
    ), f"Expected bmm/matmul, got {anchor_node.target}"
    scale0 = get_arg(dq0, "scale", float)
    scale1 = get_arg(dq1, "scale", float)
    requantize_scale = (scale0 * scale1) / get_arg(quant_node, "scale", float)
    requantize_scale_t = torch.tensor([requantize_scale])
    out_multiplier, out_shift = quantize_tensor_multiplier(requantize_scale_t)
    args = (
        get_arg(dq0, "input", fx.Node),
        get_arg(dq0, "zero_point", int),
        get_arg(dq1, "input", fx.Node),
        get_arg(dq1, "zero_point", int),
        None,
    )
    kwargs = {
        "out_multiplier": out_multiplier[0].item(),
        "out_shift": out_shift[0].item(),
        "out_zero_point": get_arg(quant_node, "zero_point", int),
        "transposed": False,
    }
    return replace_with_op(gm, anchor_node, replacement_op, args, kwargs, quant_node)
