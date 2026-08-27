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
DQ_PER_CHANNEL: OpOverload = (
    torch.ops.quantized_decomposed.dequantize_per_channel.default
)
Q_PER_TENSOR: OpOverload = torch.ops.quantized_decomposed.quantize_per_tensor.default

# Fusion needs the ExportedProgram to read per-channel scale tensors and to
# materialize the derived qparam constants, but `PassBase.call` only receives a
# GraphModule and `QuantizationPattern.fuse` is a public API implemented by ~20
# patterns. Rather than thread an extra argument through all of them,
# QuantFusionPass stashes the program here for the duration of the pass.
EXPORTED_PROGRAM_META_KEY: str = "_cadence_quant_fusion_exported_program"


def is_weight_dq(node: object) -> bool:
    """True if ``node`` is a dequantize that a quantized op can absorb.

    Only use this to guard a pattern whose fusion helper handles per-channel
    qparams, which today means the conv patterns (``fuse_conv``). ``fuse_linear``,
    ``fuse_matmul`` and the mixed w8a32 paths read the scalar ``scale`` arg, which
    a per-channel dequantize does not have, so they must keep checking for
    ``DQ_PER_TENSOR`` and let per-channel weights fall back to float.
    """
    return isinstance(node, fx.Node) and node.target in (
        DQ_PER_TENSOR,
        DQ_PER_CHANNEL,
    )


def is_per_channel_dq(node: object) -> bool:
    return isinstance(node, fx.Node) and node.target is DQ_PER_CHANNEL


def get_exported_program(gm: fx.GraphModule) -> Any:
    """The ExportedProgram being fused, or None outside QuantFusionPass."""
    return gm.meta.get(EXPORTED_PROGRAM_META_KEY)


def resolve_constant(gm: fx.GraphModule, node: object) -> torch.Tensor | None:
    """Read the tensor behind a placeholder or get_attr node."""
    if not isinstance(node, fx.Node):
        return None
    if node.op == "get_attr":
        return getattr(gm, str(node.target), None)
    ep = get_exported_program(gm)
    if ep is None:
        return None
    # local import: torch._export.utils pulls in export internals
    from torch._export.utils import (
        get_buffer,
        get_lifted_tensor_constant,
        get_param,
        is_buffer,
        is_lifted_tensor_constant,
        is_param,
    )

    if is_param(ep, node):
        return get_param(ep, node)
    if is_buffer(ep, node):
        return get_buffer(ep, node)
    if is_lifted_tensor_constant(ep, node):
        return get_lifted_tensor_constant(ep, node)
    return None


def add_constant_placeholder(
    gm: fx.GraphModule,
    tensor: torch.Tensor,
    like_node: fx.Node,
    name_hint: str,
) -> fx.Node:
    """Materialize ``tensor`` as a lifted constant placeholder.

    Per-channel fusion produces qparam vectors (bias_scale, out_multiplier,
    out_shift, weight_zero_point) that have to reach the kernel as tensors. The
    graph is already exported by this point, so they cannot be `get_attr` nodes:
    they have to be registered in ``ExportedProgram.constants`` and appear as
    placeholders with a matching entry in the graph signature. This mirrors
    ``exir.passes.constant_prop_pass.replace_with_constant_node``.
    """
    from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

    ep = get_exported_program(gm)
    assert ep is not None, "per-channel fusion requires the ExportedProgram"

    prefix = f"_cadence_{name_hint}_"
    idx = 0
    while f"{prefix}{idx}" in ep.constants:
        idx += 1
    fqn = f"{prefix}{idx}"
    ep.constants[fqn] = tensor

    # Placeholder order and graph_signature.input_specs order must agree, and
    # constants have to precede user inputs. Insert at the user-input boundary
    # in both, the same way exir's constant_prop_pass does.
    user_inputs = set(ep.graph_signature.user_inputs)
    first_user_input = next(
        (
            n
            for n in ep.graph.nodes
            if n.op == "placeholder" and n.name in user_inputs
        ),
        None,
    )
    specs = ep.graph_signature.input_specs
    if first_user_input is not None:
        with ep.graph.inserting_before(first_user_input):
            node = ep.graph.placeholder(fqn)
        spec_idx = next(
            (
                i
                for i, s in enumerate(specs)
                if getattr(s.arg, "name", None) == first_user_input.name
            ),
            len(specs),
        )
    else:
        placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
        anchor = placeholders[-1] if placeholders else next(iter(ep.graph.nodes))
        with ep.graph.inserting_after(anchor):
            node = ep.graph.placeholder(fqn)
        spec_idx = len(specs)

    fake_mode = like_node.meta["val"].fake_mode if "val" in like_node.meta else None
    if fake_mode is not None:
        node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
        node.meta["val"].constant = tensor
    else:
        node.meta["val"] = tensor

    specs.insert(
        spec_idx,
        InputSpec(
            kind=InputKind.CONSTANT_TENSOR,
            arg=TensorArgument(name=node.name),
            target=fqn,
            persistent=True,
        ),
    )
    return node


def tensor_qparam_overload(op: OpOverload) -> OpOverload:
    """Map a `.per_tensor` Cadence op to its tensor-qparam `.default` sibling.

    Both overloads take the same operands; the scalar one inlines the qparams as
    SymInt/float args, while the tensor one takes them as constant tensors. Only
    the tensor form can express per-channel.
    """
    name = op._schema.name.split("::")[-1]
    packet = getattr(torch.ops.cadence, name, None)
    assert packet is not None and hasattr(packet, "default"), (
        f"no tensor-qparam overload registered for {name}; per-channel needs one"
    )
    return packet.default


def get_weight_scale(
    gm: fx.GraphModule, dq_weight: fx.Node
) -> float | torch.Tensor:
    """Weight scale as a float (per-tensor) or a per-output-channel vector."""
    if not is_per_channel_dq(dq_weight):
        return get_arg(dq_weight, "scale", float)
    scales = resolve_constant(gm, get_arg(dq_weight, "scales", fx.Node))
    assert scales is not None, (
        f"could not resolve per-channel weight scales for {dq_weight}"
    )
    axis = get_arg(dq_weight, "axis", int)
    assert axis == 0, (
        f"Cadence per-channel weights must be quantized on the output-channel "
        f"axis (0), got axis={axis}"
    )
    return scales.to(torch.float32).flatten()


def get_weight_zero_point(
    gm: fx.GraphModule, dq_weight: fx.Node
) -> int | torch.Tensor:
    """Weight zero point as an int (per-tensor) or a per-channel int32 vector."""
    if not is_per_channel_dq(dq_weight):
        return get_arg(dq_weight, "zero_point", int)
    # Not typed as fx.Node: symmetric per-channel leaves this argument unset, and
    # get_arg raises on a type mismatch before we could check for None.
    zps = resolve_constant(gm, get_arg(dq_weight, "zero_points"))
    if zps is None:
        # symmetric per-channel: zero_points is allowed to be absent
        scales = get_weight_scale(gm, dq_weight)
        assert isinstance(scales, torch.Tensor)
        return torch.zeros_like(scales, dtype=torch.int32)
    return zps.to(torch.int32).flatten()


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
        dq_bias = bias_arg if is_weight_dq(bias_arg) else None
    per_channel = is_per_channel_dq(dq_weight)
    weight_scale = get_weight_scale(gm, dq_weight)
    input_scale = get_arg(dq_input, "scale", float)
    bias_scale = input_scale * weight_scale
    if dq_bias is not None:
        bias_q = get_arg(dq_bias, "input", fx.Node)
    else:
        # Cadence quantized conv ops require a non-optional bias argument.
        weight_node = get_arg(dq_weight, "input", fx.Node)
        with gm.graph.inserting_before(conv_node):
            # the helper only needs a representative scalar, it fills with zeros
            bias_q = create_zero_bias_int32(
                gm,
                weight_node,
                float(bias_scale.max()) if per_channel else bias_scale,
            )
    out_scale = get_arg(quant_node, "scale", float)
    requantize_scale = bias_scale / out_scale
    requantize_scale_t = (
        requantize_scale
        if isinstance(requantize_scale, torch.Tensor)
        else torch.tensor([requantize_scale])
    )
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
        "out_scale": out_scale,
        "out_zero_point": get_arg(quant_node, "zero_point", int),
    }
    replacement_op = pattern.replacement_op()  # pyre-ignore[16]
    if replacement_op == torch.ops.cadence.quantized_conv1d_ncl.per_tensor:
        input_node = get_arg(dq_input, "input", fx.Node)
        assert len(input_node.meta["val"].shape) >= 2
        in_channels = input_node.meta["val"].shape[1]
        if is_depthwise_conv(groups, in_channels):
            replacement_op = torch.ops.cadence.quantized_depthwise_conv1d_ncl.per_tensor
    if per_channel:
        # The tensor-qparam overload takes the same operands but carries the
        # qparams as constant tensors, which is what per-channel needs. This runs
        # after depthwise selection so that it swaps whichever base op was chosen.
        replacement_op = tensor_qparam_overload(replacement_op)
        kwargs["weight_zero_point"] = add_constant_placeholder(
            gm, get_weight_zero_point(gm, dq_weight), conv_node, "wzp"
        )
        kwargs["bias_scale"] = add_constant_placeholder(
            gm, bias_scale.to(torch.float32), conv_node, "bias_scale"
        )
        kwargs["out_multiplier"] = add_constant_placeholder(
            gm, out_multiplier.to(torch.int32), conv_node, "out_multiplier"
        )
        kwargs["out_shift"] = add_constant_placeholder(
            gm, out_shift.to(torch.int32), conv_node, "out_shift"
        )
    else:
        kwargs["weight_zero_point"] = get_arg(dq_weight, "zero_point", int)
        kwargs["bias_scale"] = bias_scale
        kwargs["out_multiplier"] = out_multiplier[0].item()
        kwargs["out_shift"] = out_shift[0].item()
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
    per_channel = is_per_channel_dq(dq_weight)
    weight_scale = get_weight_scale(gm, dq_weight)
    input_scale = get_arg(dq_input, "scale", float)
    bias_scale = input_scale * weight_scale
    requantize_scale = bias_scale / get_arg(quant_node, "scale", float)
    requantize_scale_t = (
        requantize_scale
        if isinstance(requantize_scale, torch.Tensor)
        else torch.tensor([requantize_scale])
    )
    out_multiplier, out_shift = quantize_tensor_multiplier(requantize_scale_t)
    if dq_bias is not None:
        bias_q = get_arg(dq_bias, "input", fx.Node)
    else:
        # Cadence quantized linear ops require a non-optional bias argument.
        weight_node = get_arg(dq_weight, "input", fx.Node)
        with gm.graph.inserting_before(op_node):
            # the helper only needs a representative scalar, it fills with zeros
            bias_q = create_zero_bias_int32(
                gm,
                weight_node,
                float(bias_scale.max()) if per_channel else bias_scale,
            )
    final_weight = (
        weight_q if weight_q is not None else get_arg(dq_weight, "input", fx.Node)
    )
    args = (get_arg(dq_input, "input", fx.Node), final_weight, bias_q)
    kwargs = {
        "src_zero_point": get_arg(dq_input, "zero_point", int),
        "out_zero_point": get_arg(quant_node, "zero_point", int),
        "offset": None,
    }
    if per_channel:
        # The tensor-qparam overload carries the qparams as constant tensors,
        # which is what per-channel needs. quantized_linear has no bias_scale
        # arg, so the scale only survives through out_multiplier/out_shift.
        replacement_op = tensor_qparam_overload(replacement_op)
        kwargs["weight_zero_point"] = add_constant_placeholder(
            gm, get_weight_zero_point(gm, dq_weight), op_node, "wzp"
        )
        kwargs["out_multiplier"] = add_constant_placeholder(
            gm, out_multiplier.to(torch.int32), op_node, "out_multiplier"
        )
        kwargs["out_shift"] = add_constant_placeholder(
            gm, out_shift.to(torch.int32), op_node, "out_shift"
        )
    else:
        kwargs["weight_zero_point"] = get_arg(dq_weight, "zero_point", int)
        kwargs["out_multiplier"] = out_multiplier[0].item()
        kwargs["out_shift"] = out_shift[0].item()
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
