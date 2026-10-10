# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
from operator import attrgetter
from typing import Any, cast, Optional

import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import (
    is_affine_quant_node,
    is_dequantize_node,
    is_per_channel_quant_node,
    is_per_tensor_quant_node,
    is_quantize_node,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from torch import fx
from torch._export.utils import _detect_fake_mode_from_gm
from torch._ops import OpOverload

_NULL_QPARAMS: list[torch.fx.node.Argument] = [None, None, torch.uint8, 0, 0]

QParamsFlat = tuple[fx.Node, fx.Node, torch.dtype, int, int]


def _resolve_const_tensor(
    node: fx.Node, owning_module: torch.nn.Module
) -> torch.Tensor:
    """Resolve a frozen get_attr constant to its backing tensor."""
    assert node.op == "get_attr", (
        f"Expected a get_attr constant, got op={node.op} for {node.name}"
    )
    return attrgetter(str(node.target))(owning_module)


def _add_scalar_constant(
    graph: fx.Graph, value: float | int, dtype: torch.dtype, name: str
) -> fx.Node:
    """Register a 0-dim scalar qparam as a frozen module attribute, return a get_attr.

    ``name`` is a preferred base; ``graph.get_attr`` uniquifies it against the
    graph, and the attribute is registered under the resulting unique name.
    """
    gm = graph.owning_module
    assert gm is not None
    # get_attr uniquifies node.name but not node.target, so sync target and
    # register the attribute under the generated name.
    const_node = graph.get_attr(name)
    const_node.target = const_node.name
    setattr(gm, const_node.name, torch.tensor(value, dtype=dtype))
    return const_node


def _create_qparams_from_node(
    node: fx.Node, graph: fx.Graph, out_dtype: torch.dtype
) -> QParamsFlat:
    """Extract QuantParams from a quantize or dequantize node.

    For per-tensor quantization, creates scalar tensors for scale and zero_point.
    For per-channel quantization, uses the existing tensor nodes and extracts axis.

    Args:
        node: A quantize or dequantize node (per-tensor or per-channel).
        graph: The graph to insert new nodes into.
        out_dtype: The dtype after applying quantization/dequantization

    Returns:
        A QuantParamsNode tuple.
    """
    quant_min = get_arg(node, "quant_min")
    quant_max = get_arg(node, "quant_max")

    if is_affine_quant_node(node):
        # Affine nodes already carry a full-rank scale/zero_point whose shape
        # encodes the block layout (block_size[i] = tensor.shape[i] //
        # scale.shape[i]) -- exactly QuantParamsStruct's convention -- so the
        # existing scale/zero_point nodes pass straight through, with no axis and
        # no reshape (this covers per-tensor/channel/group/blockwise uniformly).
        scale_node = get_arg(node, "scale", fx.Node)
        zp_node = get_arg(node, "zero_point", fx.Node)
        # pyrefly: ignore [bad-return]
        return (scale_node, zp_node, out_dtype, quant_min, quant_max)

    if is_per_tensor_quant_node(node):
        scale_val = get_arg(node, "scale", float)
        zp_val = get_arg(node, "zero_point", int)
        with graph.inserting_before(node):
            # Per-tensor scale/zero_point arrive as inline scalars; lift them to
            # frozen constants so they match the per-channel qparam representation.
            scale_node = _add_scalar_constant(graph, scale_val, torch.float32, "_scale")
            zp_node = _add_scalar_constant(graph, zp_val, torch.int64, "_zero_point")
        # pyrefly: ignore [bad-return]
        return (scale_node, zp_node, out_dtype, quant_min, quant_max)
    elif is_per_channel_quant_node(node):
        # Per-channel: QuantParamsStruct encodes granularity by the scale's shape
        # (no axis), so reshape the 1D [C] scale/zero_point to full rank --
        # [1, .., C, .., 1] with C at `axis` -- giving block_size 1 along the
        # channel axis and full along the rest.
        scale_node = get_arg(node, "scales", fx.Node)
        zp_node = get_arg(node, "zero_points", fx.Node)
        axis = get_arg(node, "axis", int)
        input_node = get_arg(node, "input", fx.Node)
        # The weight and scale/zero_point are frozen get_attr constants (convert
        # folds the quantize into a frozen param); resolve them off the owning module.
        owning_module = graph.owning_module
        assert owning_module is not None
        ndim = _resolve_const_tensor(input_node, owning_module).ndim
        scale_tensor = _resolve_const_tensor(scale_node, owning_module)
        zp_tensor = _resolve_const_tensor(zp_node, owning_module)
        view_shape = [1] * ndim
        view_shape[axis] = -1
        with graph.inserting_before(node):
            fake_mode = _detect_fake_mode_from_gm(owning_module)
            assert fake_mode, "Expected fake tensor mode!"
            scale_view = graph.call_function(
                torch.ops.aten.view.default, args=(scale_node, view_shape)
            )
            zp_view = graph.call_function(
                torch.ops.aten.view.default, args=(zp_node, view_shape)
            )
            with fake_mode:
                scale_view.meta["val"] = fake_mode.from_tensor(
                    scale_tensor.reshape(view_shape)
                )
                zp_view.meta["val"] = fake_mode.from_tensor(
                    zp_tensor.reshape(view_shape)
                )
        # pyrefly: ignore [bad-return]
        return (scale_view, zp_view, out_dtype, quant_min, quant_max)

    raise ValueError(f"Unsupported quantize node: {node}")


def create_qparams_from_dequant_node(
    dequant_node: fx.Node,
    graph: fx.Graph,
) -> QParamsFlat:
    """Extract QuantParams from a dequantize node."""
    assert is_dequantize_node(dequant_node)
    # Affine dequant names its result dtype `output_dtype`; the quantized_decomposed
    # per-tensor/channel ops name it `out_dtype`.
    dtype_arg = "output_dtype" if is_affine_quant_node(dequant_node) else "out_dtype"
    out_dtype = get_arg(dequant_node, dtype_arg, Optional[torch.dtype]) or torch.float32
    return _create_qparams_from_node(dequant_node, graph, out_dtype)


def create_qparams_from_quant_node(
    quant_node: fx.Node,
    graph: fx.Graph,
) -> QParamsFlat:
    """Extract QuantParams from a quantize node."""
    assert is_quantize_node(quant_node)
    # Affine quant names the quantized dtype `output_dtype`; quantized_decomposed
    # per-tensor/channel ops name it `dtype`.
    dtype_arg = "output_dtype" if is_affine_quant_node(quant_node) else "dtype"
    out_dtype = get_arg(quant_node, dtype_arg, torch.dtype)
    return _create_qparams_from_node(quant_node, graph, out_dtype)


def arg_names(target: OpOverload) -> list[str]:
    """The op's schema argument names, in schema order.

    The first arg is exposed as self on some schemas but as input to get_arg
    (torch.fx normalizes self -> input since self can't be a keyword-argument
    name); this normalizes to the get_arg spelling so edge names resolve directly.
    """
    return ["input" if a.name == "self" else a.name for a in target._schema.arguments]


def output_node(node: fx.Node, output_index: int = 0) -> Optional[fx.Node]:
    """The node carrying output output_index of node.

    A single-output op *is* its output; a multi-output op (native_layer_norm,
    max_pool2d_with_indices) unpacks each output via getitem(node, output_index).
    Returns None when that output node is absent.
    """
    if not isinstance(node.meta.get("val"), (tuple, list)):
        return node
    for user in node.users:
        if user.target is operator.getitem and user.args[1] == output_index:
            return user
    return None


def _get_extra_args(
    node: fx.Node,
    tensor_input_names: set[str],
    fused_target: OpOverload,
) -> list[torch.fx.node.Argument]:
    """Args of node that are neither tensor inputs nor qparam-bearing.

    Everything whose (normalized) schema name is not a tensor input
    (stride/padding/groups, alpha, a scalar operand, normalized_shape/weight/bias,
    dim/mask_type, ...) is threaded into the fused op by matching schema arg name.
    A name the fused op does not accept is an unsupported lowering and raises.
    """
    fused_arg_names = set(arg_names(fused_target))

    extra_args: list[torch.fx.node.Argument] = []
    for name in arg_names(cast(OpOverload, node.target)):
        if name in tensor_input_names:
            continue
        if name not in fused_arg_names:
            raise ValueError(
                f"Unsupported ATen arg: {name} from {node.target} "
                f"when fusing to {fused_target}"
            )
        extra_args.append(get_arg(node, name))
    return extra_args


def _build_fused_args(
    node: fx.Node,
    dequant_inputs: dict[str, fx.Node],
    quant_outputs: dict[int, fx.Node],
    fused_target: OpOverload,
    ordered_input_names: list[str],
    output_indices: list[int],
) -> list[torch.fx.node.Argument]:
    """Build the flat fused-op arg list:
        (tensor_inputs..., per-input qparams..., per-output qparams..., extra_args...)

    Tensor inputs are emitted in schema order, one qparams block each (from the
    feeding dequant, or null when the edge is unquantized), then one output qparams
    block per entry in output_indices (ascending; from the consuming quantize,
    or null), then the remaining args by name.

    Newly created qparams (scale/zero_point) nodes may not be topologically
    ordered relative to the fused op; the caller relies on a later legalize_graph.
    """
    graph = node.graph
    tensor_inputs: list[torch.fx.node.Argument] = []
    input_qparams: list[Optional[QParamsFlat]] = []
    for name in ordered_input_names:
        dequant = dequant_inputs.get(name)
        if dequant is not None:
            tensor_inputs.append(get_arg(dequant, "input"))
            input_qparams.append(create_qparams_from_dequant_node(dequant, graph))
        else:
            tensor_inputs.append(get_arg(node, name))
            input_qparams.append(None)

    out_qparams_blocks: list[Optional[QParamsFlat]] = []
    for output_index in output_indices:
        quant_node = quant_outputs.get(output_index)
        out_qparams_blocks.append(
            create_qparams_from_quant_node(quant_node, graph)
            if quant_node is not None
            else None
        )

    extra_args = _get_extra_args(node, set(ordered_input_names), fused_target)

    fused_args_list: list[torch.fx.node.Argument] = []
    fused_args_list.extend(tensor_inputs)
    for qp in [*input_qparams, *out_qparams_blocks]:
        fused_args_list.extend(qp if qp is not None else _NULL_QPARAMS)
    fused_args_list.extend(extra_args)
    return fused_args_list


def _rewire_multi_output(
    graph: fx.Graph,
    node: fx.Node,
    fused_op: fx.Node,
    quant_outputs: dict[int, fx.Node],
) -> None:
    """Rewire a multi-output op to its tuple-returning fused replacement.

    The fused op returns a same-arity tuple, so the existing getitem accessors are
    repointed from node to fused_op. For each quantized output the getitem
    now already yields the quantized dtype, so its downstream quantize is redundant
    and is collapsed away.
    """
    node.replace_all_uses_with(fused_op)
    for quant_node in quant_outputs.values():
        getitem = quant_node.all_input_nodes[0]
        getitem.meta["val"] = quant_node.meta["val"]
        quant_node.replace_all_uses_with(getitem)
        graph.erase_node(quant_node)
    graph.erase_node(node)


def _rewire_single_output(
    graph: fx.Graph,
    node: fx.Node,
    fused_op: fx.Node,
    quant_outputs: dict[int, fx.Node],
) -> None:
    """Rewire a single-output op: replace its quantize user (or the op) with fused_op."""
    quant_out = quant_outputs.get(0)
    if quant_out is not None:
        quant_out.replace_all_uses_with(fused_op)
        graph.erase_node(quant_out)
    else:
        node.replace_all_uses_with(fused_op)
    graph.erase_node(node)


def _fused_meta(
    node: fx.Node,
    quant_outputs: dict[int, fx.Node],
) -> dict[str, Any]:
    meta = node.meta.copy()
    output_val = meta["val"]
    if isinstance(output_val, (tuple, list)):
        output_values = list(output_val)
        for output_index, quant_output in quant_outputs.items():
            output_values[output_index] = quant_output.meta["val"]
        meta["val"] = (
            tuple(output_values) if isinstance(output_val, tuple) else output_values
        )
    elif quant_outputs:
        assert len(quant_outputs) == 1
        quant_output = next(iter(quant_outputs.values()))
        meta["val"] = quant_output.meta["val"]
    return meta


def fuse_aten(
    node: fx.Node,
    fused_target: OpOverload,
    activation_names: tuple[str, ...],
    weight_names: tuple[str, ...] = (),
    other_names: tuple[str, ...] = (),
    output_indices: tuple[int, ...] = (0,),
) -> fx.Node:
    """Generic fusion for an ATen op whose fused replacement follows the
    fused_quant arg convention: (tensor_inputs..., per-input qparams...,
    per-output qparams..., extra_args...).

    Detects the dequant -> op -> quant pattern around node and replaces it with
    a single fused_target node, which is returned.
    """
    tensor_input_names = {*activation_names, *weight_names, *other_names}
    # Tensor inputs in schema order (so the fused op's positional prefix lines up);
    # this also drops any name the op's schema does not have.
    ordered_input_names = [
        n for n in arg_names(cast(OpOverload, node.target)) if n in tensor_input_names
    ]
    output_indices = tuple(sorted(set(output_indices)))

    # Detect the dequant -> op -> quant structure. Every tensor input is checked for a
    # feeding dequant.
    dequant_inputs: dict[str, fx.Node] = {}
    for name in ordered_input_names:
        arg = get_arg(node, name)
        if isinstance(arg, fx.Node) and is_dequantize_node(arg) and len(arg.users) == 1:
            dequant_inputs[name] = arg

    quant_outputs: dict[int, fx.Node] = {}
    for output_index in output_indices:
        accessor = output_node(node, output_index)
        if accessor is None or len(accessor.users) != 1:
            continue
        candidate = next(iter(accessor.users.keys()))
        if is_quantize_node(candidate) and len(candidate.all_input_nodes) == 1:
            quant_outputs[output_index] = candidate

    if not dequant_inputs and not quant_outputs:
        raise ValueError(
            f"fuse_aten: {node.target} node {node.name!r} has no quantized input "
            "or output to fuse (expected a surrounding dequantize/quantize)."
        )

    graph = node.graph
    fused_meta = _fused_meta(node, quant_outputs)
    multi_output = isinstance(node.meta["val"], (tuple, list))

    fused_args_list = _build_fused_args(
        node,
        dequant_inputs,
        quant_outputs,
        fused_target,
        ordered_input_names,
        list(output_indices),
    )

    # Insert the fused op before node. Newly created qparams nodes may land after
    # it (created in front of downstream quantize nodes), leaving the graph
    # topologically invalid here; the caller relegalizes once after all fusions.
    with graph.inserting_before(node):
        fused_op = graph.call_function(fused_target, args=tuple(fused_args_list))
    fused_op.meta = fused_meta

    if multi_output:
        _rewire_multi_output(graph, node, fused_op, quant_outputs)
    else:
        _rewire_single_output(graph, node, fused_op, quant_outputs)

    # set(): the same dequant can feed two inputs (e.g. x * x), so dedup before
    # erasing to avoid a redundant erase of an already-removed node.
    for dequant_node in set(dequant_inputs.values()):
        if not dequant_node.users:
            graph.erase_node(dequant_node)

    return fused_op
