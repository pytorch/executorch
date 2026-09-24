# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from typing import cast

import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.fused_quant.colorer import is_colored
from executorch.backends.fused_quant.graph_utils import (
    compute_meta_val,
    get_qparams_from_node,
    get_scale,
    get_zero_point,
    split_fused_arg_names,
)
from executorch.backends.fused_quant.ops import QuantParamsStruct
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch import fx
from torch._ops import OpOverload
from torch.export import ExportedProgram

_OpOverload = EdgeOpOverload | OpOverload


def _copy_meta_and_compute(node: fx.Node, source: fx.Node) -> None:
    node.meta = source.meta.copy()
    node.meta["val"] = compute_meta_val(node)


def _qparam_granularity(
    qparams: QuantParamsStruct[fx.Node],
    tensor_value: torch.Tensor,
) -> str:
    qparams.validate()
    scale_value = qparams.scale.meta["val"]
    if qparams.is_per_tensor():
        return "per_tensor"
    axis = qparams.channel_axis()
    if (
        qparams.is_per_channel()
        and scale_value.ndim == tensor_value.ndim
        and axis is not None
        and scale_value.shape[axis] == tensor_value.shape[axis]
    ):
        return "per_channel"
    raise ValueError(
        "Cannot decompose grouped or blockwise fused quantization with scale shape "
        f"{tuple(scale_value.shape)} for tensor shape {tuple(tensor_value.shape)}; "
        "only per-tensor and per-channel quantization are supported"
    )


def _per_channel_qparam_args(
    graph: fx.Graph,
    qparams: QuantParamsStruct[fx.Node],
) -> tuple[fx.Node, fx.Node, int]:
    axis = qparams.channel_axis()
    assert axis is not None
    flattened: list[fx.Node] = []
    for qparam in (qparams.scale, qparams.zero_point):
        view = graph.call_function(
            exir_ops.edge.aten.view_copy.default,
            args=(qparam, [-1]),
        )
        _copy_meta_and_compute(view, qparam)
        flattened.append(view)
    return flattened[0], flattened[1], axis


def _dequantize(
    exported_program: ExportedProgram,
    graph: fx.Graph,
    tensor: fx.Node | None,
    qparams: QuantParamsStruct[fx.Node] | None,
) -> fx.Node | None:
    if tensor is None or qparams is None:
        return tensor

    if _qparam_granularity(qparams, tensor.meta["val"]) == "per_tensor":
        target = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        args = (
            tensor,
            get_scale(exported_program, qparams),
            get_zero_point(exported_program, qparams),
            qparams.quant_min,
            qparams.quant_max,
            tensor.meta["val"].dtype,
        )
    else:
        scale, zero_point, axis = _per_channel_qparam_args(graph, qparams)
        target = exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
        args = (
            tensor,
            scale,
            zero_point,
            axis,
            qparams.quant_min,
            qparams.quant_max,
            tensor.meta["val"].dtype,
        )
    dequantized = graph.call_function(
        target,
        args=args,
        kwargs={"out_dtype": qparams.dtype},
    )
    _copy_meta_and_compute(dequantized, tensor)
    return dequantized


def _quantize(
    exported_program: ExportedProgram,
    graph: fx.Graph,
    tensor: fx.Node,
    qparams: QuantParamsStruct[fx.Node] | None,
    source: fx.Node,
) -> fx.Node:
    if qparams is None:
        return tensor

    if _qparam_granularity(qparams, tensor.meta["val"]) == "per_tensor":
        target = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        args = (
            tensor,
            get_scale(exported_program, qparams),
            get_zero_point(exported_program, qparams),
            qparams.quant_min,
            qparams.quant_max,
            qparams.dtype,
        )
    else:
        scale, zero_point, axis = _per_channel_qparam_args(graph, qparams)
        target = exir_ops.edge.quantized_decomposed.quantize_per_channel.default
        args = (
            tensor,
            scale,
            zero_point,
            axis,
            qparams.quant_min,
            qparams.quant_max,
            qparams.dtype,
        )
    quantized = graph.call_function(target, args=args)
    _copy_meta_and_compute(quantized, source)
    return quantized


def _get_aten_target(
    fused_target: _OpOverload,
    fused_name: str,
    input_count: int,
    extra_names: list[str],
) -> EdgeOpOverload:
    packet = getattr(exir_ops.edge.aten, fused_name, None)
    if packet is None:
        packet = getattr(exir_ops.edge.aten, fused_name.replace("_", ""), None)
    if packet is None:
        raise ValueError(f"No ATen operator matches fused_quant::{fused_name}")

    overload_name = fused_target._schema.overload_name
    candidate_names = [overload_name, "default", "Tensor"]
    candidate_names.extend(getattr(packet, "_overload_names", ()))

    for candidate_name in (name for name in dict.fromkeys(candidate_names) if name):
        candidate = getattr(packet, candidate_name, None)
        if candidate is None:
            continue
        arguments = candidate._schema.arguments
        if (
            not any(argument.name == "out" for argument in arguments)
            and len(arguments) == input_count + len(extra_names)
            and all(
                "Tensor" in str(argument.type) for argument in arguments[:input_count]
            )
            and [argument.name for argument in arguments[input_count:]] == extra_names
        ):
            return candidate
    raise ValueError(f"No functional ATen overload matches fused_quant::{fused_name}")


def _replace_multi_output(
    exported_program: ExportedProgram,
    node: fx.Node,
    aten_node: fx.Node,
    out_qparams: QuantParamsStruct[fx.Node] | None,
) -> None:
    users = list(node.users)
    if any(
        user.target is not operator.getitem or not isinstance(user.args[1], int)
        for user in users
    ):
        raise ValueError(
            f"Multi-output fused op {node.target} must be consumed through getitem"
        )

    graph = node.graph
    replacements: dict[int, fx.Node] = {}
    for user in users:
        output_index = cast(int, user.args[1])
        replacement = replacements.get(output_index)
        if replacement is None:
            replacement = graph.call_function(
                operator.getitem,
                args=(aten_node, output_index),
            )
            _copy_meta_and_compute(replacement, user)
            if output_index == 0:
                replacement = _quantize(
                    exported_program, graph, replacement, out_qparams, user
                )
            replacements[output_index] = replacement
        user.replace_all_uses_with(replacement)
        graph.erase_node(user)
    graph.erase_node(node)


def _decompose_requantize(
    exported_program: ExportedProgram,
    node: fx.Node,
) -> None:
    graph = node.graph
    inp = get_arg(node, "inp", fx.Node)
    inp_qparams = get_qparams_from_node(node, "inp")
    out_qparams = get_qparams_from_node(node, "out")
    assert inp_qparams is not None and out_qparams is not None
    _qparam_granularity(inp_qparams, inp.meta["val"])
    _qparam_granularity(out_qparams, node.meta["val"])

    with graph.inserting_before(node):
        dequantized = _dequantize(exported_program, graph, inp, inp_qparams)
        assert dequantized is not None
        quantized = _quantize(exported_program, graph, dequantized, out_qparams, node)
    node.replace_all_uses_with(quantized)
    graph.erase_node(node)


def _decompose_fused_op(exported_program: ExportedProgram, node: fx.Node) -> None:
    fused_target = cast(_OpOverload, node.target)
    fused_name = fused_target._schema.name.split("::", 1)[1]
    if fused_name == "requantize":
        _decompose_requantize(exported_program, node)
        return

    input_names, extra_names = split_fused_arg_names(fused_target)
    input_tensors = [get_arg(node, name, fx.Node | None) for name in input_names]
    input_qparams = [get_qparams_from_node(node, name) for name in input_names]
    out_qparams = get_qparams_from_node(node, "out")
    for tensor, qparams in zip(input_tensors, input_qparams):
        if qparams is not None:
            assert tensor is not None
            _qparam_granularity(qparams, tensor.meta["val"])
    if out_qparams is not None:
        output_value = node.meta["val"]
        if isinstance(output_value, (tuple, list)):
            output_value = output_value[0]
        _qparam_granularity(out_qparams, output_value)
    graph = node.graph

    with graph.inserting_before(node):
        inputs: list[fx.node.Argument] = []
        for tensor, qparams in zip(input_tensors, input_qparams):
            inputs.append(_dequantize(exported_program, graph, tensor, qparams))

        extra_values = [get_arg(node, name) for name in extra_names]
        aten_target = _get_aten_target(
            fused_target, fused_name, len(inputs), extra_names
        )
        args = list(inputs)
        kwargs: dict[str, fx.node.Argument] = {}
        for argument, value in zip(
            aten_target._schema.arguments[len(inputs) :],
            extra_values,
            strict=True,
        ):
            if argument.kwarg_only:
                kwargs[argument.name] = value
            else:
                args.append(value)
        aten_node = graph.call_function(aten_target, args=tuple(args), kwargs=kwargs)
        _copy_meta_and_compute(aten_node, node)

        if isinstance(node.meta.get("val"), (tuple, list)):
            _replace_multi_output(exported_program, node, aten_node, out_qparams)
            return

        output = _quantize(exported_program, graph, aten_node, out_qparams, node)

    node.replace_all_uses_with(output)
    graph.erase_node(node)


class DecomposeFusedQuant(ExportedProgramPassBase):
    """Decompose fused-quant ops into per-tensor/channel dq, ATen, and q ops.

    Colored nodes are left alone: a backend that claimed an op is going to lower
    it itself, and defusing it would take it away. See
    :class:`executorch.backends.fused_quant.colorer.ColorerBase`.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or not isinstance(node.target, (EdgeOpOverload, OpOverload))
                or node.target.namespace != "fused_quant"
                or is_colored(node)
            ):
                continue
            _decompose_fused_op(exported_program, node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
        return ExportedProgramPassResult(exported_program, modified)

    def ensures(self, exported_program: ExportedProgram) -> None:
        exported_program.validate()
