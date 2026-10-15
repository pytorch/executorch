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
    get_qparams_from_node,
    get_scale,
    get_zero_point,
    split_fused_arg_names,
)
from executorch.backends.fused_quant.ops import QuantParamsStruct
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import (
    add_constant,
    compute_meta_val,
    delete_constant_placeholder,
    get_constant,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch import fx
from torch._C import Argument
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import UnsupportedOperatorException
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_OpOverload = EdgeOpOverload | OpOverload


def _copy_meta_and_compute(node: fx.Node, source: fx.Node) -> None:
    node.meta = source.meta.copy()
    node.meta["val"] = compute_meta_val(node)


def _aten_meta_from_fused(
    aten_node: fx.Node,
    fused_node: fx.Node,
    out_qparams: QuantParamsStruct[fx.Node] | None,
    float_dtype: torch.dtype,
) -> None:
    """Set the meta of the ATen node standing in for a fused op.

    Some ATen ops (e.g. ``_masked_softmax``) have no fake kernel, so their meta
    cannot be computed. The fused op's own fake kernel already has the shape;
    only the dtype differs when the output was quantized.
    """
    aten_node.meta = fused_node.meta.copy()
    try:
        aten_node.meta["val"] = compute_meta_val(aten_node)
    except UnsupportedOperatorException:
        fused_val = fused_node.meta["val"]
        if isinstance(fused_val, (tuple, list)):
            raise
        aten_node.meta["val"] = (
            fused_val.to(float_dtype) if out_qparams is not None else fused_val
        )


def _qparam_granularity(
    qparams: QuantParamsStruct[fx.Node],
    tensor_value: torch.Tensor,
) -> str:
    qparams.validate()
    scale_value = qparams.scale.meta["val"]
    if qparams.is_per_tensor():
        return "per_tensor"
    if qparams.is_per_channel(tensor_value.shape):
        return "per_channel"
    raise ValueError(
        "Cannot decompose grouped or blockwise fused quantization with scale shape "
        f"{tuple(scale_value.shape)} for tensor shape {tuple(tensor_value.shape)}; "
        "only per-tensor and per-channel quantization are supported"
    )


def _per_channel_qparam_args(
    exported_program: ExportedProgram,
    graph: fx.Graph,
    qparams: QuantParamsStruct[fx.Node],
) -> tuple[fx.Node, fx.Node, int]:
    """The 1-D scale/zero_point a per-channel q/dq op takes, plus its axis.

    Constant qparams become new 1-D constants rather than a view of the
    full-rank one, so partitioners that require per-channel qparams to be
    placeholders can consume the decomposed graph without a constant fold.
    """
    axis = qparams.channel_axis()
    assert axis is not None
    flattened: list[fx.Node] = []
    for qparam in (qparams.scale, qparams.zero_point):
        value = get_constant(exported_program, qparam)
        if value is not None:
            flattened.append(
                add_constant(
                    exported_program,
                    f"{qparam.name}_flat",
                    value.reshape(-1),
                    qparam,
                    InputKind.CONSTANT_TENSOR,
                )
            )
            continue
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
        scale, zero_point, axis = _per_channel_qparam_args(
            exported_program, graph, qparams
        )
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
        scale, zero_point, axis = _per_channel_qparam_args(
            exported_program, graph, qparams
        )
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

    with graph.inserting_before(node):
        dequantized = _dequantize(exported_program, graph, inp, inp_qparams)
        assert dequantized is not None
        quantized = _quantize(exported_program, graph, dequantized, out_qparams, node)
    node.replace_all_uses_with(quantized)
    graph.erase_node(node)


def _is_schema_default(argument: Argument, value: fx.node.Argument) -> bool:
    if not argument.has_default_value():
        return False
    try:
        return bool(value == argument.default_value)
    except (RuntimeError, ValueError):
        # Comparison on tensor-valued arguments is not a scalar bool.
        return False


def _qparam_blocks(
    node: fx.Node,
) -> list[tuple[QuantParamsStruct[fx.Node], torch.Tensor]]:
    """Every qparams block of a fused node, with the tensor value it applies to."""
    input_names, _ = split_fused_arg_names(cast(_OpOverload, node.target))
    blocks: list[tuple[QuantParamsStruct[fx.Node], torch.Tensor]] = []
    for name in input_names:
        qparams = get_qparams_from_node(node, name)
        if qparams is not None:
            blocks.append((qparams, get_arg(node, name, fx.Node).meta["val"]))
    out_qparams = get_qparams_from_node(node, "out")
    if out_qparams is not None:
        output_value = node.meta["val"]
        if isinstance(output_value, (tuple, list)):
            output_value = output_value[0]
        blocks.append((out_qparams, output_value))
    return blocks


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
            if not argument.kwarg_only:
                args.append(value)
            elif not _is_schema_default(argument, value):
                # A kwarg pinned to its default is one that torch.export would
                # have left off the node entirely. Emitting it anyway makes the
                # rebuilt node wider than the ATen op it is standing in for, and
                # backends that forward kwargs verbatim choke on the extra
                # argument -- e.g. alpha=1.0 reaching the two-operand tosa.SUB.
                kwargs[argument.name] = value
        aten_node = graph.call_function(aten_target, args=tuple(args), kwargs=kwargs)
        float_dtype = next(
            (
                cast(torch.Tensor, arg.meta["val"]).dtype
                for arg in inputs
                if isinstance(arg, fx.Node)
                and cast(torch.Tensor, arg.meta["val"]).is_floating_point()
            ),
            torch.float32,
        )
        _aten_meta_from_fused(aten_node, node, out_qparams, float_dtype)

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
        nodes = [
            node
            for node in graph.nodes
            if is_fused_quant_node(node) and not is_colored(node)
        ]
        if not nodes:
            return ExportedProgramPassResult(exported_program, False)

        # Validate everything first so an unsupported node fails the pass before
        # any of the graph has been rewritten.
        qparam_placeholders: set[fx.Node] = set()
        for node in nodes:
            for qparams, value in _qparam_blocks(node):
                _qparam_granularity(qparams, value)
                qparam_placeholders.update(
                    qparam
                    for qparam in (qparams.scale, qparams.zero_point)
                    if qparam.op == "placeholder"
                )

        for node in nodes:
            _decompose_fused_op(exported_program, node)

        graph.eliminate_dead_code()
        for placeholder in qparam_placeholders:
            if (
                not placeholder.users
                and get_constant(exported_program, placeholder) is not None
            ):
                delete_constant_placeholder(exported_program, placeholder)
        graph.lint()
        graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)

    def ensures(self, exported_program: ExportedProgram) -> None:
        exported_program.validate()


def is_fused_quant_node(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, (EdgeOpOverload, OpOverload))
        and node.target.namespace == "fused_quant"
    )


class AssertNoFusedQuantOps(ExportedProgramPassBase):
    """Fail if any fused_quant op is left in the graph.

    fused_quant ops are an ahead-of-time IR with no runtime kernels. Run this
    after lowering, once every backend has taken what it claimed and
    :class:`DecomposeFusedQuant` has decomposed the rest, so a leak surfaces here
    rather than as a missing out-variant at ``to_executorch``.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        leaked = [
            node.name
            for node in exported_program.graph_module.graph.nodes
            if is_fused_quant_node(node)
        ]
        if leaked:
            raise RuntimeError(
                f"fused_quant ops remain after lowering: {leaked}. Colored nodes a "
                "backend did not lower must be decomposed with DecomposeFusedQuant "
                "after to_backend."
            )
        return ExportedProgramPassResult(exported_program, False)
