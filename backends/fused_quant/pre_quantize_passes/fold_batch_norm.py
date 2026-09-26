# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import cast

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    get_constant,
    get_input_kind,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_BATCH_NORM = torch.ops.aten._native_batch_norm_legit_no_training.default
_CONVOLUTION = torch.ops.aten.convolution.default
_CONV_TARGETS = (
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv3d.default,
    _CONVOLUTION,
)
_SLICE_TARGETS = (
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
)
_CONTIGUOUS_OR_CLONE_TARGETS = (
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
)


@dataclass(frozen=True)
class _FoldConstants:
    conv_weight_node: fx.Node
    conv_bias_node: fx.Node | None
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    bn_weight: torch.Tensor | None
    bn_bias: torch.Tensor | None
    running_mean: torch.Tensor
    running_var: torch.Tensor


def _get_optional_constant(
    exported_program: ExportedProgram,
    node: fx.Node,
    name: str,
) -> tuple[fx.Node | None, torch.Tensor | None]:
    arg = get_arg(node, name)
    if arg is None:
        return None, None
    if not isinstance(arg, fx.Node):
        return None, None
    return arg, get_constant(exported_program, arg)


def _get_fold_constants(
    exported_program: ExportedProgram,
    conv: fx.Node,
    batch_norm: fx.Node,
) -> _FoldConstants | None:
    conv_weight_node, conv_weight = _get_optional_constant(
        exported_program, conv, "weight"
    )
    conv_bias_node, conv_bias = _get_optional_constant(exported_program, conv, "bias")
    bn_weight_node, bn_weight = _get_optional_constant(
        exported_program, batch_norm, "weight"
    )
    bn_bias_node, bn_bias = _get_optional_constant(exported_program, batch_norm, "bias")
    mean_node, running_mean = _get_optional_constant(
        exported_program, batch_norm, "running_mean"
    )
    var_node, running_var = _get_optional_constant(
        exported_program, batch_norm, "running_var"
    )
    required = (
        conv_weight_node,
        conv_weight,
        mean_node,
        running_mean,
        var_node,
        running_var,
    )
    if any(value is None for value in required):
        return None
    optional = (
        (conv_bias_node, conv_bias),
        (bn_weight_node, bn_weight),
        (bn_bias_node, bn_bias),
    )
    if any(node is not None and tensor is None for node, tensor in optional):
        return None
    assert conv_weight_node is not None
    assert conv_weight is not None
    assert running_mean is not None
    assert running_var is not None
    return _FoldConstants(
        conv_weight_node,
        conv_bias_node,
        conv_weight,
        conv_bias,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
    )


def _native_output_accessors(batch_norm: fx.Node) -> list[fx.Node] | None:
    output_accessors: list[fx.Node] = []
    for user in batch_norm.users:
        if user.target is not operator.getitem or not isinstance(user.args[1], int):
            return None
        if user.args[1] == 0:
            output_accessors.append(user)
        elif user.users:
            return None
    return output_accessors or None


def _is_supported_conv(conv: fx.Node) -> bool:
    if conv.op != "call_function" or conv.target not in _CONV_TARGETS:
        return False
    if conv.target is _CONVOLUTION and get_arg(conv, "transposed", bool):
        return False
    return True


def _is_non_channel_slice(node: fx.Node) -> bool:
    if node.op != "call_function" or node.target not in _SLICE_TARGETS:
        return False

    dim = get_arg(node, "dim", int)
    if dim >= 0:
        return dim != 1

    inp = node.args[0]
    if not isinstance(inp, fx.Node):
        return False
    inp_val = inp.meta["val"]
    if not isinstance(inp_val, torch.Tensor):
        return False
    return dim + inp_val.ndim != 1


def _is_contiguous_or_clone(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _CONTIGUOUS_OR_CLONE_TARGETS


def _get_fusion_nodes(batch_norm: fx.Node) -> tuple[fx.Node, fx.Node] | None:
    batch_norm_input = get_arg(batch_norm, "input")
    if not isinstance(batch_norm_input, fx.Node):
        return None
    if _is_supported_conv(batch_norm_input):
        if len(batch_norm_input.users) != 1:
            return None
        return batch_norm_input, batch_norm_input

    replacement = batch_norm_input
    slice_node = batch_norm_input
    if _is_contiguous_or_clone(slice_node):
        if len(slice_node.users) != 1:
            return None
        slice_input = slice_node.args[0]
        if not isinstance(slice_input, fx.Node):
            return None
        slice_node = slice_input

    if not _is_non_channel_slice(slice_node) or len(slice_node.users) != 1:
        return None
    conv = slice_node.args[0]
    if (
        not isinstance(conv, fx.Node)
        or not _is_supported_conv(conv)
        or len(conv.users) != 1
    ):
        return None
    return conv, replacement


def _make_fused_parameters(
    constants: _FoldConstants,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    channels = constants.running_mean.numel()
    channel_shape = (channels,)
    if constants.conv_weight.shape[0] != channels:
        return None
    for tensor in (
        constants.running_mean,
        constants.running_var,
        constants.conv_bias,
        constants.bn_weight,
        constants.bn_bias,
    ):
        if tensor is not None and tensor.shape != channel_shape:
            return None

    scale = torch.rsqrt(constants.running_var + eps)
    if constants.bn_weight is not None:
        scale = scale * constants.bn_weight
    conv_bias = constants.conv_bias
    if conv_bias is None:
        conv_bias = torch.zeros_like(constants.running_mean)
    fused_bias = (conv_bias - constants.running_mean) * scale
    if constants.bn_bias is not None:
        fused_bias = fused_bias + constants.bn_bias
    scale_shape = (channels, *([1] * (constants.conv_weight.ndim - 1)))
    fused_weight = constants.conv_weight * scale.reshape(scale_shape)
    dtype = constants.conv_weight.dtype
    return (
        fused_weight.to(dtype=dtype).detach().contiguous(),
        fused_bias.to(dtype=dtype).detach().contiguous(),
    )


def _constant_kind(
    exported_program: ExportedProgram,
    node: fx.Node,
) -> InputKind:
    return get_input_kind(exported_program, node) or InputKind.CONSTANT_TENSOR


def _insert_folded_conv(
    exported_program: ExportedProgram,
    conv: fx.Node,
    batch_norm: fx.Node,
    constants: _FoldConstants,
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor,
) -> fx.Node:
    weight = add_constant(
        exported_program,
        f"{batch_norm.name}_folded_weight",
        fused_weight,
        batch_norm,
        kind=_constant_kind(exported_program, constants.conv_weight_node),
    )
    bias_source = constants.conv_bias_node or constants.conv_weight_node
    bias = add_constant(
        exported_program,
        f"{batch_norm.name}_folded_bias",
        fused_bias,
        batch_norm,
        kind=_constant_kind(exported_program, bias_source),
    )
    owning_module = conv.graph.owning_module
    if owning_module is None:
        raise RuntimeError(f"Convolution {conv} does not belong to a graph module")
    normalized_args = conv.normalized_arguments(
        owning_module,
        normalize_to_only_use_kwargs=True,
    )
    if normalized_args is None:
        raise RuntimeError(f"Could not normalize convolution arguments for {conv}")
    conv_kwargs = dict(normalized_args.kwargs)
    conv_input = conv_kwargs.pop("input")
    conv_kwargs.pop("weight")
    conv_kwargs.pop("bias")
    with batch_norm.graph.inserting_before(conv):
        folded_conv = batch_norm.graph.call_function(
            cast(OpOverload, conv.target),
            args=(conv_input, weight, bias),
            kwargs=conv_kwargs,
        )
        folded_conv.meta = conv.meta.copy()
    return folded_conv


def _erase_folded_pattern(
    conv: fx.Node,
    batch_norm: fx.Node,
    output_accessors: list[fx.Node],
    replacement: fx.Node,
    folded_conv: fx.Node,
) -> None:
    graph = batch_norm.graph
    conv.replace_all_uses_with(folded_conv)
    if replacement is conv:
        replacement = folded_conv
    for accessor in output_accessors:
        accessor.replace_all_uses_with(replacement)
    for accessor in list(batch_norm.users):
        graph.erase_node(accessor)
    graph.erase_node(batch_norm)
    graph.erase_node(conv)


class FoldBatchNorm(ExportedProgramPassBase):
    """Fold inference BatchNorm parameters into an upstream convolution."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        for batch_norm in graph.find_nodes(op="call_function", target=_BATCH_NORM):
            modified |= self._fold(exported_program, batch_norm)

        if modified:
            exported_program = constant_prop_pass(exported_program)
        return ExportedProgramPassResult(exported_program, modified)

    def _fold(
        self,
        exported_program: ExportedProgram,
        batch_norm: fx.Node,
    ) -> bool:
        output_accessors = _native_output_accessors(batch_norm)
        fusion_nodes = _get_fusion_nodes(batch_norm)
        if output_accessors is None or fusion_nodes is None:
            return False
        conv, replacement = fusion_nodes

        constants = _get_fold_constants(exported_program, conv, batch_norm)
        if constants is None:
            return False
        fused_parameters = _make_fused_parameters(
            constants,
            get_arg(batch_norm, "eps", float),
        )
        if fused_parameters is None:
            return False
        fused_weight, fused_bias = fused_parameters
        folded_conv = _insert_folded_conv(
            exported_program,
            conv,
            batch_norm,
            constants,
            fused_weight,
            fused_bias,
        )
        _erase_folded_pattern(
            conv,
            batch_norm,
            output_accessors,
            replacement,
            folded_conv,
        )
        return True
