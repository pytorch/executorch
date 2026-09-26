# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    compute_meta_val,
    get_constant,
    get_input_kind,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_BATCH_NORM_TARGET = torch.ops.aten._native_batch_norm_legit_no_training.default

_CONVOLUTION = torch.ops.aten.convolution.default


@dataclass(frozen=True)
class _BatchNormConstants:
    weight_node: Optional[fx.Node]
    bias_node: Optional[fx.Node]
    mean_node: fx.Node
    var_node: fx.Node
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]
    running_mean: torch.Tensor
    running_var: torch.Tensor


def _constant_kind(
    exported_program: ExportedProgram,
    *nodes: Optional[fx.Node],
) -> InputKind:
    for node in nodes:
        if node is not None:
            kind = get_input_kind(exported_program, node)
            if kind is not None:
                return kind
    return InputKind.CONSTANT_TENSOR


def _native_output_accessors(batch_norm: fx.Node) -> Optional[list[fx.Node]]:
    """Return output-0 accessors when the tuple's auxiliary outputs are unused."""
    output_accessors: list[fx.Node] = []
    for user in batch_norm.users:
        if user.target is not operator.getitem or not isinstance(user.args[1], int):
            return None
        if user.args[1] == 0:
            output_accessors.append(user)
        elif user.users:
            return None
    return output_accessors or None


def _constant_arg(
    exported_program: ExportedProgram,
    batch_norm: fx.Node,
    name: str,
) -> tuple[Optional[fx.Node], Optional[torch.Tensor]]:
    arg = get_arg(batch_norm, name)
    if arg is None:
        return None, None
    if not isinstance(arg, fx.Node):
        return None, None
    return arg, get_constant(exported_program, arg)


def _get_constants(
    exported_program: ExportedProgram,
    batch_norm: fx.Node,
) -> Optional[_BatchNormConstants]:
    weight_node, weight = _constant_arg(exported_program, batch_norm, "weight")
    bias_node, bias = _constant_arg(exported_program, batch_norm, "bias")
    mean_node, running_mean = _constant_arg(
        exported_program, batch_norm, "running_mean"
    )
    var_node, running_var = _constant_arg(exported_program, batch_norm, "running_var")
    if (
        mean_node is None
        or var_node is None
        or running_mean is None
        or running_var is None
        or (weight_node is not None and weight is None)
        or (bias_node is not None and bias is None)
    ):
        return None

    channels = running_mean.numel()
    expected_shape = (channels,)
    tensors = (running_mean, running_var, weight, bias)
    if any(tensor is not None and tensor.shape != expected_shape for tensor in tensors):
        return None
    return _BatchNormConstants(
        weight_node,
        bias_node,
        mean_node,
        var_node,
        weight,
        bias,
        running_mean,
        running_var,
    )


def _make_fused_parameters(
    constants: _BatchNormConstants,
    eps: float,
    dtype: torch.dtype,
    spatial_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.rsqrt(constants.running_var + eps)
    if constants.weight is not None:
        scale = scale * constants.weight
    fused_bias = -constants.running_mean * scale
    if constants.bias is not None:
        fused_bias = fused_bias + constants.bias

    channels = constants.running_mean.numel()
    fused_weight = scale.reshape(channels, 1, *([1] * spatial_dims))
    return (
        fused_weight.to(dtype=dtype).detach().contiguous(),
        fused_bias.to(dtype=dtype).detach().contiguous(),
    )


def _add_fused_parameters(
    exported_program: ExportedProgram,
    batch_norm: fx.Node,
    constants: _BatchNormConstants,
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor,
) -> tuple[fx.Node, fx.Node]:
    weight_kind = _constant_kind(
        exported_program, constants.weight_node, constants.var_node
    )
    bias_kind = _constant_kind(
        exported_program, constants.bias_node, constants.mean_node
    )
    conv_weight = add_constant(
        exported_program,
        f"{batch_norm.name}_conv_weight",
        fused_weight,
        batch_norm,
        kind=weight_kind,
    )
    conv_bias = add_constant(
        exported_program,
        f"{batch_norm.name}_conv_bias",
        fused_bias,
        batch_norm,
        kind=bias_kind,
    )
    return conv_weight, conv_bias


def _insert_convolution(
    batch_norm: fx.Node,
    inp: fx.Node,
    conv_weight: fx.Node,
    conv_bias: fx.Node,
    output_meta: dict[str, object],
) -> fx.Node:
    graph = batch_norm.graph
    inp_val = inp.meta["val"]
    assert isinstance(inp_val, torch.Tensor)
    spatial_dims = max(inp_val.ndim - 2, 1)
    conv_input = inp
    with graph.inserting_before(batch_norm):
        if inp_val.ndim == 2:
            conv_input = graph.call_function(
                torch.ops.aten.unsqueeze.default,
                args=(inp, -1),
            )
            conv_input.meta = inp.meta.copy()
            conv_input.meta["val"] = inp_val.unsqueeze(-1)

        conv = graph.call_function(
            _CONVOLUTION,
            args=(
                conv_input,
                conv_weight,
                conv_bias,
                [1] * spatial_dims,
                [0] * spatial_dims,
                [1] * spatial_dims,
                False,
                [0] * spatial_dims,
                conv_weight.meta["val"].shape[0],
            ),
        )
        conv.meta = batch_norm.meta.copy()
        conv_val = compute_meta_val(conv)
        assert isinstance(conv_val, torch.Tensor)
        conv.meta["val"] = conv_val

        if inp_val.ndim != 2:
            return conv
        squeeze = graph.call_function(
            torch.ops.aten.squeeze.dim,
            args=(conv, -1),
        )
        squeeze.meta = output_meta.copy()
        squeeze.meta["val"] = conv_val.squeeze(-1)
        return squeeze


def _replace_outputs(
    batch_norm: fx.Node,
    output_accessors: list[fx.Node],
    replacement: fx.Node,
) -> None:
    graph = batch_norm.graph
    for accessor in output_accessors:
        accessor.replace_all_uses_with(replacement)
    for accessor in list(batch_norm.users):
        graph.erase_node(accessor)

    graph.erase_node(batch_norm)


class ReplaceBatchNormWithConv(ExportedProgramPassBase):
    """Replace inference batch normalization with a depthwise convolution."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False
        for bn_node in graph.find_nodes(op="call_function", target=_BATCH_NORM_TARGET):
            modified |= self._replace(exported_program, bn_node)

        if modified:
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)

    def _replace(
        self,
        exported_program: ExportedProgram,
        batch_norm: fx.Node,
    ) -> bool:
        output_accessors = _native_output_accessors(batch_norm)
        if output_accessors is None:
            return False

        inp = get_arg(batch_norm, "input", fx.Node)
        inp_val = inp.meta["val"]
        if not isinstance(inp_val, torch.Tensor) or not 2 <= inp_val.ndim <= 5:
            return False

        constants = _get_constants(exported_program, batch_norm)
        if constants is None:
            return False

        spatial_dims = max(inp_val.ndim - 2, 1)
        eps = get_arg(batch_norm, "eps", float)
        fused_weight, fused_bias = _make_fused_parameters(
            constants, eps, inp_val.dtype, spatial_dims
        )
        conv_weight, conv_bias = _add_fused_parameters(
            exported_program,
            batch_norm,
            constants,
            fused_weight,
            fused_bias,
        )
        replacement = _insert_convolution(
            batch_norm,
            inp,
            conv_weight,
            conv_bias,
            output_accessors[0].meta,
        )
        _replace_outputs(batch_norm, output_accessors, replacement)
        return True
