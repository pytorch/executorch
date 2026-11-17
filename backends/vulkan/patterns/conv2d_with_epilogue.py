# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

residual_ops = {
    exir_ops.edge.aten.add.Tensor: "add",
    exir_ops.edge.aten.sub.Tensor: "sub",
    exir_ops.edge.aten.mul.Tensor: "mul",
}


def tensor_val(node: torch.fx.node.Argument) -> Optional[torch.Tensor]:
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


def fits_channels_packed_texture(val: torch.Tensor) -> bool:
    return bool(
        utils.filter_invalid_reprs(
            val, utils.CHANNELS_PACKED_TEXTURE, utils.DEFAULT_TEXTURE_LIMITS
        ).valid_texture_layouts
    )


def is_fusable_conv(node: torch.fx.node.Argument) -> bool:
    """
    A 2D, batch 1 floating point convolution with a single user, whose input and
    output fit channels-packed textures. The partitioner delegates the nodes of a
    match without checking them, so this repeats the convolution's own checks.
    """
    out = tensor_val(node)
    if (
        out is None
        or out.dtype not in (torch.float16, torch.float32)
        or node.target != exir_ops.edge.aten.convolution.default
    ):
        return False
    assert isinstance(node, torch.fx.Node)
    x = tensor_val(node.args[0])
    return (
        len(node.args) == 9
        and len(node.users) == 1
        and x is not None
        and x.dim() == 4
        and x.shape[0] == 1
        and fits_channels_packed_texture(x)
        and fits_channels_packed_texture(out)
    )


def activation_of(node: torch.fx.Node) -> Optional[tuple[str, float, float]]:
    """Returns the activation name and parameters of a fusable activation."""
    # The fused op takes the parameters as constants.
    if any(isinstance(arg, torch.fx.Node) for arg in node.args[1:]):
        return None
    if node.target == exir_ops.edge.aten.relu.default:
        return ("relu", 0.0, 0.0)
    if node.target == exir_ops.edge.aten.hardtanh.default:
        lo = node.args[1] if len(node.args) > 1 else -1.0
        hi = node.args[2] if len(node.args) > 2 else 1.0
        return ("clamp", float(lo), float(hi))
    if node.target == exir_ops.edge.aten.clamp.default:
        lo, hi = [*node.args[1:], None, None][:2]
        lo = float("-inf") if lo is None else float(lo)
        hi = float("inf") if hi is None else float(hi)
        return ("clamp", lo, hi)
    if node.target == exir_ops.edge.aten.leaky_relu.default:
        slope = node.args[1] if len(node.args) > 1 else 0.01
        return ("leaky_relu", float(slope), 0.0)
    return None


class Conv2dEpilogueMatch(PatternMatch):
    """
    Matches activation(residual_op(conv(...), residual [* scale])) and
    activation(conv(...)), where the activation and the per-channel scale of the
    residual are optional.
    """

    def __init__(self, node: torch.fx.Node) -> None:  # noqa: C901
        self.match_found = False
        self.residual_node: Optional[torch.fx.Node] = None
        self.scaled_residual_node: Optional[torch.fx.Node] = None
        self.scale_node: Optional[torch.fx.Node] = None
        self.residual_op = residual_ops.get(node.target)
        self.alpha = node.kwargs.get("alpha", 1)
        if self.residual_op is None:
            # activation(conv(...)), where node is the conv's only user.
            if activation_of(node) is None or not is_fusable_conv(node.args[0]):
                return
            self.residual_op = "add"
            self.conv_node = node.args[0]
            self.all_nodes = [self.conv_node]
            self.match_activation(self.conv_node)
            return

        # residual_op(conv, residual) must equal the matched node.
        operands = [(node.args[0], node.args[1])]
        if self.residual_op == "mul" or (self.residual_op == "add" and self.alpha == 1):
            operands.append((node.args[1], node.args[0]))
        for conv, residual in operands:
            out, res = tensor_val(conv), tensor_val(residual)
            if (
                is_fusable_conv(conv)
                and residual is not conv
                and res is not None
                and out is not None
                and res.shape == out.shape
                and res.dtype == out.dtype
            ):
                break
        else:
            return
        assert isinstance(conv, torch.fx.Node) and isinstance(residual, torch.fx.Node)
        self.conv_node = conv
        self.residual_node = residual
        self.scaled_residual_node = residual
        self.all_nodes = [conv, node]

        channels = out.shape[1]
        if (
            residual.target == exir_ops.edge.aten.mul.Tensor
            and len(residual.users) == 1
        ):
            for x, scale in (residual.args, residual.args[::-1]):
                x_val, scale_val = tensor_val(x), tensor_val(scale)
                if (
                    x_val is not None
                    and scale_val is not None
                    and x_val.shape == out.shape
                    and x_val.dtype == out.dtype
                    and scale_val.dtype == out.dtype
                    and tuple(scale_val.shape)
                    in ((1, channels, 1, 1), (channels, 1, 1))
                    and isinstance(scale, torch.fx.Node)
                    and scale.op == "placeholder"
                ):
                    assert isinstance(x, torch.fx.Node)
                    self.residual_node = x
                    self.scale_node = scale
                    self.all_nodes.append(residual)
                    break

        self.match_activation(node)

    def match_activation(self, node: torch.fx.Node) -> None:
        """Completes the match, fusing the only user of `node` if an activation."""
        self.anchor_node = node
        self.activation = ("none", 0.0, 0.0)
        if len(node.users) == 1:
            user = next(iter(node.users))
            activation = activation_of(user)
            if activation is not None:
                self.anchor_node = user
                self.activation = activation
                self.all_nodes.append(user)

        self.input_nodes = [
            n
            for n in (self.conv_node.args[0], self.residual_node)
            if isinstance(n, torch.fx.Node)
        ]
        self.output_nodes = [self.anchor_node]
        self.match_found = True


@register_pattern_detector("conv2d_with_epilogue")
def find_conv2d_with_epilogue_pattern(
    node: torch.fx.Node,
) -> Optional[Conv2dEpilogueMatch]:
    if node.target not in residual_ops and activation_of(node) is None:
        return None
    match = Conv2dEpilogueMatch(node)
    return match if match.match_found else None


@register_pattern_replacement("conv2d_with_epilogue")
def replace_conv2d_with_epilogue_pattern(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: Conv2dEpilogueMatch,
) -> Optional[bool]:
    residual, scale = match.residual_node, match.scale_node
    # The runtime binds the residual as a tensor and prepacks the scale, so a
    # constant x or a computed scale is fused as part of the residual x * scale.
    if scale is not None and (
        not utils.is_param_node(ep, scale) or utils.is_param_node(ep, residual)
    ):
        residual, scale = match.scaled_residual_node, None
    if residual is not None and utils.is_param_node(ep, residual):
        return False

    with graph_module.graph.inserting_before(match.anchor_node):
        fused = graph_module.graph.call_function(
            exir_ops.edge.et_vk.conv2d.with_epilogue,
            (
                *match.conv_node.args,
                residual,
                scale,
                match.residual_op,
                match.alpha,
                *match.activation,
            ),
        )
    fused.meta = match.anchor_node.meta.copy()
    match.anchor_node.replace_all_uses_with(fused)
