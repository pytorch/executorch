# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

from typing import Optional

import executorch.backends.vulkan.utils as utils

import torch
import torch.nn.functional as F

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from torch.export.graph_signature import InputKind


class QuantizedLinearMatch(PatternMatch):
    def __init__(self, mm_node: torch.fx.Node) -> None:
        self.anchor_node = mm_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        const_node, arg_chain = utils.trace_args_until_placeholder(
            self.anchor_node.args[1]
        )

        # mat2 is not a constant tensor - no match
        if const_node is None:
            return

        dequantize_weight_node = None
        # Search for a dequantize node in the arg chain of weight
        for node in arg_chain:
            if isinstance(node, torch.fx.Node) and utils.is_dequant_node(node):
                dequantize_weight_node = node
        # weight is not quantized - no match
        if dequantize_weight_node is None:
            return

        self.weight_node = const_node
        self.dequantize_weight_node = dequantize_weight_node
        self.all_nodes.extend(arg_chain)

        # By default, assume dequant node is from quantized_decomposed namespace
        scales_arg_idx = 1
        zeros_arg_idx = 2
        # torchao dequantize has a different function schema than quantized_decomposed
        if (
            self.dequantize_weight_node.target
            == exir_ops.edge.torchao.dequantize_affine.default
        ):
            scales_arg_idx = 2
            zeros_arg_idx = 3

        # Identify weight quantization parameter nodes
        self.weight_scales_node, arg_chain = utils.trace_args_until_placeholder(
            self.dequantize_weight_node.args[scales_arg_idx]
        )
        assert self.weight_scales_node is not None
        self.all_nodes.extend(arg_chain)

        self.weight_zeros_node, arg_chain = utils.trace_args_until_placeholder(
            self.dequantize_weight_node.args[zeros_arg_idx]
        )
        assert self.weight_zeros_node is not None
        self.all_nodes.extend(arg_chain)

        # Identify output node
        self.output_node = self.anchor_node

        # The implementation has a limitation that output channels must be a
        # multiple of 4. This is to ensure that data loads are aligned well with
        # texel boundaries. If this is not true, then don't match the pattern.
        out_channels = self.output_node.meta["val"].shape[-1]
        if out_channels % 4 != 0:
            return

        # Identify input node
        self.fp_input_node, self.quantize_input_node, dq_node = (
            utils.maybe_skip_q_dq_arg_chain(self.anchor_node.args[0])
        )
        assert self.fp_input_node is not None
        self.all_nodes.append(self.fp_input_node)

        # The implementation has a limitation that input channels must be a
        # multiple of 4. This is to ensure that data loads are aligned well with
        # texel boundaries. If this is not true, then don't match the pattern.
        in_channels = self.fp_input_node.meta["val"].shape[-1]
        if in_channels % 4 != 0:
            return

        # Identify bias node, if applicable
        self.bias_node = None
        if self.anchor_node.target == exir_ops.edge.aten.addmm.default:
            self.bias_node, arg_chain = utils.trace_args_until_placeholder(
                self.anchor_node.args[2]
            )
            assert self.bias_node is not None
            self.all_nodes.extend(arg_chain)

        # If input is not quantized, then we are done
        if self.quantize_input_node is None:
            self.match_found = True
            return

        scales_arg_idx = 1
        zeros_arg_idx = 2

        # torchao op has a slightly different function schema
        if (
            self.quantize_input_node.target
            == exir_ops.edge.torchao.quantize_affine.default
        ):
            scales_arg_idx = 2
            zeros_arg_idx = 3

        self.input_scales_node = self.quantize_input_node.args[scales_arg_idx]
        self.input_zeros_node = self.quantize_input_node.args[zeros_arg_idx]

        assert dq_node is not None
        self.all_nodes.extend(
            [
                self.quantize_input_node,
                dq_node,
            ]
        )

        self.match_found = True

    def is_weight_only_quantized(self) -> bool:
        return self.quantize_input_node is None

    def is_weight_pergroup_quantized(self) -> bool:
        weight_shape = self.weight_node.meta["val"].shape
        scales_shape = self.weight_scales_node.meta["val"].shape
        if len(scales_shape) != 2:
            return False

        # Check that:
        # height dim of scales is same as height dim of weight (N / output channels dim)
        # width dim of weight (K / in channels dim) is divisible by width dim of scales
        # (number of quantization groups)
        return scales_shape[-2] == weight_shape[-2] and (
            weight_shape[-1] % scales_shape[-1] == 0
        )

    def is_weight_perchannel_quantized(self) -> bool:
        weight_shape = self.weight_node.meta["val"].shape
        scales_shape = self.weight_scales_node.meta["val"].shape
        if len(scales_shape) != 1:
            return False

        # scales should have same size as weight's output channels dim
        return scales_shape[0] == weight_shape[-2]

    def is_input_static_per_tensor_quantized(self) -> bool:
        if self.quantize_input_node is None:
            return False

        # For static quantization per tensor quantization, the scales and zeros
        # are scalars.
        return isinstance(self.input_scales_node, float)

    def is_input_dynamic_perchannel_quantized(self) -> bool:
        if self.quantize_input_node is None:
            return False

        # For dynamic quantization, input scale node should be a getitem operator
        # retrieving the output of a choose_qparams op
        if self.input_scales_node.target != operator.getitem:
            return False

        # The getitem node should be retrieving from a choose_qparams op
        if not utils.is_choose_qparams_node(self.input_scales_node.args[0]):
            return False

        scales_shape = self.input_scales_node.meta["val"].shape
        input_shape = self.fp_input_node.meta["val"].shape

        return input_shape[-2] == scales_shape[-1]


linear_anchor_nodes = {
    exir_ops.edge.aten.linear.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
}


@register_pattern_detector("quantized_linear")
def find_quantized_linear_patterns(
    node: torch.fx.Node,
) -> Optional[QuantizedLinearMatch]:
    if node.target not in linear_anchor_nodes:
        return None

    matched_pattern = QuantizedLinearMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Constant tensor manipulation
##


def pack_4bit_weight_tensor(weight_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a 8-bit weight tensor containing values quantized to 4 bits, create a packed
    weight tensor by transposing the weight tensor, then packing 2 4-bit values in one
    8-bit value.

    An input weight tensor of shape (N, K) will produce a packed weight tensor of shape
    (K, N / 2).
    """

    # Assert we got a properly quantized tensor.
    min_val, max_val = weight_tensor.min().item(), weight_tensor.max().item()
    assert (
        max_val <= 7 and min_val >= -8
    ), f"pack_4bit_weight_tensor: [min_val,max_val] out of [-8, 7] range, got [{min_val}, {max_val}]"

    # Assuming we have a 2d tensor
    if weight_tensor.ndim != 2:
        weight_tensor = weight_tensor.squeeze()
    assert (
        weight_tensor.ndim == 2
    ), f"pack_4bit_weight_tensor: expecting input tensor to be 2d, got {weight_tensor.ndim}"

    # Need to pad innermost dim to be a multiple of 8, since the minimum load granularity
    # is int32 (4 bytes), which contains 8 4-bit values.
    if weight_tensor.shape[-1] % 8 != 0:
        num_pad = 8 - (weight_tensor.shape[-1] % 8)
        weight_tensor = F.pad(input=weight_tensor, pad=(0, num_pad))

    # Shape after padding
    _, in_channels = weight_tensor.shape
    assert in_channels % 8 == 0, "convert_to_qc4w: expecting ic to be divisible by 8"

    # Adjust weight_tensor tensor for zp
    weight_tensor = weight_tensor.to(dtype=torch.uint8) + 8
    # Pack each 4-bit value into a single 8-bit value
    return weight_tensor[::, 1::2] << 4 | weight_tensor[::, ::2]


def compute_per_group_sums(weight_tensor: torch.Tensor, group_size: int):
    """
    Compute the sum of weights per quantization group.

    Args:
        weight_tensor (torch.Tensor): Tensor of shape [out_channels, in_channels], dtype int8.
        group_size (int): Number of input channels per quantization group.

    Returns:
        torch.Tensor: Tensor of shape [num_groups, out_channels], where num_groups = in_channels // group_size.
    """
    out_channels, in_channels = weight_tensor.shape
    num_groups = in_channels // group_size
    # Reshape to [out_channels, num_groups, group_size]
    reshaped = weight_tensor.view(out_channels, num_groups, group_size)
    # Sum over group_size dimension to get [out_channels, num_groups]
    sums = reshaped.sum(dim=2)
    # Transpose to [num_groups, out_channels]
    sums = sums.transpose(0, 1).contiguous()
    # Pad out_channels dim (dim=1) to be a multiple of 8 if needed
    out_channels = sums.shape[1]
    if out_channels % 8 != 0:
        num_pad = 8 - (out_channels % 8)
        sums = F.pad(sums, (0, num_pad))

    return sums.to(torch.int32).contiguous()


##
## Pattern Replacement
##


def make_linear_q4gsw_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
    weight_tensor: torch.Tensor,
    weight_scales_tensor: torch.Tensor,
):
    num_groups = weight_scales_tensor.shape[-1]
    in_channels = weight_tensor.shape[-1]
    group_size = in_channels // num_groups

    weight_tensor = pack_4bit_weight_tensor(weight_tensor)
    # Use this function for convenience to update the state dict with the packed
    # weight tensor. Alignment will already have been done in the above function.
    weight_tensor = utils.align_width_and_update_state_dict(
        ep, match.weight_node, weight_tensor, align_to=1, force_update=True
    )

    # Also transpose the weight scales tensor to shape [num_groups, N]
    weight_scales_tensor = weight_scales_tensor.transpose(0, 1).contiguous()
    # Align to multiple of 8 to ensure that data loads from the weight scales
    # tensor do not go out of bounds. Each thread computes 8 output channels.
    utils.align_width_and_update_state_dict(
        ep,
        match.weight_scales_node,
        weight_scales_tensor,
        align_to=8,
        force_update=True,
    )

    with graph_module.graph.inserting_before(match.output_node):
        linear_q4gsw_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_q4gsw.default,
            args=(
                match.fp_input_node,
                match.weight_node,
                match.weight_scales_node,
                group_size,
            ),
        )

    linear_q4gsw_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(linear_q4gsw_node)


def make_linear_dq8ca_q4gsw_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
    weight_tensor: torch.Tensor,
    weight_scales_tensor: torch.Tensor,
):
    num_groups = weight_scales_tensor.shape[-1]
    in_channels = weight_tensor.shape[-1]
    group_size = in_channels // num_groups

    # Compute per quant group sums before packing the weight tensor
    sum_per_quant_group = compute_per_group_sums(weight_tensor, group_size)

    weight_tensor = pack_4bit_weight_tensor(weight_tensor)
    # Use this function for convenience to update the state dict with the packed
    # weight tensor. Alignment will already have been done in the above function.
    weight_tensor = utils.align_width_and_update_state_dict(
        ep, match.weight_node, weight_tensor, align_to=1, force_update=True
    )

    # Also transpose the weight scales tensor to shape [num_groups, N]
    weight_scales_tensor = weight_scales_tensor.transpose(0, 1).contiguous()
    utils.align_width_and_update_state_dict(
        ep,
        match.weight_scales_node,
        weight_scales_tensor,
        align_to=1,
        force_update=True,
    )

    first_graph_node = list(graph_module.graph.nodes)[0]
    with graph_module.graph.inserting_before(first_graph_node):
        weight_tensor_name = utils.get_tensor_name(ep, match.weight_node)
        # Pre-compute the weight sums which are needed to apply activation zero point
        # when using integer accumulation.
        sums_name = weight_tensor_name + "_sums"
        # Sanitize the name
        sums_name = sums_name.replace(".", "_")

        weight_sums_node = create_constant_placeholder(
            exp_program=ep,
            graph=graph_module.graph,
            kind=InputKind.CONSTANT_TENSOR,
            name=sums_name,
            data=sum_per_quant_group,
        )

    with graph_module.graph.inserting_before(match.output_node):
        qlinear_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_dq8ca_q4gsw.default,
            args=(
                match.fp_input_node,
                match.input_scales_node,
                match.input_zeros_node,
                match.weight_node,
                weight_sums_node,
                match.weight_scales_node,
                group_size,
            ),
        )

    qlinear_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(qlinear_node)


def make_linear_q8ta_q8csw_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
    weight_tensor: torch.Tensor,
):
    first_graph_node = list(graph_module.graph.nodes)[0]
    with graph_module.graph.inserting_before(first_graph_node):
        weight_tensor_name = utils.get_tensor_name(ep, match.weight_node)
        # Pre-compute the weight sums which are needed to apply activation zero point
        # when using integer accumulation.
        sum_per_output_channel = weight_tensor.sum(dim=1).to(torch.int32).contiguous()
        sums_name = weight_tensor_name + "_sums"
        # Sanitize the name
        sums_name = sums_name.replace(".", "_")

        weight_sums_node = create_constant_placeholder(
            exp_program=ep,
            graph=graph_module.graph,
            kind=InputKind.CONSTANT_TENSOR,
            name=sums_name,
            data=sum_per_output_channel,
        )

    with graph_module.graph.inserting_before(match.output_node):
        qlinear_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_q8ta_q8csw.default,
            args=(
                match.fp_input_node,
                match.input_scales_node,
                match.input_zeros_node,
                match.weight_node,
                weight_sums_node,
                match.weight_scales_node,
            ),
        )

    qlinear_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(qlinear_node)


@register_pattern_replacement("quantized_linear")
def replace_quantized_linear_patterns(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
):
    # Extract relevant tensors
    weight_tensor = get_param_tensor(ep, match.weight_node)
    assert weight_tensor is not None

    assert match.weight_scales_node is not None
    weight_scales_tensor = get_param_tensor(ep, match.weight_scales_node)
    assert weight_scales_tensor is not None

    assert match.weight_zeros_node is not None
    weight_zeros_tensor = get_param_tensor(ep, match.weight_zeros_node)
    assert weight_zeros_tensor is not None

    # Biases not supported at the moment
    if match.bias_node is not None:
        return

    # Route to appropriate custom op
    if (
        match.is_weight_only_quantized()
        and match.is_weight_pergroup_quantized()
        and utils.is_in_4bit_range(weight_tensor)
    ):
        make_linear_q4gsw_op(
            ep, graph_module, match, weight_tensor, weight_scales_tensor
        )
    elif (
        match.is_input_dynamic_perchannel_quantized()
        and match.is_weight_pergroup_quantized()
        and utils.is_in_4bit_range(weight_tensor)
    ):
        make_linear_dq8ca_q4gsw_op(
            ep, graph_module, match, weight_tensor, weight_scales_tensor
        )
    elif (
        match.is_input_static_per_tensor_quantized()
        and match.is_weight_perchannel_quantized()
    ):
        make_linear_q8ta_q8csw_custom_op(ep, graph_module, match, weight_tensor)
