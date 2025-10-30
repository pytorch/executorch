# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.backends.vulkan.utils as utils

import torch

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


class QuantizedConvolutionMatch(PatternMatch):
    def __init__(self, conv_node: torch.fx.Node) -> None:
        self.anchor_node = conv_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        # Extract convolution parameters
        self.stride = conv_node.args[3] if len(conv_node.args) > 3 else [1, 1]
        self.padding = conv_node.args[4] if len(conv_node.args) > 4 else [0, 0]
        self.dilation = conv_node.args[5] if len(conv_node.args) > 5 else [1, 1]
        self.groups = conv_node.args[8] if len(conv_node.args) > 8 else 1

        const_node, arg_chain = utils.trace_args_until_placeholder(
            self.anchor_node.args[1]
        )

        # weight is not a constant tensor - no match
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

        # Identify weight quantization parameter nodes
        self.weight_scales_node, arg_chain = utils.trace_args_until_placeholder(
            self.dequantize_weight_node.args[1]
        )
        assert self.weight_scales_node is not None
        self.all_nodes.extend(arg_chain)

        self.weight_zeros_node, arg_chain = utils.trace_args_until_placeholder(
            self.dequantize_weight_node.args[2]
        )
        assert self.weight_zeros_node is not None
        self.all_nodes.extend(arg_chain)

        # Identify output node
        self.output_node = self.anchor_node

        out_channels = self.output_node.meta["val"].shape[-3]
        # The implementation requires that for non-depthwise grouped convolutions, a
        # group does not cross the texel boundary. The output channels per group must be
        # a multiple of 4. If this is not true, then don't match the pattern.
        if (self.groups > 1 and self.groups < out_channels) and (
            out_channels / self.groups
        ) % 4 != 0:
            return

        # Identify bias node, if applicable
        self.bias_node = None
        if len(self.anchor_node.args) > 2 and self.anchor_node.args[2] is not None:
            self.bias_node, arg_chain = utils.trace_args_until_placeholder(
                self.anchor_node.args[2]
            )
            if self.bias_node is not None:
                self.all_nodes.extend(arg_chain)

        # Identify input node
        primary_input_node = self.anchor_node.args[0]
        assert isinstance(primary_input_node, torch.fx.Node)
        # Argument must be a dequant node for static quantization
        if not utils.is_dequant_node(primary_input_node):
            return

        self.dequantize_input_node = primary_input_node
        self.quantize_input_node = self.dequantize_input_node.args[0]

        self.input_scales_node = self.dequantize_input_node.args[1]
        self.input_zeros_node = self.dequantize_input_node.args[2]

        self.all_nodes.extend([self.dequantize_input_node])

        # The convolution output must have only one user; it will be either a relu node
        # or a dequantize node.
        if len(self.output_node.users) != 1:
            return

        cur_node = list(self.output_node.users)[0]
        self.relu_node = None
        if cur_node.target == exir_ops.edge.aten.relu.default:
            self.relu_node = cur_node
            cur_node = list(cur_node.users)[0]

        if not utils.is_quant_node(cur_node):
            return

        self.quantize_output_node = cur_node
        self.output_scales_node = self.quantize_output_node.args[1]
        self.output_zeros_node = self.quantize_output_node.args[2]

        self.match_found = True


convolution_anchor_nodes = {
    exir_ops.edge.aten.conv2d.default,
    exir_ops.edge.aten.convolution.default,
}


@register_pattern_detector("quantized_convolution")
def find_quantized_convolution_patterns(
    node: torch.fx.Node,
) -> Optional[QuantizedConvolutionMatch]:
    if node.target not in convolution_anchor_nodes:
        return None

    matched_pattern = QuantizedConvolutionMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Pattern Replacement
##


@register_pattern_replacement("quantized_convolution")
def make_conv2d_q8ta_q8csw_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedConvolutionMatch,
):
    weight_tensor = get_param_tensor(ep, match.weight_node)
    assert weight_tensor is not None

    assert match.weight_scales_node is not None
    weight_scales_tensor = get_param_tensor(ep, match.weight_scales_node)
    assert weight_scales_tensor is not None

    assert match.weight_zeros_node is not None
    weight_zeros_tensor = get_param_tensor(ep, match.weight_zeros_node)
    assert weight_zeros_tensor is not None

    bias_tensor = None
    if match.bias_node is not None:
        bias_tensor = get_param_tensor(ep, match.bias_node)
        assert bias_tensor is not None

    OC, IC_per_group, H, W = weight_tensor.shape

    is_depthwise_conv = IC_per_group == 1 and match.groups == OC

    if is_depthwise_conv:
        assert OC % 4 == 0, "depthwise conv requires that OC is divisible by 4"
        # Depthwise convs use a specialized layout; the weight tensor is reshaped to
        # (H, W, OC)
        weight_tensor = (
            weight_tensor.permute(2, 3, 1, 0).contiguous().view(H, W, OC).contiguous()
        )
    else:
        # Reshape weight tensor from (OC, IC_per_group, H, W) to (OC, H * W * IC_per_group)
        # (i.e. matrix format). This prepares the weights for Im2Col-based convolution.
        weight_tensor = (
            weight_tensor.permute(0, 2, 3, 1)
            .contiguous()
            .view(OC, H * W * IC_per_group)
            .contiguous()
        )

    # Need to make sure that OC dim is a multiple of 4 so that data load/stores are well
    # aligned with texel boundaries. Add padding to align to the next multiple of 4 if
    # needed.
    utils.align_width_and_update_state_dict(
        ep, match.weight_node, weight_tensor, force_update=True
    )
    utils.align_width_and_update_state_dict(
        ep, match.weight_scales_node, weight_scales_tensor
    )

    if bias_tensor is not None:
        utils.align_width_and_update_state_dict(ep, match.bias_node, bias_tensor)

    first_graph_node = list(graph_module.graph.nodes)[0]
    with graph_module.graph.inserting_before(first_graph_node):
        qweight_tensor_name = utils.get_tensor_name(ep, match.weight_node)
        # Pre-compute the weight sums which are needed to apply activation zero point
        # when using integer accumulation. For the reshaped 2D weight matrix (IC_per_group * H * W, OC),
        # sum over dimension 0 to get sums per output channel
        sum_per_output_channel = weight_tensor.sum(dim=1).to(torch.int32).contiguous()
        sums_name = qweight_tensor_name + "_sums"
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
        op_target = exir_ops.edge.et_vk.conv2d_q8ta_q8csw_q8to.default
        if is_depthwise_conv:
            op_target = exir_ops.edge.et_vk.conv2d_q8ta_q8csw_q8to_dw.default

        qconv_node = graph_module.graph.create_node(
            "call_function",
            op_target,
            args=(
                match.quantize_input_node,
                match.input_scales_node,
                match.input_zeros_node,
                match.weight_node,
                weight_sums_node,
                match.weight_scales_node,
                match.output_scales_node,
                match.output_zeros_node,
                match.bias_node,  # Add bias after weight_scales
                [H, W],  # Pass kernel size information before stride
                match.stride,
                match.padding,
                match.dilation,
                match.groups,
            ),
        )

    qconv_node.meta["val"] = match.output_node.meta["val"]
    match.quantize_output_node.replace_all_uses_with(qconv_node)
