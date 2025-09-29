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


class QuantizedBinaryMatch(PatternMatch):
    def __init__(self, binary_node: torch.fx.Node) -> None:
        self.anchor_node = binary_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        # Extract alpha parameter if it exists (for add operations)
        self.alpha = 1.0
        if len(binary_node.args) > 2 and binary_node.args[2] is not None:
            # Alpha is typically a scalar value
            if isinstance(binary_node.args[2], (int, float)):
                self.alpha = binary_node.args[2]

        # Identify input nodes - both should be dequantize nodes for static quantization
        if len(binary_node.args) < 2:
            return

        input_a_node = binary_node.args[0]
        assert isinstance(input_a_node, torch.fx.Node)
        input_b_node = binary_node.args[1]
        assert isinstance(input_b_node, torch.fx.Node)

        # Both arguments must be dequant nodes for static quantization
        if not utils.is_dequant_node(input_a_node) or not utils.is_dequant_node(
            input_b_node
        ):
            return

        self.dequantize_input_a_node = input_a_node
        self.dequantize_input_b_node = input_b_node

        # Extract quantization parameters for input A
        self.quantize_input_a_node = self.dequantize_input_a_node.args[0]
        self.input_a_scales_node = self.dequantize_input_a_node.args[1]
        self.input_a_zeros_node = self.dequantize_input_a_node.args[2]

        # Extract quantization parameters for input B
        self.quantize_input_b_node = self.dequantize_input_b_node.args[0]
        self.input_b_scales_node = self.dequantize_input_b_node.args[1]
        self.input_b_zeros_node = self.dequantize_input_b_node.args[2]

        self.all_nodes.extend(
            [self.dequantize_input_a_node, self.dequantize_input_b_node]
        )

        # Identify output node
        self.output_node = self.anchor_node

        # The binary operation output must have only one user; it will be either a relu node
        # or a quantize node.
        if len(self.output_node.users) != 1:
            return

        cur_node = list(self.output_node.users)[0]
        self.relu_node = None
        if cur_node.target == exir_ops.edge.aten.relu.default:
            self.relu_node = cur_node
            self.all_nodes.append(self.relu_node)
            # If there's a relu, get its user (should be the quantize node)
            if len(cur_node.users) != 1:
                return
            cur_node = list(cur_node.users)[0]

        if not utils.is_quant_node(cur_node):
            return

        self.quantize_output_node = cur_node
        self.output_scales_node = self.quantize_output_node.args[1]
        self.output_zeros_node = self.quantize_output_node.args[2]

        self.all_nodes.append(self.quantize_output_node)

        self.match_found = True


# Define the binary operation anchor nodes that we support
binary_anchor_nodes = {
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.add_.Tensor,
}


@register_pattern_detector("quantized_binary")
def find_quantized_binary_patterns(
    node: torch.fx.Node,
) -> Optional[QuantizedBinaryMatch]:
    if node.target not in binary_anchor_nodes:
        return None

    matched_pattern = QuantizedBinaryMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Pattern Replacement
##


@register_pattern_replacement("quantized_binary")
def make_add_q8ta_q8ta_q8to_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedBinaryMatch,
):
    # Determine the operation type based on the anchor node
    op_target = None
    if match.anchor_node.target in {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.add_.Tensor,
    }:
        op_target = exir_ops.edge.et_vk.add_q8ta_q8ta_q8to.default
    else:
        # For future binary operations, add more mappings here
        raise NotImplementedError(
            f"Unsupported binary operation: {match.anchor_node.target}"
        )

    with graph_module.graph.inserting_before(match.output_node):
        qbinary_node = graph_module.graph.create_node(
            "call_function",
            op_target,
            args=(
                match.quantize_input_a_node,
                match.quantize_input_b_node,
                match.input_a_scales_node,
                match.input_a_zeros_node,
                match.input_b_scales_node,
                match.input_b_zeros_node,
                match.output_scales_node,
                match.output_zeros_node,
                match.alpha,  # Alpha parameter for scaling
            ),
        )

    qbinary_node.meta["val"] = match.output_node.meta["val"]
    match.quantize_output_node.replace_all_uses_with(qbinary_node)
