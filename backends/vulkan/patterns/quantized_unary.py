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


class QuantizedUnaryMatch(PatternMatch):
    def __init__(self, unary_node: torch.fx.Node) -> None:
        self.anchor_node = unary_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        # The unary op takes a single input which must be a dequantize node
        if len(unary_node.args) < 1:
            return

        input_node = unary_node.args[0]
        assert isinstance(input_node, torch.fx.Node)

        if not utils.is_dequant_node(input_node):
            return

        self.dequantize_input_node = input_node

        # Extract quantization parameters for the input
        self.quantize_input_node = self.dequantize_input_node.args[0]
        self.input_scales_node = self.dequantize_input_node.args[1]
        self.input_zeros_node = self.dequantize_input_node.args[2]

        self.all_nodes.append(self.dequantize_input_node)

        # The unary op output must have exactly one user: a quantize node
        self.output_node = self.anchor_node

        if len(self.output_node.users) != 1:
            return

        cur_node = list(self.output_node.users)[0]

        if not utils.is_quant_node(cur_node):
            return

        self.quantize_output_node = cur_node
        self.output_scales_node = self.quantize_output_node.args[1]
        self.output_zeros_node = self.quantize_output_node.args[2]

        self.all_nodes.append(self.quantize_output_node)

        self.match_found = True


# Unary operation anchor nodes that we support
unary_anchor_nodes = {
    exir_ops.edge.aten.relu.default,
}


@register_pattern_detector("quantized_unary")
def find_quantized_unary_patterns(
    node: torch.fx.Node,
) -> Optional[QuantizedUnaryMatch]:
    if node.target not in unary_anchor_nodes:
        return None

    matched_pattern = QuantizedUnaryMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Pattern Replacement
##


@register_pattern_replacement("quantized_unary")
def make_q8ta_unary_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedUnaryMatch,
):
    op_target = None
    if match.anchor_node.target == exir_ops.edge.aten.relu.default:
        op_target = exir_ops.edge.et_vk.q8ta_relu.default
    else:
        raise NotImplementedError(
            f"Unsupported unary operation: {match.anchor_node.target}"
        )

    with graph_module.graph.inserting_before(match.output_node):
        qunary_node = graph_module.graph.create_node(
            "call_function",
            op_target,
            args=(
                match.quantize_input_node,
                match.input_scales_node,
                match.input_zeros_node,
                match.output_scales_node,
                match.output_zeros_node,
            ),
        )

    qunary_node.meta["val"] = match.output_node.meta["val"]
    match.quantize_output_node.replace_all_uses_with(qunary_node)
