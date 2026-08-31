# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Set

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx.node import Argument


# Set of ops that act as no-ops on values (i.e. clones / dim_order copies that
# preserve dtype and shape). The matcher transparently skips these between the
# dequantize, pixel_shuffle, and quantize nodes.
_NOOP_PASSTHROUGH_TARGETS: Set[object] = {
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.dim_order_ops._clone_dim_order.default,
}


def _is_noop_passthrough(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in _NOOP_PASSTHROUGH_TARGETS


def _skip_passthrough_user(
    node: torch.fx.Node, collected: List[torch.fx.Node]
) -> Optional[torch.fx.Node]:
    """Given `node`, advance to its next non-passthrough user, walking through
    any chain of clone/dim_order_copy ops in between (collecting them in
    `collected`). Returns None if `node` has not exactly one user, or if any
    intermediate passthrough has more than one user."""
    if len(node.users) != 1:
        return None
    cur = next(iter(node.users))
    while _is_noop_passthrough(cur):
        collected.append(cur)
        if len(cur.users) != 1:
            return None
        cur = next(iter(cur.users))
    return cur


class QuantizedPixelShuffleMatch(PatternMatch):
    """
    Matches an un-decomposed PixelShuffle wrapped between a quant/dequant pair:

      q8ta_dequantize_per_tensor (int8 -> fp32)
      [optional] clone / _clone_dim_order
      aten.pixel_shuffle.default (upscale_factor = r)
      [optional] clone / _clone_dim_order
      q8ta_quantize_per_tensor (fp32 -> int8)

    The anchor is the dequantize node since it is a unique entry point.

    This relies on the partitioner's `ops_to_not_decompose()` hook preserving
    `aten.pixel_shuffle.default` through edge lowering, so we do not need to
    re-detect the decomposed view -> permute -> view pattern.
    """

    def __init__(self, dequantize_node: torch.fx.Node) -> None:
        self.anchor_node: torch.fx.Node = dequantize_node
        self.match_found: bool = False
        self.all_nodes: List[torch.fx.Node] = [dequantize_node]

        # Validate the dequantize node is one of the quant decomposed ops.
        if not utils.is_dequant_node(dequantize_node):
            return

        # Walk forward to the pixel_shuffle node (skipping any clones).
        pixel_shuffle_node = _skip_passthrough_user(dequantize_node, self.all_nodes)
        if pixel_shuffle_node is None:
            return
        if pixel_shuffle_node.op != "call_function":
            return
        if pixel_shuffle_node.target != exir_ops.edge.aten.pixel_shuffle.default:
            return

        # Walk forward to the quantize node (skipping any clones).
        quantize_node = _skip_passthrough_user(pixel_shuffle_node, self.all_nodes)
        if quantize_node is None or not utils.is_quant_node(quantize_node):
            return

        # pixel_shuffle args are (input, upscale_factor).
        if len(pixel_shuffle_node.args) < 2:
            return
        upscale_factor = pixel_shuffle_node.args[1]
        if not isinstance(upscale_factor, int):
            return

        # Capture the nodes and quant params we need for the replacement.
        self.dequantize_input_node = dequantize_node
        self.pixel_shuffle_node: torch.fx.Node = pixel_shuffle_node
        self.quantize_output_node: torch.fx.Node = quantize_node

        self.input_int8_node: Argument = dequantize_node.args[0]
        self.input_scales_node: Argument = dequantize_node.args[1]
        self.input_zeros_node: Argument = dequantize_node.args[2]
        self.output_scales_node: Argument = quantize_node.args[1]
        self.output_zeros_node: Argument = quantize_node.args[2]
        self.upscale_factor: int = upscale_factor

        self.all_nodes.extend([pixel_shuffle_node, quantize_node])
        # The replacement target replaces uses of the quantize node.
        self.output_node: torch.fx.Node = quantize_node

        self.match_found = True


@register_pattern_detector("quantized_pixel_shuffle")
def find_quantized_pixel_shuffle_pattern(
    node: torch.fx.Node,
) -> Optional[QuantizedPixelShuffleMatch]:
    if node.op != "call_function":
        return None
    if not utils.is_dequant_node(node):
        return None
    matched = QuantizedPixelShuffleMatch(node)
    if matched.match_found:
        return matched
    return None


@register_pattern_replacement("quantized_pixel_shuffle")
def make_quantized_pixel_shuffle_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedPixelShuffleMatch,
) -> None:
    op_target = exir_ops.edge.et_vk.q8ta_pixel_shuffle.default

    # The fused op takes the *inverse* of the output scale to match the
    # runtime kernel's expectation.
    output_scale = match.output_scales_node
    inv_output_scale: object
    if isinstance(output_scale, (int, float)):
        inv_output_scale = 1.0 / float(output_scale)
    else:
        # Intentional bail-out at the replacement step (not a TODO). The
        # matcher deliberately does not pre-validate that the output scale is
        # scalar because every observed quantize_per_tensor in real models has
        # a baked-in float scale; if that assumption breaks, we want a loud
        # failure here at fusion time rather than a silent miscompile.
        # If the output scale is a graph node (rare for static per-tensor
        # quant, but possible), insert a reciprocal computation. For all the
        # cases observed in the model the scales are baked-in floats, so we
        # raise here to make the failure visible rather than producing a
        # silent miscompile.
        raise NotImplementedError(
            "quantized_pixel_shuffle pattern only supports scalar output scales"
        )

    with graph_module.graph.inserting_before(match.output_node):
        new_node = graph_module.graph.create_node(
            "call_function",
            op_target,
            args=(
                match.input_int8_node,
                match.input_scales_node,
                match.input_zeros_node,
                inv_output_scale,
                match.output_zeros_node,
                match.upscale_factor,
            ),
        )

    new_node.meta["val"] = match.output_node.meta["val"]
    match.quantize_output_node.replace_all_uses_with(new_node)
