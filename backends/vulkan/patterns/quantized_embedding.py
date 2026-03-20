# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.backends.vulkan.utils as utils
import torch
from executorch.backends.transforms.utils import get_param_tensor
from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


class QuantizedEmbeddingMatch(PatternMatch):
    def __init__(self, node: torch.fx.Node) -> None:
        self.anchor_node = node
        self.match_found = False
        self.all_nodes = [node]

        # quantized_decomposed.embedding_4bit.dtype args:
        # (weight, weight_scales, weight_zero_points, quant_min, quant_max,
        #  indices, *, dtype)
        self.weight_node = node.args[0]
        self.scales_node = node.args[1]
        self.indices_node = node.args[5]

        # Validate quantization parameters match our shader's assumptions.
        # The shader hardcodes the 4-bit signed offset (subtract 8), which
        # corresponds to quant_min=-8, quant_max=7, zero_points=0.
        quant_min = node.args[3]
        quant_max = node.args[4]
        if quant_min != -8 or quant_max != 7:
            self.match_found = False
            return

        # weight_zero_points (args[2]) should be None or all-zeros
        weight_zp_node = node.args[2]
        if weight_zp_node is not None:
            # If it's a constant tensor, verify it's all zeros
            if (
                isinstance(weight_zp_node, torch.fx.Node)
                and "val" in weight_zp_node.meta
            ):
                zp_val = weight_zp_node.meta["val"]
                if isinstance(zp_val, torch.Tensor) and not torch.all(zp_val == 0):
                    self.match_found = False
                    return

        # Trace weight to its placeholder
        const_node, arg_chain = utils.trace_args_until_placeholder(self.weight_node)
        if const_node is not None:
            self.weight_node = const_node
            self.all_nodes.extend(arg_chain)

        # Trace scales to their placeholder
        scales_node, arg_chain = utils.trace_args_until_placeholder(self.scales_node)
        if scales_node is not None:
            self.scales_node = scales_node
            self.all_nodes.extend(arg_chain)

        self.match_found = True


embedding_4bit_target = exir_ops.edge.quantized_decomposed.embedding_4bit.dtype


def _detect_tied_linear_weight(
    ep: ExportedProgram,
    weight_node: torch.fx.Node,
    weight_tensor: torch.Tensor,
) -> bool:
    """Check if this embedding weight is tied to a linear weight.

    The embedding weight is packed uint8 [vocab_size, embed_dim/2]. The linear
    output weight may be stored as unpacked int8 [vocab_size, embed_dim]. If we
    find a placeholder whose int8 values match our unpacked embedding values,
    the weights are tied and we should use the linear packing to enable dedup.
    """
    vocab_size = weight_tensor.shape[0]
    embed_dim = weight_tensor.shape[1] * 2

    # Unpack embedding weight using embedding convention (high nibble first)
    emb_high = (weight_tensor >> 4).to(torch.int8) - 8
    emb_low = (weight_tensor & 0xF).to(torch.int8) - 8
    emb_unpacked = torch.stack([emb_high, emb_low], dim=-1).reshape(
        vocab_size, embed_dim
    )

    for node in ep.graph_module.graph.nodes:
        if node.op != "placeholder" or node == weight_node:
            continue

        try:
            candidate = get_param_tensor(ep, node)
        except RuntimeError:
            continue
        if candidate is None:
            continue
        if candidate.shape != (vocab_size, embed_dim) or candidate.dtype != torch.int8:
            continue

        if torch.equal(emb_unpacked, candidate):
            return True

    return False


@register_pattern_detector("quantized_embedding")
def find_quantized_embedding_patterns(
    node: torch.fx.Node,
) -> Optional[QuantizedEmbeddingMatch]:
    if node.target != embedding_4bit_target:
        return None

    matched_pattern = QuantizedEmbeddingMatch(node)
    if matched_pattern.match_found:
        return matched_pattern
    return None


@register_pattern_replacement("quantized_embedding")
def replace_quantized_embedding_patterns(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedEmbeddingMatch,
):
    weight_tensor = get_param_tensor(ep, match.weight_node)
    assert weight_tensor is not None

    scales_tensor = get_param_tensor(ep, match.scales_node)
    assert scales_tensor is not None

    is_linear_weight = _detect_tied_linear_weight(ep, match.weight_node, weight_tensor)

    if is_linear_weight:
        # Repack using linear convention (low nibble = even, high nibble = odd)
        vocab_size = weight_tensor.shape[0]
        high = (weight_tensor >> 4).to(torch.int8) - 8
        low = (weight_tensor & 0xF).to(torch.int8) - 8
        unpacked = torch.stack([high, low], dim=-1).reshape(vocab_size, -1)
        repacked = unpacked.to(torch.uint8) + 8
        weight_tensor = repacked[:, 1::2] << 4 | repacked[:, ::2]
        # Update the state dict with repacked tensor
        for key, value in ep.state_dict.items():
            if value.data_ptr() == get_param_tensor(ep, match.weight_node).data_ptr():
                ep.state_dict[key] = weight_tensor
                break

    # Compute group_size from weight and scales shapes
    embed_dim = weight_tensor.shape[1] * 2  # packed, 2 values per byte
    groups_per_row = scales_tensor.shape[1] if scales_tensor.ndim > 1 else 1
    group_size = embed_dim // groups_per_row

    with graph_module.graph.inserting_before(match.anchor_node):
        embedding_q4gsw_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.embedding_q4gsw.default,
            args=(
                match.weight_node,
                match.scales_node,
                group_size,
                match.indices_node,
                is_linear_weight,
            ),
        )

    embedding_q4gsw_node.meta["val"] = match.anchor_node.meta["val"]
    match.anchor_node.replace_all_uses_with(embedding_q4gsw_node)
