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
from executorch.backends.vulkan.patterns.weight_packing_utils import (
    pack_4bit_weight_tensor,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


embedding_4bit_target = exir_ops.edge.quantized_decomposed.embedding_4bit.dtype
embedding_target = exir_ops.edge.aten.embedding.default
torchao_dequantize_affine_target = exir_ops.edge.torchao.dequantize_affine.default


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

        # The weight placeholder stores values PACKED as uint8 [vocab,
        # embed_dim / 2], so embed_dim is twice the inner dim. The op
        # implementation requires that embed dim % 32 == 0 due to load/store
        # granularity for the weight tensor; enforce that check now.
        weight_val = (
            self.weight_node.meta.get("val", None)
            if isinstance(self.weight_node, torch.fx.Node)
            else None
        )
        if not isinstance(weight_val, torch.Tensor) or weight_val.ndim != 2:
            return
        embed_dim = int(weight_val.shape[-1]) * 2  # packed, 2 values per byte
        if embed_dim % 32 != 0:
            return

        self.match_found = True


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

    # The quantized_decomposed.embedding_4bit op (which is being replaced)
    # already stores weights as packed uint8 [vocab, embed_dim / 2] (low nibble = odd,
    # high nibble = even). However, in the Vulkan runtime 4-bit linear layers
    # expect the reverse nibble packing (low nibble = even, high nibble = odd).
    # In LLMs, where quantized embeddings are most frequently used, the embedding
    # layer will share weights with the final LM head linear layer. For simplicity,
    # always repack the weight tensor in the format expected by 4 bit linear layers;
    # the runtime shader supports both the original and repacked packing formats
    # for weights.
    if utils.register_param_mutation(ep, match.weight_node, "4 bit linear weight"):
        # Repack using linear convention (low nibble = even, high nibble = odd)
        vocab_size = weight_tensor.shape[0]
        high = (weight_tensor >> 4).to(torch.int8) - 8
        low = (weight_tensor & 0xF).to(torch.int8) - 8
        unpacked = torch.stack([high, low], dim=-1).reshape(vocab_size, -1)
        weight_tensor = pack_4bit_weight_tensor(unpacked)
        utils.align_width_and_update_state_dict(
            ep, match.weight_node, weight_tensor, align_to=1, force_update=True
        )

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
                True,  # is_linear_weight
            ),
        )

    embedding_q4gsw_node.meta["val"] = match.anchor_node.meta["val"]
    match.anchor_node.replace_all_uses_with(embedding_q4gsw_node)


class TorchAOQuantizedEmbeddingMatch(PatternMatch):
    """Matches a torchao 4-bit weight-only quantized embedding and rewrites it
    as a single et_vk.embedding_q4gsw.default node.

    The recognized graph shape is a split torchao.dequantize_affine ->
    aten.embedding, whose weight is unpacked int8 [vocab, embed_dim] with values
    in [-8, 7]. This requires symmetric 4-bit signed quantization (quant_min=-8,
    quant_max=7, zero_point=0) and per-row groupwise blocks (block_size=[1, G]),
    which the runtime shader assumes via a fixed subtract-8 offset.
    """

    def __init__(self, node: torch.fx.Node) -> None:  # noqa: C901
        self.anchor_node = node
        self.match_found = False
        self.all_nodes = [node]

        # aten.embedding.default args: (weight, indices, *)
        dequant_node = node.args[0]
        self.indices_node = node.args[1]

        if not isinstance(dequant_node, torch.fx.Node):
            return
        if dequant_node.target != torchao_dequantize_affine_target:
            return

        self.all_nodes.append(dequant_node)

        # torchao.dequantize_affine args:
        # (input, block_size, scale, zero_point, input_dtype, quant_min,
        #  quant_max, ...)
        block_size = dequant_node.args[1]
        input_dtype = dequant_node.args[4] if len(dequant_node.args) > 4 else None
        quant_min = dequant_node.args[5] if len(dequant_node.args) > 5 else None
        quant_max = dequant_node.args[6] if len(dequant_node.args) > 6 else None

        # The shader hardcodes the 4-bit signed offset (subtract 8), which
        # corresponds to quant_min=-8, quant_max=7, zero_point=0.
        if quant_min != -8 or quant_max != 7:
            return

        # Key off the dequant node's declared input_dtype, not the weight
        # placeholder's live meta: a sibling match sharing the same (tied) weight
        # may have repacked that placeholder in place (flipping it to packed
        # uint8 [vocab, embed_dim / 2]), which would spuriously reject us here.
        if input_dtype != torch.int8:
            return

        # block_size must be per-row groupwise: [1, group_size]
        if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
            return
        if block_size[0] != 1:
            return
        self.group_size = int(block_size[1])

        # Trace weight (args[0]) and scales (args[2]) to their placeholders. A
        # placeholder-backed zero_point's symmetric (zero_point == 0)
        # requirement is verified on the real tensor in the replacement
        # function, where the ExportedProgram is available; checking the fake
        # meta tensor here would trigger a data-dependent guard error.
        weight_node, arg_chain = utils.trace_args_until_placeholder(
            dequant_node.args[0]
        )
        if weight_node is None:
            return
        self.weight_node = weight_node
        self.all_nodes.extend(arg_chain)

        # Read embed_dim from the dequant node's float output meta, not the
        # weight placeholder's meta: a tied weight may have been repacked in
        # place by a sibling match (halving its inner dim), but this output meta
        # is stable. Runtime shader requires embed_dim % 32 == 0 and the groups
        # to tile the row exactly; reject otherwise rather than emit an op the
        # runtime would abort on.
        dequant_val = dequant_node.meta.get("val", None)
        if not isinstance(dequant_val, torch.Tensor) or dequant_val.ndim != 2:
            return
        embed_dim = int(dequant_val.shape[-1])
        if self.group_size <= 0 or embed_dim % self.group_size != 0:
            return
        if embed_dim % 32 != 0:
            return

        scales_node, arg_chain = utils.trace_args_until_placeholder(
            dequant_node.args[2]
        )
        if scales_node is None:
            return
        self.scales_node = scales_node
        self.all_nodes.extend(arg_chain)

        # zero_point (args[3]) must be provably zero, since the shader hardcodes
        # a subtract-8 offset that assumes symmetric quantization. Reject the
        # match otherwise so the op falls back cleanly rather than miscomputing.
        zero_point = dequant_node.args[3]
        self.zero_point_node = None
        if zero_point is None:
            # Symmetric quant; zero_point == 0 is implied.
            pass
        elif isinstance(zero_point, torch.fx.Node):
            zero_point_node, arg_chain = utils.trace_args_until_placeholder(zero_point)
            if zero_point_node is None:
                # Untraceable to a placeholder; cannot verify it is zero.
                return
            self.zero_point_node = zero_point_node
            self.all_nodes.extend(arg_chain)
        else:
            # Inline scalar / list / tuple; verify the literal value(s) are zero.
            values = (
                zero_point if isinstance(zero_point, (list, tuple)) else [zero_point]
            )
            if any(v != 0 for v in values):
                return

        self.match_found = True


@register_pattern_detector("torchao_quantized_embedding")
def find_torchao_quantized_embedding_patterns(
    node: torch.fx.Node,
) -> Optional[TorchAOQuantizedEmbeddingMatch]:
    if node.target != embedding_target:
        return None

    matched_pattern = TorchAOQuantizedEmbeddingMatch(node)
    if matched_pattern.match_found:
        return matched_pattern
    return None


@register_pattern_replacement("torchao_quantized_embedding")
def replace_torchao_quantized_embedding_patterns(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: TorchAOQuantizedEmbeddingMatch,
):
    # Always repack with the packing expected by 4 bit linear layers for
    # simplicity. See replace_quantized_embedding_patterns() for more details
    if utils.register_param_mutation(ep, match.weight_node, "4 bit linear weight"):
        weight_tensor = get_param_tensor(ep, match.weight_node)
        assert weight_tensor is not None

        # The shader applies a fixed signed-4-bit offset (subtract 8), which
        # assumes symmetric quantization (zero_point == 0). The None / inline
        # literal cases were already proven zero in the matcher; a placeholder
        # was committed to during matching, so its backing tensor must be
        # fetchable and verifiable here.
        if match.zero_point_node is not None:
            zero_point_tensor = get_param_tensor(ep, match.zero_point_node)
            if zero_point_tensor is None:
                raise RuntimeError(
                    "embedding_q4gsw: zero_point traced to placeholder "
                    f"{match.zero_point_node.name!r} but its backing tensor "
                    "could not be fetched to verify symmetric quantization "
                    "(zero_point == 0)."
                )
            assert torch.all(
                zero_point_tensor == 0
            ), "embedding_q4gsw requires symmetric quantization (zero_point == 0)"

        packed_weight = pack_4bit_weight_tensor(weight_tensor)

        utils.align_width_and_update_state_dict(
            ep, match.weight_node, packed_weight, align_to=1, force_update=True
        )

    group_size = match.group_size

    with graph_module.graph.inserting_before(match.anchor_node):
        embedding_q4gsw_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.embedding_q4gsw.default,
            args=(
                match.weight_node,
                match.scales_node,
                group_size,
                match.indices_node,
                True,  # is_linear_weight
            ),
        )

    embedding_q4gsw_node.meta["val"] = match.anchor_node.meta["val"]
    match.anchor_node.replace_all_uses_with(embedding_q4gsw_node)
