# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


_MISSING = object()


def _get_argument(
    node: torch.fx.Node,
    index: int,
    name: str,
    default: Any = _MISSING,
) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def is_update_cache_node(node: Any) -> bool:
    return utils.node_has_target(node, "llama::update_cache")


def is_custom_sdpa_node(node: Any) -> bool:
    return utils.node_has_target(node, "llama::custom_sdpa")


def is_sdpa_with_kv_cache_node(node: Any) -> bool:
    return utils.node_has_target(node, "llama::sdpa_with_kv_cache")


class CausalSDPAMatch(PatternMatch):
    def __init__(self, custom_sdpa_node: torch.fx.Node) -> None:
        self.anchor_node = custom_sdpa_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        self.query_node = _get_argument(custom_sdpa_node, 0, "query")
        self.key_cache_node = _get_argument(custom_sdpa_node, 1, "key")
        self.value_cache_node = _get_argument(custom_sdpa_node, 2, "value")
        self.start_pos_node = _get_argument(custom_sdpa_node, 3, "start_pos")
        if any(
            value is _MISSING
            for value in (
                self.query_node,
                self.key_cache_node,
                self.value_cache_node,
                self.start_pos_node,
            )
        ):
            return

        if not isinstance(self.key_cache_node, torch.fx.Node) or not isinstance(
            self.value_cache_node, torch.fx.Node
        ):
            return

        self.attn_mask_node = _get_argument(custom_sdpa_node, 4, "attn_mask", None)
        self.dropout_p_node = _get_argument(custom_sdpa_node, 5, "drpout_p", 0.0)
        self.is_causal_node = _get_argument(custom_sdpa_node, 6, "is_causal", False)
        self.scale_node = _get_argument(custom_sdpa_node, 7, "scale", None)

        # try to find update key cache node
        self.update_key_cache_node = None
        for user in self.key_cache_node.users:
            if is_update_cache_node(user) and user.args:
                self.update_key_cache_node = user
                break

        self.key_projection_node = None
        if self.update_key_cache_node is not None:
            self.key_projection_node = self.update_key_cache_node.args[0]

        # find update value cache node
        self.update_value_cache_node = None
        for user in self.value_cache_node.users:
            if is_update_cache_node(user) and user.args:
                self.update_value_cache_node = user
                break

        self.value_projection_node = None
        if self.update_value_cache_node is not None:
            self.value_projection_node = self.update_value_cache_node.args[0]

        # We have additional optional arguments but we don't need to capture them
        # since the new op doesn't use them

        self.match_found = all(
            node is not None
            for node in (
                self.update_key_cache_node,
                self.key_projection_node,
                self.update_value_cache_node,
                self.value_projection_node,
            )
        ) and (
            self.attn_mask_node is None
            and self.dropout_p_node == 0
            and self.is_causal_node is True
            and self.scale_node is None
        )


class NonCausalSDPAMatch(PatternMatch):
    def __init__(self, custom_sdpa_node: torch.fx.Node) -> None:
        self.anchor_node = custom_sdpa_node
        self.all_nodes = [self.anchor_node]
        self.match_found = False

        self.query_node = _get_argument(custom_sdpa_node, 0, "query")
        self.key_node = _get_argument(custom_sdpa_node, 1, "key")
        self.value_node = _get_argument(custom_sdpa_node, 2, "value")
        self.attn_mask_node = _get_argument(custom_sdpa_node, 4, "attn_mask", None)
        self.dropout_p_node = _get_argument(custom_sdpa_node, 5, "drpout_p", 0.0)
        self.is_causal_node = _get_argument(custom_sdpa_node, 6, "is_causal", False)
        self.scale_node = _get_argument(custom_sdpa_node, 7, "scale", None)

        if not all(
            isinstance(node, torch.fx.Node)
            for node in (self.query_node, self.key_node, self.value_node)
        ):
            return
        if self.attn_mask_node is not None and not isinstance(
            self.attn_mask_node, torch.fx.Node
        ):
            return
        if self.dropout_p_node != 0 or self.is_causal_node is not False:
            return
        if self.scale_node is not None and not isinstance(
            self.scale_node, (int, float)
        ):
            return

        q_meta = self.query_node.meta.get("val")
        k_meta = self.key_node.meta.get("val")
        v_meta = self.value_node.meta.get("val")
        if any(meta is None for meta in (q_meta, k_meta, v_meta)):
            return

        q_shape = q_meta.shape
        k_shape = k_meta.shape
        v_shape = v_meta.shape
        if (
            q_meta.dtype not in utils.FP_T
            or k_meta.dtype != q_meta.dtype
            or v_meta.dtype != q_meta.dtype
            or len(q_shape) != 4
            or len(k_shape) != 4
            or len(v_shape) != 4
            or q_shape[0] != k_shape[0]
            or q_shape[0] != v_shape[0]
            or k_shape[1] != v_shape[1]
            or k_shape[2] != v_shape[2]
            or q_shape[3] != k_shape[3]
            or q_shape[3] != v_shape[3]
            or k_shape[2] <= 0
            or q_shape[2] < k_shape[2]
            or q_shape[2] % k_shape[2] != 0
        ):
            return

        if self.attn_mask_node is not None:
            mask_meta = self.attn_mask_node.meta.get("val")
            if mask_meta is None or mask_meta.dtype != q_meta.dtype:
                return
            mask_shape = mask_meta.shape
            output_shape = (q_shape[0], q_shape[2], q_shape[1], k_shape[1])
            if not 2 <= len(mask_shape) <= 4:
                return
            for mask_size, output_size in zip(
                reversed(mask_shape), reversed(output_shape)
            ):
                if mask_size != 1 and mask_size != output_size:
                    return

        self.match_found = True


@register_pattern_detector("causal_sdpa")
def find_causal_sdpa_patterns(
    node: torch.fx.Node,
) -> Optional[CausalSDPAMatch]:
    if not is_custom_sdpa_node(node):
        return None

    matched_pattern = CausalSDPAMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


@register_pattern_detector("noncausal_sdpa")
def find_noncausal_sdpa_patterns(
    node: torch.fx.Node,
) -> Optional[NonCausalSDPAMatch]:
    if not is_custom_sdpa_node(node):
        return None

    matched_pattern = NonCausalSDPAMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Pattern Replacement
##


def find_singleton_start_pos_node(graph_module: torch.fx.GraphModule):
    for node in graph_module.graph.nodes:
        if is_update_cache_node(node):
            return node.args[2]

        if is_sdpa_with_kv_cache_node(node):
            return node.args[5]

    raise Exception(
        "Could not find an instance of llama::update_cache or sdpa_with_kv_cache"
    )


@register_pattern_replacement("causal_sdpa")
def replace_custom_sdpa_with_causal_sdpa(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: CausalSDPAMatch,
):
    assert match.update_key_cache_node is not None
    assert match.key_projection_node is not None
    assert match.update_value_cache_node is not None
    assert match.value_projection_node is not None

    singleton_start_pos_node = find_singleton_start_pos_node(graph_module)

    with graph_module.graph.inserting_before(match.anchor_node):
        new_node = graph_module.graph.create_node(
            "call_function",
            torch.ops.llama.sdpa_with_kv_cache.default,
            args=(
                match.query_node,
                match.key_projection_node,
                match.value_projection_node,
                match.key_cache_node,
                match.value_cache_node,
                singleton_start_pos_node,
                1,
                match.attn_mask_node,
                match.dropout_p_node,
                match.is_causal_node,
                match.scale_node,
            ),
        )

    new_node.meta["val"] = match.anchor_node.meta["val"]
    match.anchor_node.replace_all_uses_with(new_node)

    # Manually erase update_cache nodes since DCE will not remove them since they
    # modify inputs (specifically, the cache args are modified)
    graph_module.graph.erase_node(match.update_key_cache_node)
    graph_module.graph.erase_node(match.update_value_cache_node)


@register_pattern_replacement("noncausal_sdpa")
def replace_custom_sdpa_with_noncausal_sdpa(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: NonCausalSDPAMatch,
):
    del ep

    with graph_module.graph.inserting_before(match.anchor_node):
        query = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.permute_copy.default,
            args=(match.query_node, [0, 2, 1, 3]),
        )
        key = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.permute_copy.default,
            args=(match.key_node, [0, 2, 1, 3]),
        )
        value = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.permute_copy.default,
            args=(match.value_node, [0, 2, 1, 3]),
        )
        sdpa = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.sdpa.default,
            args=(query, key, value, match.attn_mask_node, match.scale_node),
        )
        output = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten.permute_copy.default,
            args=(sdpa, [0, 2, 1, 3]),
        )

    query.meta["val"] = match.query_node.meta["val"].permute(0, 2, 1, 3).contiguous()
    key.meta["val"] = match.key_node.meta["val"].permute(0, 2, 1, 3).contiguous()
    value.meta["val"] = match.value_node.meta["val"].permute(0, 2, 1, 3).contiguous()
    sdpa.meta["val"] = query.meta["val"].contiguous()
    output.meta["val"] = sdpa.meta["val"].permute(0, 2, 1, 3).contiguous()
    match.anchor_node.replace_all_uses_with(output)
