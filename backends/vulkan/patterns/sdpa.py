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

        # llama.custom_sdpa has signature:
        # custom_sdpa(query, key_cache, value_cache, start_pos, attn_mask, dropout_p, is_causal, scale) -> output
        if len(custom_sdpa_node.args) < 4:
            return

        self.query_node = custom_sdpa_node.args[0]
        self.key_cache_node = custom_sdpa_node.args[1]
        self.value_cache_node = custom_sdpa_node.args[2]
        self.start_pos_node = custom_sdpa_node.args[3]
        self.attn_mask_node = custom_sdpa_node.args[4]
        self.dropout_p_node = custom_sdpa_node.args[5]
        self.is_causal_node = custom_sdpa_node.args[6]
        if len(custom_sdpa_node.args) > 7:
            self.scale_node = custom_sdpa_node.args[7]
        else:
            self.scale_node = None

        # try to find update key cache node
        self.update_key_cache_node = None
        for user in self.key_cache_node.users:
            if is_update_cache_node(user):
                self.update_key_cache_node = user
                break

        self.key_projection_node = None
        if self.update_key_cache_node is not None:
            self.key_projection_node = self.update_key_cache_node.args[0]

        # find update value cache node
        self.update_value_cache_node = None
        for user in self.value_cache_node.users:
            if is_update_cache_node(user):
                self.update_value_cache_node = user
                break

        self.value_projection_node = None
        if self.update_value_cache_node is not None:
            self.value_projection_node = self.update_value_cache_node.args[0]

        # We have additional optional arguments but we don't need to capture them
        # since the new op doesn't use them

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
