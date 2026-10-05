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


def find_cache_update(
    cache_node: torch.fx.Node,
    start_pos_node: torch.fx.Node,
    nodes_before_attention: set[torch.fx.Node],
) -> Optional[torch.fx.Node]:
    for user in cache_node.users:
        if (
            is_update_cache_node(user)
            and user.args[1] is cache_node
            and user.args[2] is start_pos_node
            and not user.users
            and user in nodes_before_attention
        ):
            return user

    return None


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

        nodes_before_attention = set()
        for node in custom_sdpa_node.graph.nodes:
            if node is self.anchor_node:
                break
            nodes_before_attention.add(node)

        self.update_key_cache_node = find_cache_update(
            self.key_cache_node, self.start_pos_node, nodes_before_attention
        )

        self.key_projection_node = None
        if self.update_key_cache_node is not None:
            self.key_projection_node = self.update_key_cache_node.args[0]

        self.update_value_cache_node = find_cache_update(
            self.value_cache_node, self.start_pos_node, nodes_before_attention
        )

        self.value_projection_node = None
        if self.update_value_cache_node is not None:
            self.value_projection_node = self.update_value_cache_node.args[0]

        key_cache_users = {self.anchor_node, self.update_key_cache_node}
        value_cache_users = {self.anchor_node, self.update_value_cache_node}
        self.match_found = (
            self.update_key_cache_node is not None
            and self.key_projection_node is not None
            and self.update_value_cache_node is not None
            and self.value_projection_node is not None
            and self.key_cache_node is not self.value_cache_node
            and self.update_key_cache_node is not self.update_value_cache_node
            and self.key_projection_node is not self.key_cache_node
            and self.key_projection_node is not self.value_cache_node
            and self.value_projection_node is not self.key_cache_node
            and self.value_projection_node is not self.value_cache_node
            and set(self.key_cache_node.users) == key_cache_users
            and set(self.value_cache_node.users) == value_cache_users
        )


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


@register_pattern_replacement("causal_sdpa")
def replace_custom_sdpa_with_causal_sdpa(
    ep: Optional[ExportedProgram],
    graph_module: torch.fx.GraphModule,
    match: CausalSDPAMatch,
):
    assert match.update_key_cache_node is not None
    assert match.key_projection_node is not None
    assert match.update_value_cache_node is not None
    assert match.value_projection_node is not None

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
                match.start_pos_node,
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
