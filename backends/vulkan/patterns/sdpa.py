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


# Names and positions of (query, key_cache, attn_mask, is_causal) in each op's
# schema.
_LLM_SDPA_ARGS = {
    "llama::custom_sdpa": (
        ("query", 0),
        ("key", 1),
        ("attn_mask", 4),
        ("is_causal", 6),
    ),
    "llama::sdpa_with_kv_cache": (
        ("query", 0),
        ("key_cache", 3),
        ("attn_mask", 7),
        ("is_causal", 9),
    ),
}

_AUTO_FUNCTIONALIZED = (
    torch.ops.higher_order.auto_functionalized,
    torch.ops.higher_order.auto_functionalized_v2,
)


def _llm_sdpa_target(node: torch.fx.Node) -> Optional[str]:
    """The LLM SDPA op called by node, directly or through auto_functionalized."""
    if node.target in _AUTO_FUNCTIONALIZED:
        op = node.args[0]
        name = op.name() if hasattr(op, "name") else None
        return name if name in _LLM_SDPA_ARGS else None
    return next(
        (target for target in _LLM_SDPA_ARGS if utils.node_has_target(node, target)),
        None,
    )


def _get_llm_sdpa_arg(node: torch.fx.Node, name: str, idx: int, default: Any) -> Any:
    # auto_functionalized passes every schema argument as a kwarg, except that
    # v2 replaces a mutable argument with metadata about its base.
    if node.target not in _AUTO_FUNCTIONALIZED and idx < len(node.args):
        return node.args[idx]
    return node.kwargs.get(name, default)


def _get_v2_mutable_arg_shape(node: torch.fx.Node, name: str) -> Optional[list]:
    """
    The shape of a mutable argument of an auto_functionalized_v2 node, which is
    either one of _all_bases or a view of one. Unknown sizes are None.
    """
    prefix = f"_{name}"
    base_index = node.kwargs.get(f"{prefix}_base_index", None)
    bases = node.kwargs.get("_all_bases", None)
    if not isinstance(base_index, int) or not bases or base_index >= len(bases):
        return None
    base = bases[base_index]
    base_val = base.meta.get("val", None) if isinstance(base, torch.fx.Node) else None
    if not isinstance(base_val, torch.Tensor):
        return None
    if f"{prefix}_size" in node.kwargs:
        # A view regenerated with as_strided
        return list(node.kwargs[f"{prefix}_size"])
    shape = list(base_val.shape)
    if f"{prefix}_slice_dim" in node.kwargs:
        dim = node.kwargs[f"{prefix}_slice_dim"]
        start = node.kwargs.get(f"{prefix}_slice_start", None)
        end = node.kwargs.get(f"{prefix}_slice_end", None)
        if all(isinstance(v, (int, type(None))) for v in (shape[dim], start, end)):
            shape[dim] = len(range(*slice(start, end).indices(shape[dim])))
        else:
            shape[dim] = None
    # Otherwise the argument is the base itself, or an alias of it
    return shape


def _get_llm_sdpa_cache_shape(
    node: torch.fx.Node, name: str, idx: int
) -> Optional[list]:
    cache = _get_llm_sdpa_arg(node, name, idx, None)
    if cache is None and node.target == torch.ops.higher_order.auto_functionalized_v2:
        return _get_v2_mutable_arg_shape(node, name)
    cache_val = (
        cache.meta.get("val", None) if isinstance(cache, torch.fx.Node) else None
    )
    return list(cache_val.shape) if isinstance(cache_val, torch.Tensor) else None


def _is_update_cache_node(node: torch.fx.Node) -> bool:
    if node.target in _AUTO_FUNCTIONALIZED:
        op = node.args[0]
        return hasattr(op, "name") and op.name() == "llama::update_cache"
    return utils.node_has_target(node, "llama::update_cache")


def _check_update_cache_node(node: torch.fx.Node) -> bool:
    # custom_sdpa reads either the mutated cache directly, or the cache returned
    # by the auto_functionalized update_cache.
    readers = set(node.users)
    for user in node.users:
        readers.update(user.users)
    if utils.node_has_target(node, "llama::update_cache") and isinstance(
        node.args[1], torch.fx.Node
    ):
        readers.update(node.args[1].users)
    return all(
        check_llm_sdpa_node(reader)
        for reader in readers
        if utils.node_has_target(reader, "llama::custom_sdpa")
    )


def _check_llm_sdpa_mask(
    attn_mask: torch.fx.Node, q: Any, cache_shape: Optional[list]
) -> bool:
    mask_val = attn_mask.meta.get("val", None)
    q_val = q.meta.get("val", None) if isinstance(q, torch.fx.Node) else None
    if not all(isinstance(v, torch.Tensor) for v in (mask_val, q_val)):
        return False
    if cache_shape is None or len(cache_shape) < 2:
        return False
    if mask_val.dim() != 2 or mask_val.dtype != q_val.dtype:
        return False
    # The mask must be [S, C]. Only reject sizes that are known to mismatch.
    return all(
        not (isinstance(size, int) and isinstance(expected, int)) or size == expected
        for size, expected in (
            (mask_val.shape[0], q_val.shape[1]),
            (mask_val.shape[1], cache_shape[1]),
        )
    )


def check_llm_sdpa_node(node: torch.fx.Node) -> bool:
    """
    The Vulkan LLM SDPA is either causal without an attn_mask, or takes a 2D
    [S, C] additive attn_mask (with is_causal unset) of the same dtype as the
    query. Other configurations stay on CPU instead of failing at load time.
    The node may call the op directly or through auto_functionalized. An
    update_cache stays on CPU with the custom_sdpa that reads its cache, so
    that both see the same cache.
    """
    if _is_update_cache_node(node):
        return _check_update_cache_node(node)

    target = _llm_sdpa_target(node)
    if target is None:
        return True
    q_arg, cache_arg, mask_arg, is_causal_arg = _LLM_SDPA_ARGS[target]

    attn_mask = _get_llm_sdpa_arg(node, *mask_arg, None)
    is_causal = _get_llm_sdpa_arg(node, *is_causal_arg, False)
    if attn_mask is None:
        return bool(is_causal)
    if is_causal or not isinstance(attn_mask, torch.fx.Node):
        return False
    return _check_llm_sdpa_mask(
        attn_mask,
        _get_llm_sdpa_arg(node, *q_arg, None),
        _get_llm_sdpa_cache_shape(node, *cache_arg),
    )


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

    # A matched node is delegated without the partitioner's per-op checks, so
    # leave unsupported mask configurations unmatched to keep them on CPU.
    if not check_llm_sdpa_node(node):
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
