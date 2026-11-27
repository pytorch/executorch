# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

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


_AUTO_FUNCTIONALIZED = (
    torch.ops.higher_order.auto_functionalized,
    torch.ops.higher_order.auto_functionalized_v2,
)

# The key cache argument of each LLM SDPA op, whose dim 1 is the cache length.
_LLM_SDPA_KEY_CACHE_ARG = {
    "llama::custom_sdpa": "key",
    "llama::sdpa_with_kv_cache": "key_cache",
}


def get_llama_op_args(node: torch.fx.Node) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    The llama op called by node, directly or through auto_functionalized, and
    its arguments by schema name, with defaults for trimmed trailing args.
    auto_functionalized_v2 passes a mutable argument as metadata about its
    base instead, so that argument is missing.
    """
    if node.op != "call_function":
        return None, {}
    if node.target in _AUTO_FUNCTIONALIZED:
        op = node.args[0]
        return (op.name() if hasattr(op, "name") else None), dict(node.kwargs)
    schema = getattr(node.target, "_schema", None)
    if schema is None or not schema.name.startswith("llama::"):
        return None, {}
    args = {}
    for i, arg in enumerate(schema.arguments):
        if i < len(node.args):
            args[arg.name] = node.args[i]
        else:
            args[arg.name] = node.kwargs.get(arg.name, arg.default_value)
    return schema.name, args


def get_update_cache_readers(node: torch.fx.Node) -> List[torch.fx.Node]:
    """The custom_sdpa nodes that read the cache an update_cache writes."""
    # custom_sdpa reads either the mutated cache directly, or the cache returned
    # by the auto_functionalized update_cache.
    readers = set(node.users)
    for user in node.users:
        readers.update(user.users)
    _, args = get_llama_op_args(node)
    if isinstance(args.get("cache"), torch.fx.Node):
        readers.update(args["cache"].users)
    return [r for r in readers if get_llama_op_args(r)[0] == "llama::custom_sdpa"]


def _node_val(arg: Any) -> Optional[torch.Tensor]:
    val = arg.meta.get("val", None) if isinstance(arg, torch.fx.Node) else None
    return val if isinstance(val, torch.Tensor) else None


def llm_sdpa_mask_needs_texture(node: torch.fx.Node) -> bool:
    """
    Whether an LLM SDPA node (or an update_cache whose cache it reads) has an
    attn_mask in a different dtype than the query. Only the texture shaders,
    which sample the mask as floats, can read it.
    """
    name, args = get_llama_op_args(node)
    if name == "llama::update_cache":
        return any(
            llm_sdpa_mask_needs_texture(r) for r in get_update_cache_readers(node)
        )
    mask_val = _node_val(args.get("attn_mask"))
    q_val = _node_val(args.get("query"))
    return mask_val is not None and q_val is not None and mask_val.dtype != q_val.dtype


def llm_sdpa_has_attn_mask(node: torch.fx.Node) -> bool:
    """Whether an LLM SDPA node, or a custom_sdpa reading an update_cache's
    cache, has an attn_mask."""
    name, args = get_llama_op_args(node)
    if name == "llama::update_cache":
        return any(llm_sdpa_has_attn_mask(r) for r in get_update_cache_readers(node))
    return args.get("attn_mask") is not None


def check_llm_sdpa_node(node: torch.fx.Node) -> bool:
    """
    The Vulkan LLM SDPA is either causal without an attn_mask, or takes a 2D
    [S, C] fp32 or fp16 additive attn_mask with is_causal unset. Other
    configurations stay on CPU instead of failing at load time. An
    update_cache stays on CPU with the custom_sdpa that reads its cache, so
    that both see the same cache.
    """
    name, args = get_llama_op_args(node)
    if name == "llama::update_cache":
        return all(check_llm_sdpa_node(r) for r in get_update_cache_readers(node))
    if name not in _LLM_SDPA_KEY_CACHE_ARG:
        return True

    attn_mask = args.get("attn_mask")
    if attn_mask is None:
        return bool(args.get("is_causal"))
    mask_val = _node_val(attn_mask)
    q_val = _node_val(args.get("query"))
    if args.get("is_causal") or mask_val is None or q_val is None:
        return False
    if mask_val.dim() != 2 or not {mask_val.dtype, q_val.dtype} <= utils.FP_T:
        return False
    # The mask must be [S, C]. Sizes that are symbolic, or a cache that
    # auto_functionalized_v2 passes as a base, are checked at runtime.
    cache_val = _node_val(args.get(_LLM_SDPA_KEY_CACHE_ARG[name]))
    expected = [q_val.shape[1], None if cache_val is None else cache_val.shape[1]]
    return all(
        not isinstance(size, int) or not isinstance(exp, int) or size == exp
        for size, exp in zip(mask_val.shape, expected)
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
        # torch.export drops trailing arguments that equal their defaults
        _, args = get_llama_op_args(custom_sdpa_node)
        self.attn_mask_node = args.get("attn_mask")
        self.dropout_p_node = args.get("drpout_p", 0.0)
        self.is_causal_node = args.get("is_causal", False)
        self.scale_node = args.get("scale")

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


def _find_llm_sdpa_pattern(
    node: torch.fx.Node, with_attn_mask: bool
) -> Optional[CausalSDPAMatch]:
    if not is_custom_sdpa_node(node) or llm_sdpa_has_attn_mask(node) != with_attn_mask:
        return None

    # A matched node is delegated without the partitioner's per-op checks, so
    # leave unsupported mask configurations unmatched to keep them on CPU.
    if not check_llm_sdpa_node(node):
        return None

    matched_pattern = CausalSDPAMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


@register_pattern_detector("causal_sdpa")
def find_causal_sdpa_patterns(
    node: torch.fx.Node,
) -> Optional[CausalSDPAMatch]:
    return _find_llm_sdpa_pattern(node, with_attn_mask=False)


# The same fusion for a custom_sdpa with an additive attn_mask, as a separate
# pattern so that backends without mask support can skip it.
@register_pattern_detector("masked_sdpa")
def find_masked_sdpa_patterns(
    node: torch.fx.Node,
) -> Optional[CausalSDPAMatch]:
    return _find_llm_sdpa_pattern(node, with_attn_mask=True)


##
## Pattern Replacement
##


@register_pattern_replacement("causal_sdpa")
@register_pattern_replacement("masked_sdpa")
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
