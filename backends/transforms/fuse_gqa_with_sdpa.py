# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fold KV head repetition into scaled_dot_product_attention."""

import torch

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.shape_prop import _extract_tensor_metadata


def _resolve_aten(target: object) -> "torch._ops.OpOverload | None":
    """Return the underlying aten OpOverload for an edge or plain aten target."""
    inner = getattr(target, "_op", None)
    if isinstance(inner, torch._ops.OpOverload):
        return inner
    if isinstance(target, torch._ops.OpOverload):
        return target
    return None


def _single_user(node: object) -> bool:
    return isinstance(node, Node) and len(node.users) == 1


def _matches_shape(node: Node, shape: list) -> bool:
    value = node.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and value.dim() == len(shape)
        and all(statically_known_true(a == b) for a, b in zip(value.shape, shape))
    )


def _unwrap_rank_normalization(node: object) -> Node | None:
    if (
        not isinstance(node, Node)
        or _resolve_aten(node.target) is not torch.ops.aten.unsqueeze_copy.default
        or not node.args
    ):
        return None
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    inner = node.args[0]
    value = inner.meta.get("val") if isinstance(inner, Node) else None
    if (
        dim == 0
        and isinstance(value, torch.Tensor)
        and value.dim() == 3
        and _matches_shape(node, [1, *value.shape])
    ):
        return inner
    return None


def _unwrap_repeat_interleave(node: Node, rank: int, n_heads: int) -> Node | None:
    base = node.args[0] if node.args else node.kwargs.get("self")
    value = base.meta.get("val") if isinstance(base, Node) else None
    if rank not in (3, 4) or not isinstance(value, torch.Tensor) or value.dim() != rank:
        return None
    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim")
    repeats = node.args[1] if len(node.args) > 1 else node.kwargs.get("repeats")
    head_axis = rank - 3
    n_kv = value.shape[head_axis]
    if (
        dim not in (head_axis, -3)
        or not isinstance(n_kv, int)
        or n_kv <= 0
        or n_heads <= n_kv
        or n_heads % n_kv
        or not isinstance(repeats, int)
        or repeats != n_heads // n_kv
    ):
        return None
    folded = list(value.shape)
    folded[head_axis] = n_heads
    return base if _matches_shape(node, folded) else None


def _unwrap_repeat_kv(node: object, rank: int, n_heads: int) -> Node | None:
    """Match head repetition for [H, S, D] or [B, H, S, D] KV tensors."""
    if not _single_user(node):
        return None
    target = _resolve_aten(node.target)
    if target is torch.ops.aten.repeat_interleave.self_int:
        return _unwrap_repeat_interleave(node, rank, n_heads)
    if target is not torch.ops.aten.view_copy.default:
        return None
    inner = node.args[0] if node.args else node.kwargs.get("self")
    clone = None
    if _resolve_aten(getattr(inner, "target", None)) is torch.ops.aten.clone.default:
        if not _single_user(inner):
            return None
        clone = inner
        inner = inner.args[0] if inner.args else inner.kwargs.get("self")

    if _resolve_aten(
        getattr(inner, "target", None)
    ) is not torch.ops.aten.expand_copy.default or not _single_user(inner):
        return None
    expand = inner
    unsqueeze = expand.args[0] if expand.args else expand.kwargs.get("self")
    if _resolve_aten(
        getattr(unsqueeze, "target", None)
    ) is not torch.ops.aten.unsqueeze_copy.default or not _single_user(unsqueeze):
        return None
    base = unsqueeze.args[0] if unsqueeze.args else unsqueeze.kwargs.get("self")
    value = base.meta.get("val") if isinstance(base, Node) else None
    if not isinstance(value, torch.Tensor) or value.dim() != rank:
        return None

    head_axis = rank - 3
    repeat_axis = head_axis + 1
    dim = unsqueeze.args[1] if len(unsqueeze.args) > 1 else unsqueeze.kwargs.get("dim")
    if dim not in (repeat_axis, -3):
        return None
    n_kv = value.shape[head_axis]
    if not isinstance(n_kv, int) or n_kv <= 0 or n_heads <= n_kv or n_heads % n_kv:
        return None

    shape = list(value.shape)
    inserted = shape[:repeat_axis] + [1] + shape[repeat_axis:]
    expanded = shape[:repeat_axis] + [n_heads // n_kv] + shape[repeat_axis:]
    folded = shape[:head_axis] + [n_heads] + shape[head_axis + 1 :]
    # Prove that only the head-repeat dimension changed, not sequence or batch.
    if not (
        _matches_shape(unsqueeze, inserted)
        and _matches_shape(expand, expanded)
        and (clone is None or _matches_shape(clone, expanded))
        and _matches_shape(node, folded)
    ):
        return None
    return base


class FuseGQAWithSDPAPass(ExportPass):
    """Fold KV head repetition into scaled_dot_product_attention.

    Grouped-query attention repeats each KV head n_rep = n_heads // n_kv times so
    K/V match the query head count before SDPA. aten SDPA can do that broadcast
    itself via ``enable_gqa=True``, so when both K and V feed SDPA through a
    repeat_kv chain or repeat_interleave we drop the expansion and pass the un-repeated
    rank-3 ``[n_kv, T, D]`` or rank-4 ``[B, n_kv, T, D]`` tensors directly,
    setting ``enable_gqa=True``. This removes
    the materialized KV copies (expand + clone) from the graph. ``enable_gqa``
    broadcasts kv head i to query heads ``[i*n_rep, (i+1)*n_rep)``, matching
    repeat_kv's head ordering, so the result is unchanged. Leading singleton
    wrappers from SDPA rank normalization are preserved around the base tensors.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or _resolve_aten(node.target) is not sdpa:
                continue
            if node.kwargs.get("enable_gqa"):
                continue
            q, k, v = (
                node.args[i] if i < len(node.args) else node.kwargs.get(name)
                for i, name in enumerate(("query", "key", "value"))
            )
            q_val = q.meta.get("val") if isinstance(q, Node) else None
            if not isinstance(q_val, torch.Tensor) or q_val.dim() not in (3, 4):
                continue
            n_heads = q_val.shape[-3]
            if not isinstance(n_heads, int):
                continue
            k_inner = _unwrap_rank_normalization(k)
            v_inner = _unwrap_rank_normalization(v)
            normalized = (
                q_val.dim() == 4 and k_inner is not None and v_inner is not None
            )
            rank = 3 if normalized else q_val.dim()
            k_base = _unwrap_repeat_kv(k_inner if normalized else k, rank, n_heads)
            v_base = _unwrap_repeat_kv(v_inner if normalized else v, rank, n_heads)
            if k_base is None or v_base is None:
                continue
            if k_base.meta["val"].shape[-3] != v_base.meta["val"].shape[-3]:
                continue

            args, kwargs = list(node.args), dict(node.kwargs)
            for i, name, base, wrapper in (
                (1, "key", k_base, k),
                (2, "value", v_base, v),
            ):
                if normalized:
                    # Keep any other consumers of the original wrapper unchanged.
                    with graph.inserting_before(node):
                        padded = graph.call_function(wrapper.target, (base, 0))
                    value = torch.ops.aten.unsqueeze_copy.default(base.meta["val"], 0)
                    padded.meta = {
                        **wrapper.meta,
                        "val": value,
                        "tensor_meta": _extract_tensor_metadata(value),
                    }
                    base = padded
                if i < len(args):
                    args[i] = base
                else:
                    kwargs[name] = base
            node.args = tuple(args)
            node.kwargs = {**kwargs, "enable_gqa": True}
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)
