# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse HuggingFace frequency construction and rotate_half into ``native.rope``."""

import math

import torch

from executorch.backends.native.custom_ops import rope_op
from executorch.backends.native.passes._utils import (
    _fake_tensor,
    _resolve_aten,
    _single_user,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true

_SLICE_OPS = {
    torch.ops.aten.slice_copy.Tensor,
    torch.ops.aten.slice.Tensor,
}


def _ndim(node: object) -> "int | None":
    t = _fake_tensor(node)
    return t.dim() if t is not None else None


def _last_dim_size(node: object) -> "int | None":
    t = _fake_tensor(node)
    if t is None:
        return None
    size = t.shape[-1]
    return int(size) if isinstance(size, int) else None


def _half_slice_source(node: object, first: bool) -> "Node | None":
    """Return ``X`` if ``node`` slices X's last dim to its first/second half.

    Matches the ``slice_copy`` (functional) form, since this runs before
    ReplaceCopyWithAliasPass. The slice must be single-use, on the last dim, and
    cover exactly ``[0, d/2)`` (first) or ``[d/2, d)`` (second) of an even last
    dim ``d``.
    """
    if not isinstance(node, Node) or _resolve_aten(node.target) not in _SLICE_OPS:
        return None
    if not _single_user(node):
        return None
    args = node.args
    if len(args) < 2 or (args[4] if len(args) > 4 else node.kwargs.get("step", 1)) != 1:
        return None
    src, dim = args[0], args[1]
    if not isinstance(src, Node) or not isinstance(dim, int):
        return None
    d = _last_dim_size(src)
    nd = _ndim(src)
    if d is None or nd is None or d % 2 != 0 or dim % nd != nd - 1:
        return None
    half = d // 2
    start = args[2] if len(args) > 2 else 0
    end = args[3] if len(args) > 3 else None
    start = 0 if start is None else start
    end = d if end is None else end
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    end = min(end, d)
    if first:
        return src if (start == 0 and end == half) else None
    return src if (start == half and end == d) else None


def _rotate_half_source(rotated_mul: Node) -> "tuple[Node, Node] | None":
    """Match ``cat([-x2, x1], -1) * sin`` and return ``(X, sin)``."""
    if len(rotated_mul.args) < 2:
        return None
    a0, a1 = rotated_mul.args[0], rotated_mul.args[1]
    cat_node, sin = None, None
    for cand, other in ((a0, a1), (a1, a0)):
        if (
            isinstance(cand, Node)
            and _resolve_aten(cand.target) is torch.ops.aten.cat.default
        ):
            cat_node, sin = cand, other
            break
    if cat_node is None or not isinstance(sin, Node) or not _single_user(cat_node):
        return None

    tensors = cat_node.args[0]
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
        return None
    neg_node, x1_node = tensors
    if not isinstance(neg_node, Node) or not isinstance(x1_node, Node):
        return None
    if _resolve_aten(
        neg_node.target
    ) is not torch.ops.aten.neg.default or not _single_user(neg_node):
        return None
    x2_node = neg_node.args[0]

    x_from_x1 = _half_slice_source(x1_node, first=True)
    x_from_x2 = _half_slice_source(x2_node, first=False)
    if x_from_x1 is None or x_from_x1 is not x_from_x2:
        return None

    cat_dim = cat_node.args[1] if len(cat_node.args) > 1 else 0
    nd = _ndim(x_from_x1)
    if not isinstance(cat_dim, int) or nd is None or cat_dim % nd != nd - 1:
        return None
    return (x_from_x1, sin)


def _other_factor(mul_node: Node, x: Node) -> "Node | None":
    """Return the other multiplicand of ``mul_node`` if one of them is ``x``."""
    if len(mul_node.args) < 2:
        return None
    a0, a1 = mul_node.args[0], mul_node.args[1]
    if a0 is x and isinstance(a1, Node):
        return a1
    if a1 is x and isinstance(a0, Node):
        return a0
    return None


def _rope_operands(add_node: Node) -> "tuple[Node, Node, Node] | None":
    """Match ``x*cos + rotate_half(x)*sin`` and return ``(x, cos, sin)``.

    ``rotate_half(x) = cat([-x2, x1], -1)`` with ``x1 = x[..., :d/2]`` and
    ``x2 = x[..., d/2:]``. Returns None if the add is not this pattern. All
    intermediate nodes must be single-use so they DCE away after the rewrite;
    ``x``/``cos``/``sin`` may be shared (q and k reuse the same cos/sin).
    """
    mul = torch.ops.aten.mul.Tensor
    if add_node.kwargs.get("alpha", 1) != 1:
        return None
    operands = [a for a in add_node.args if isinstance(a, Node)]
    if len(operands) != 2:
        return None
    a, b = operands
    for direct, rotated in ((a, b), (b, a)):
        if _resolve_aten(direct.target) is not mul or not _single_user(direct):
            continue
        if _resolve_aten(rotated.target) is not mul or not _single_user(rotated):
            continue
        match = _rotate_half_source(rotated)
        if match is None:
            continue
        x, sin = match
        cos = _other_factor(direct, x)
        if cos is None:
            continue
        return (x, cos, sin)
    return None


def _last_dim_slice(node: object) -> "tuple[Node, int, int] | None":
    """Return ``(src, start, end)`` if ``node`` slices ``src``'s last dim.

    Matches the functional ``slice_copy`` (and aliasing ``slice``) forms on the
    last dim with concrete int bounds.
    """
    if not isinstance(node, Node) or _resolve_aten(node.target) not in _SLICE_OPS:
        return None
    args = node.args
    if len(args) < 2 or (args[4] if len(args) > 4 else node.kwargs.get("step", 1)) != 1:
        return None
    src, dim = args[0], args[1]
    if not isinstance(src, Node) or not isinstance(dim, int):
        return None
    nd = _ndim(src)
    d = _last_dim_size(src)
    if nd is None or d is None or dim % nd != nd - 1:
        return None
    start = args[2] if len(args) > 2 else 0
    end = args[3] if len(args) > 3 else None
    start = 0 if start is None else start
    end = d if end is None else end
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return (src, start, min(end, d))


def _partial_rotary_cat(x: object, add_node: Node) -> "tuple[Node, Node] | None":
    """Detect the partial-rotary wrapper around a rope apply.

    Returns ``(full_x, cat_node)`` when ``x`` is ``full_x[..., :rotary]`` and the
    apply result ``add_node`` is concatenated with the pass-through
    ``full_x[..., rotary:]`` on the last dim; else None.
    """
    x_slice = _last_dim_slice(x)
    if x_slice is None:
        return None
    full_x, x_start, rotary = x_slice
    if x_start != 0:
        return None
    users = list(add_node.users)
    if len(users) != 1:
        return None
    cat = users[0]
    if _resolve_aten(cat.target) is not torch.ops.aten.cat.default:
        return None
    tensors = cat.args[0]
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
        return None
    if tensors[0] is not add_node:
        return None
    pass_slice = _last_dim_slice(tensors[1])
    if pass_slice is None:
        return None
    pass_src, pass_start, pass_end = pass_slice
    d = _last_dim_size(full_x)
    if pass_src is not full_x or pass_start != rotary or d is None or pass_end != d:
        return None
    cat_dim = cat.args[1] if len(cat.args) > 1 else 0
    nd = _ndim(full_x)
    if not isinstance(cat_dim, int) or nd is None or cat_dim % nd != nd - 1:
        return None
    return (full_x, cat)


def _is_op(node: object, *ops: torch._ops.OpOverload) -> bool:
    return isinstance(node, Node) and _resolve_aten(node.target) in ops


def _same_shape(a: object, b: object) -> bool:
    ta, tb = _fake_tensor(a), _fake_tensor(b)
    return (
        ta is not None
        and tb is not None
        and ta.dim() == tb.dim()
        and all(statically_known_true(x == y) for x, y in zip(ta.shape, tb.shape))
    )


def _identity_views(node: Node) -> Node:
    # Matmul decomposition adds shape-preserving expand/view wrappers.
    while _is_op(
        node,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand_copy.default,
        torch.ops.aten.expand.default,
    ) and _same_shape(node, node.args[0]):
        node = node.args[0]
    return node


def _unsqueeze_source(node: object, dim: int) -> "Node | None":
    if not _is_op(
        node, torch.ops.aten.unsqueeze_copy.default, torch.ops.aten.unsqueeze.default
    ):
        return None
    nd = _ndim(node)
    axis = node.args[1]
    if nd is None or not isinstance(axis, int) or axis % nd != dim:
        return None
    return node.args[0]


def _float_source(node: Node) -> "Node | None":
    """Peel one FP32 cast, without discarding earlier precision changes."""
    t = _fake_tensor(node)
    if t is None or t.dtype != torch.float32:
        return None
    if _is_op(node, torch.ops.aten._to_copy.default):
        src = node.args[0]
        src_t = _fake_tensor(src)
        if src_t is None or src_t.device != t.device:
            return None
        return src
    return node


def _scalar_constant(node: object) -> "float | None":
    if isinstance(node, (int, float)):
        return float(node) if math.isfinite(node) else None
    t = _fake_tensor(node)
    constant = getattr(t, "constant", None)
    if constant is None or constant.dim() != 0:
        return None
    value = constant.item()
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(value)
        else None
    )


def _trig_source(
    table: Node, trig: torch._ops.OpOverload, x: Node
) -> "tuple[Node, float] | None":
    """Match FP32 trig -> optional scale -> model-dtype cast -> head broadcast."""
    t, xt = _fake_tensor(table), _fake_tensor(x)
    if (
        t is None
        or xt is None
        or t.dtype != xt.dtype
        or xt.dim() != 4
        or not xt.is_floating_point()
    ):
        return None
    node = _unsqueeze_source(table, 1)
    if node is None:
        return None
    if _is_op(node, torch.ops.aten._to_copy.default):
        src = node.args[0]
        st = _fake_tensor(src)
        if st is None or st.device != t.device:
            return None
        node = src
    nt = _fake_tensor(node)
    if nt is None or nt.dtype != torch.float32:
        return None
    scale = 1.0
    if _is_op(node, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
        a, b = node.args[:2]
        if _is_op(b, trig):
            a, b = b, a
        scale = _scalar_constant(b)
        if scale is None:
            return None
        node = a
    if not _is_op(node, trig):
        return None
    phase = node.args[0]
    pt = _fake_tensor(phase)
    if pt is None or pt.dtype != torch.float32 or pt.dim() != 3:
        return None
    return phase, scale


def _frequency_product(phase: Node) -> "tuple[Node, Node] | None":  # noqa: C901
    """Recover positions/inverse frequencies from HF's duplicated batched outer product."""
    if not _is_op(phase, torch.ops.aten.cat.default):
        return None
    tensors = phase.args[0]
    if (
        not isinstance(tensors, (list, tuple))
        or len(tensors) != 2
        or tensors[0] is not tensors[1]
    ):
        return None
    if (phase.args[1] if len(phase.args) > 1 else 0) not in (-1, 2):
        return None
    transposed = tensors[0]
    if _is_op(
        transposed, torch.ops.aten.permute_copy.default, torch.ops.aten.permute.default
    ):
        if list(transposed.args[1]) != [0, 2, 1]:
            return None
    elif _is_op(
        transposed, torch.ops.aten.transpose_copy.int, torch.ops.aten.transpose.int
    ):
        if {transposed.args[1] % 3, transposed.args[2] % 3} != {1, 2}:
            return None
    else:
        return None
    product = _identity_views(transposed.args[0])
    if not _is_op(product, torch.ops.aten.bmm.default, torch.ops.aten.matmul.default):
        return None
    inv, pos = (_identity_views(a) for a in product.args[:2])
    pos = _float_source(pos)
    if pos is None:
        return None
    position_ids = _unsqueeze_source(pos, 1)
    if position_ids is None or _ndim(position_ids) != 2:
        return None
    # Only the batch dimension may be expanded on inverse frequencies.
    while _is_op(
        inv, torch.ops.aten.expand_copy.default, torch.ops.aten.expand.default
    ):
        src = inv.args[0]
        it, st = _fake_tensor(inv), _fake_tensor(src)
        if it is None or st is None or it.dim() != 3 or st.dim() != 3:
            return None
        if not all(
            statically_known_true(a == b) for a, b in zip(it.shape[1:], st.shape[1:])
        ):
            return None
        inv = src
    inv = _float_source(inv)
    if inv is None:
        return None
    inv = _unsqueeze_source(inv, 2)
    inv_freq = _unsqueeze_source(inv, 0)
    if inv_freq is None or _ndim(inv_freq) != 1:
        return None
    # Both matmul operands and the product must use the FP32 contract.
    for n in (*product.args[:2], product):
        t = _fake_tensor(n)
        if t is None or t.dtype != torch.float32:
            return None
    return position_ids, inv_freq


def _rope_frequencies(
    x: Node, cos: Node, sin: Node
) -> "tuple[Node, Node, float] | None":
    cosine = _trig_source(cos, torch.ops.aten.cos.default, x)
    sine = _trig_source(sin, torch.ops.aten.sin.default, x)
    if cosine is None or sine is None or cosine != sine:
        return None
    phase, scale = cosine
    operands = _frequency_product(phase)
    if operands is None:
        return None
    positions, inv_freq = operands
    xt, pt, it, ct = (_fake_tensor(n) for n in (x, positions, inv_freq, cos))
    if (
        xt is None
        or pt is None
        or it is None
        or ct is None
        or not _same_shape(cos, sin)
    ):
        return None
    checks = (
        (pt.shape[0] == 1) | (pt.shape[0] == xt.shape[0]),
        pt.shape[1] == xt.shape[2],
        2 * it.shape[0] == xt.shape[3],
        ct.shape[0] == pt.shape[0],
        ct.shape[2] == pt.shape[1],
        ct.shape[3] == xt.shape[3],
    )
    if not all(statically_known_true(check) for check in checks):
        return None
    return positions, inv_freq, scale


class FuseRoPEPass(ExportPass):
    """Fuse HF frequency construction and split-half rotary embedding.

    Matches FP32 position/inverse-frequency matmul, duplicated phases, scaled
    cos/sin cast to the input dtype, and the rotate_half apply. Emits
    ``native.rope(x, position_ids, inv_freq, False, attention_scale)``.
    Precomputed tables and unrecognized frequency constructions stay decomposed.

    Partial rotary also folds the prefix slice and passthrough concatenation:
    the op rotates ``2 * inv_freq.numel()`` channels of the full input. Frequency
    producers may be shared by Q/K or other layers; DCE removes only unused ones.
    Runs pre-partition, before ReplaceCopyWithAliasPass.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        add_op = torch.ops.aten.add.Tensor
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or _resolve_aten(node.target) is not add_op:
                continue
            operands = _rope_operands(node)
            if operands is None:
                continue
            x, cos, sin = operands
            frequencies = _rope_frequencies(x, cos, sin)
            if frequencies is None:
                continue
            position_ids, inv_freq, attention_scale = frequencies

            # Partial rotary: rewrite the slice/apply/cat to one native.rope on the
            # full tensor (rope passes the tail through), so no strided slice is
            # left behind. Otherwise rewrite the apply add in place.
            partial = _partial_rotary_cat(x, node)
            rope_input = partial[0] if partial is not None else x
            replaced = partial[1] if partial is not None else node
            if not _same_shape(rope_input, replaced):
                continue

            with graph.inserting_before(replaced):
                fused = graph.call_function(
                    rope_op,
                    (rope_input, position_ids, inv_freq, False, attention_scale),
                )
            fused.meta.update(replaced.meta)
            replaced.replace_all_uses_with(fused)
            graph.erase_node(replaced)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)
