# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Shared cache-attention and position-table graph utilities."""

import functools
from typing import Any, Dict, List, Optional

import torch

aten = torch.ops.aten

_CACHE_WRITES = (aten.index_copy_.default, aten.index_copy.default)


_WEIGHT_ARITHMETIC = (
    aten.add.Tensor,
    aten.add.Scalar,
    aten.mul.Tensor,
    aten.mul.Scalar,
    aten.rsub.Scalar,
    aten._to_copy.default,
    aten.to.dtype,
    aten.to.dtype_layout,
)


_LAST_DIM_NORM = (
    aten.add.Tensor,
    aten.add.Scalar,
    aten.mul.Tensor,
    aten.mul.Scalar,
    aten.pow.Tensor_Scalar,
    aten.rsqrt.default,
    aten.mean.dim,
    aten._to_copy.default,
    aten.to.dtype,
)


_REPEAT_OPS = (
    aten.slice.Tensor,
    aten.unsqueeze.default,
    aten.expand.default,
    aten.reshape.default,
    aten.view.default,
    aten.clone.default,
    aten._unsafe_view.default,
)


def fold_weight_arithmetic(gm: torch.fx.GraphModule) -> None:
    """Precompute values that depend only on weights, e.g. Gemma's RMSNorm
    scale ``1 + weight``, so the normalization keeps the
    ``x * rsqrt(mean(x^2) + eps) * weight`` shape QNN fuses into its RmsNorm
    op. Unfused, the square overflows in fp16."""
    graph = gm.graph
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _WEIGHT_ARITHMETIC:
            continue
        tensors = [a for a in node.args if isinstance(a, torch.fx.Node)]
        if not tensors or any(a.op != "get_attr" for a in tensors):
            continue
        args = [
            (
                functools.reduce(getattr, a.target.split("."), gm)
                if isinstance(a, torch.fx.Node)
                else a
            )
            for a in node.args
        ]
        with torch.no_grad():
            value = node.target(*args, **node.kwargs)
        name = f"_qnn_folded_{count}"
        count += 1
        gm.register_buffer(name, value.contiguous(), persistent=False)
        with graph.inserting_before(node):
            folded = graph.get_attr(name)
        folded.meta = dict(node.meta)
        node.replace_all_uses_with(folded)
        graph.erase_node(node)


def _on_last_dim(node: torch.fx.Node, dim: int) -> bool:
    rank = node.meta["val"].dim()
    return dim % rank == rank - 1


def _rotate_half_input(cat: Any) -> Optional[tuple]:
    """``(x, x1, x2)`` if ``cat`` is ``cat(-x2, x1)`` of ``x``'s halves."""
    if not isinstance(cat, torch.fx.Node) or cat.target != aten.cat.default:
        return None
    if len(cat.args[0]) != 2 or (
        len(cat.args) > 1 and not _on_last_dim(cat, cat.args[1])
    ):
        return None
    neg, x1 = cat.args[0]
    if neg.target != aten.neg.default:
        return None
    x2 = neg.args[0]
    if x1.target != aten.slice.Tensor or x2.target != aten.slice.Tensor:
        return None
    x = x1.args[0]
    if x2.args[0] is not x or not _on_last_dim(x, x1.args[1]):
        return None
    half = x.meta["val"].shape[-1] // 2
    if tuple(x1.args[2:4]) != (0, half) or x2.args[2] != half:
        return None
    return x, x1, x2


def _rope_in_halves(graph, x1, x2, cos, sin, half, tables) -> torch.fx.Node:
    def part(t, lo, hi):
        # Every layer reads the same cos/sin halves: slice each table once.
        if (t, lo, hi) not in tables:
            with graph.inserting_after(t):
                tables[(t, lo, hi)] = graph.call_function(
                    aten.slice.Tensor, (t, -1, lo, hi)
                )
        return tables[(t, lo, hi)]

    def mul(a, b):
        return graph.call_function(aten.mul.Tensor, (a, b))

    cos1, cos2 = part(cos, 0, half), part(cos, half, 2 * half)
    sin1, sin2 = part(sin, 0, half), part(sin, half, 2 * half)
    first = graph.call_function(aten.sub.Tensor, (mul(x1, cos1), mul(x2, sin1)))
    second = graph.call_function(aten.add.Tensor, (mul(x2, cos2), mul(x1, sin2)))
    return graph.call_function(aten.cat.default, ([first, second], -1))


def split_rotate_half(graph: torch.fx.Graph) -> None:
    """Rewrite RoPE written with ``rotate_half``, ``x * cos + cat(-x2, x1) * sin``,
    into its two halves, ``cat(x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2)``.

    Same values; HTP runs the halved form faster, as it skips building the
    rotated copy of ``x`` (the form Qualcomm's static_llama uses)."""
    tables: Dict[tuple, torch.fx.Node] = {}
    for add in list(graph.nodes):
        if add.target != aten.add.Tensor or not all(
            isinstance(a, torch.fx.Node) and a.target == aten.mul.Tensor
            for a in add.args[:2]
        ):
            continue
        for direct, rotated in (add.args[:2], add.args[1::-1]):
            x, cos = direct.args
            cat, sin = rotated.args
            found = _rotate_half_input(cat)
            if found is None or found[0] is not x:
                continue
            with graph.inserting_before(add):
                new = _rope_in_halves(
                    graph,
                    found[1],
                    found[2],
                    cos,
                    sin,
                    x.meta["val"].shape[-1] // 2,
                    tables,
                )
            new.meta["val"] = add.meta.get("val")
            add.replace_all_uses_with(new)
            break


def _rank(node: torch.fx.Node) -> int:
    val = node.meta.get("val")
    return val.dim() if isinstance(val, torch.Tensor) else 0


def _norm_chain(out: torch.fx.Node) -> Optional[tuple]:
    """``(source, nodes)`` if ``out`` normalizes one 4-D ``source`` over its
    last dimension, through ops of ``_LAST_DIM_NORM`` and weights of at most
    one dimension, and nothing else reads the intermediate values."""
    if out.op != "call_function" or out.target not in _LAST_DIM_NORM:
        return None
    nodes, stack, source = [], [out], None
    while stack:
        node = stack.pop()
        if node in nodes:
            continue
        if node.op == "call_function" and node.target in _LAST_DIM_NORM:
            if node.target == aten.mean.dim and not all(
                _on_last_dim(node.args[0], d) for d in node.args[1]
            ):
                return None
            nodes.append(node)
            stack.extend(a for a in node.args if isinstance(a, torch.fx.Node))
        elif _rank(node) == 4 and source in (None, node):
            source = node
        elif _rank(node) > 1:
            return None
    if source is None or not any(n.target == aten.mean.dim for n in nodes):
        return None
    if not _only_read_within(nodes, out):
        return None
    return source, nodes


def _only_read_within(nodes: List[torch.fx.Node], out: torch.fx.Node) -> bool:
    """Whether every value of ``nodes`` but ``out`` is read only by ``nodes``
    (shape assertions aside)."""
    return all(
        user in nodes or user.target == aten._assert_tensor_metadata.default
        for node in nodes
        if node is not out
        for user in node.users
    )


def norm_after_head_transpose(graph: torch.fx.Graph) -> None:
    """Move a per-head normalization that runs before the ``[B, S, H, D]`` →
    ``[B, H, S, D]`` transpose to after it, e.g. Qwen3's query and key norms.

    Same values, as the norm is over ``D``; HTP runs it per head after the
    attention is split by head, which the ``[B, S, H, D]`` layout turns into
    strided copies (the order Qualcomm's static_llama uses)."""
    for transpose in list(graph.nodes):
        if transpose.target != aten.transpose.int:
            continue
        out = transpose.args[0]
        dims = sorted(d % 4 for d in transpose.args[1:3])
        if out.meta["val"].dim() != 4 or dims != [1, 2] or len(out.users) != 1:
            continue
        found = _norm_chain(out)
        if found is None:
            continue
        source, nodes = found
        with graph.inserting_after(source):
            moved = graph.call_function(
                aten.transpose.int, (source, *transpose.args[1:3])
            )
        for node in nodes:
            for user in list(node.users):
                if user.target == aten._assert_tensor_metadata.default:
                    graph.erase_node(user)
            node.replace_input_with(source, moved)
        transpose.replace_all_uses_with(out)
        graph.erase_node(transpose)


def producer_cache_write(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Walk back from an attention K/V argument to the cache write it reads."""
    while node.op == "call_function":
        if node.target in _CACHE_WRITES:
            return node
        if node.target not in _REPEAT_OPS:
            return None
        node = node.args[0]
    return None


def _position_only_nodes(gm: torch.fx.GraphModule, position: torch.fx.Node):
    """Nodes computed from the position input and constants alone."""
    position_only = set()
    for node in gm.graph.nodes:
        if node is position or node.op == "get_attr":
            position_only.add(node)
        elif node.op in ("call_function", "call_module"):
            inputs = node.all_input_nodes
            if (
                inputs
                and all(i in position_only for i in inputs)
                and any(
                    i is position or i in position_only and i.op != "get_attr"
                    for i in inputs
                )
            ):
                position_only.add(node)
    return position_only


def _eval_nodes(gm, nodes, position, value):
    env = {}
    for node in gm.graph.nodes:
        if node not in nodes:
            continue
        if node is position:
            env[node] = value
        elif node.op == "get_attr":
            obj = gm
            for atom in node.target.split("."):
                obj = getattr(obj, atom)
            env[node] = obj
        else:
            args = torch.fx.node.map_arg(node.args, env.__getitem__)
            kwargs = torch.fx.node.map_arg(node.kwargs, env.__getitem__)
            if node.op == "call_module":
                env[node] = gm.get_submodule(node.target)(*args, **kwargs)
            else:
                env[node] = node.target(*args, **kwargs)
    return env


def fold_position_tables(  # noqa: C901
    gm, position, new_position, max_seq_len, dtype, index_meta=None
):
    """Replace every value that depends only on the position (RoPE cos/sin) with a
    lookup into a table precomputed for all ``max_seq_len`` positions.

    HTP has no efficient path for the trigonometry, and a 16-bit quantized angle
    loses precision quickly, so this is what QNN's own decoders do by hand.
    """
    position_only = _position_only_nodes(gm, position)
    frontier = [
        n
        for n in position_only
        if n.op != "get_attr"
        and n is not position
        and any(u not in position_only for u in n.users)
        and isinstance(n.meta.get("val"), torch.Tensor)
    ]
    if not frontier:
        return
    # Fold at the source of each value rather than at every per-layer view of it,
    # so all layers share one lookup.
    roots = []
    for node in frontier:
        while (
            node.target in _REPEAT_OPS
            and node.args[0] in position_only
            and node.args[0] is not position
            and node.args[0].op != "get_attr"
            and isinstance(node.args[0].meta.get("val"), torch.Tensor)
        ):
            node = node.args[0]
        if node not in roots:
            roots.append(node)
    frontier = roots
    pos_val = position.meta["val"]
    ar = pos_val.shape[-1]
    # Every position of the step at once: values that depend on the position
    # element-wise (RoPE) are the same along the step axis then.
    rows = [
        _eval_nodes(
            gm,
            position_only,
            position,
            torch.full(tuple(pos_val.shape), p, dtype=pos_val.dtype),
        )
        for p in range(max_seq_len)
    ]
    graph = gm.graph
    # Build and check every table on the untouched graph first, then rewrite.
    tables: List[torch.Tensor] = []
    plans = []
    probe = torch.arange(ar, dtype=pos_val.dtype).reshape(pos_val.shape)
    if ar < max_seq_len:
        probe = probe + 1
    probe_env = _eval_nodes(gm, position_only, position, probe) if ar > 1 else None
    for node in frontier:
        out_shape = list(node.meta["val"].shape)
        if ar == 1:
            step_axis = None
            entries = [row[node] for row in rows]
        else:
            if ar not in out_shape:
                raise ValueError(
                    f"Cannot fold {node.name}: no axis matches the {ar} step positions."
                )
            step_axis = out_shape.index(ar)
            entries = [row[node].select(step_axis, 0) for row in rows]
        table = torch.stack(entries).to(dtype).contiguous()
        if step_axis is not None:
            # The lookup must reproduce the value at mixed positions.
            got = (
                table.index_select(0, probe.reshape(-1).long())
                .movedim(0, step_axis)
                .reshape(out_shape)
            )
            if not torch.allclose(got, probe_env[node].to(dtype), atol=1e-5):
                raise ValueError(
                    f"{node.name} is not element-wise in the position; cannot fold it."
                )
        index = next(
            (
                j
                for j, t in enumerate(tables)
                if t.shape == table.shape and torch.equal(t, table)
            ),
            None,
        )
        if index is None:
            index = len(tables)
            tables.append(table)
            gm.register_buffer(f"_qnn_position_table_{index}", table, persistent=False)
        plans.append((node, index, step_axis, out_shape, table.dim()))
    for node, index, step_axis, out_shape, table_dim in plans:
        with graph.inserting_before(node):
            attr = graph.get_attr(f"_qnn_position_table_{index}")
            flat = graph.call_function(aten.view.default, (new_position, [-1]))
            if index_meta is not None:
                flat.meta.update(index_meta)
            picked = graph.call_function(aten.index_select.default, (attr, 0, flat))
            if step_axis is not None and step_axis != 0:
                perm = list(range(1, table_dim))
                perm.insert(step_axis, 0)
                picked = graph.call_function(aten.permute.default, (picked, perm))
            lookup = graph.call_function(aten.reshape.default, (picked, out_shape))
        node.replace_all_uses_with(lookup)
    # Shape asserts on the folded values would keep the old subgraph alive.
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and node.target
            in (aten._assert_tensor_metadata.default, aten._assert_scalar.default)
            and all(i in position_only for i in node.all_input_nodes)
        ):
            graph.erase_node(node)
