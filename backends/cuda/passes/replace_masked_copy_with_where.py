# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph Transformation Pass for Masked Copies.

Rewrites a copy through a boolean mask into a select:

    out[mask] = src[mask]  ->  out = torch.where(mask, src, out)

A mask index has a result size that depends on the data, so ``index.Tensor``
and ``index_put`` with a mask run in the backend's index shims, and each call
reads the mask's count back to the host, which waits for the GPU. When the
values written are the same mask's picks from a tensor shaped like the
destination, every element goes back where it came from, so ``where`` gives
the same result, element for element, with no count and no host read.
"""

from typing import Any, List, Optional

import sympy
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

_INDEX = exir_ops.edge.aten.index.Tensor
_INDEX_PUT = exir_ops.edge.aten.index_put.default
_ASSERT_SCALAR = torch.ops.aten._assert_scalar.default


def _same_value(a: Any, b: Any) -> bool:
    """
    Whether a and b are the same node, or the same deterministic op applied to
    the same values, as with ``x[m.view(-1)] = y[m.view(-1)]``.
    """
    # A work list, not recursion: two long mask computations would pass the
    # recursion limit, and each node pair is compared once, since shared
    # inputs would otherwise be compared again on every path that reaches them.
    pending = [(a, b)]
    compared = set()
    while pending:
        a, b = pending.pop()
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            pending.extend(zip(a, b))
            continue
        if not isinstance(a, Node) or not isinstance(b, Node):
            if isinstance(a, Node) or isinstance(b, Node) or a != b:
                return False
            continue
        if a is b or (a, b) in compared:
            continue
        compared.add((a, b))
        if a.op != "call_function" or a.target != b.target:
            return False
        if (
            isinstance(a.target, (EdgeOpOverload, torch._ops.OpOverload))
            and torch.Tag.nondeterministic_seeded in a.target.tags
        ):
            return False
        if a.kwargs.keys() != b.kwargs.keys():
            return False
        pending.append((a.args, b.args))
        pending.extend((a.kwargs[k], b.kwargs[k]) for k in a.kwargs)
    return True


class ReplaceMaskedCopyWithWherePass(PassBase):
    """
    Replace ``index_put(out, indices, index(src, indices))`` with
    ``where(mask, src, out)`` when ``indices`` is one boolean mask (after any
    number of ``None``), ``src`` and ``out`` have the same shape and dtype, and
    the put does not accumulate.

    The two masks may be one node or two equal computations. The ``index`` is
    then removed too if nothing else reads it but the range checks export adds
    on its size, since those hold for any mask.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            mask = self._mask_of_copy(node)
            if mask is None:
                continue
            self._replace(graph_module, node, mask)
            modified = True
        if modified:
            graph_module.graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    @staticmethod
    def _mask_of_copy(node: Node) -> Optional[Node]:
        """The mask of an index_put that copies through it, else None."""
        if node.op != "call_function" or node.target != _INDEX_PUT:
            return None
        out, indices, values = node.args[:3]
        accumulate = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
        )
        if accumulate or not isinstance(values, Node) or values.target != _INDEX:
            return None
        src, src_indices = values.args
        if not _same_value(list(src_indices), list(indices)):
            return None
        masks = [index for index in indices if index is not None]
        if len(masks) != 1:
            return None
        mask_val = masks[0].meta.get("val")
        out_val = out.meta.get("val")
        src_val = src.meta.get("val")
        if not all(isinstance(v, torch.Tensor) for v in (mask_val, out_val, src_val)):
            return None
        if mask_val.dtype != torch.bool or mask_val.dim() == 0:
            return None
        if src_val.dtype != out_val.dtype or src_val.dim() != out_val.dim():
            return None
        if not all(
            statically_known_true(a == b) for a, b in zip(src_val.shape, out_val.shape)
        ):
            return None
        return masks[0]

    def _replace(self, graph_module: GraphModule, node: Node, mask: Node) -> None:
        graph = graph_module.graph
        out, indices, values = node.args[:3]
        src = values.args[0]
        position = indices.index(mask)
        # The mask covers the dims after the leading Nones; broadcasting lines
        # it up from the right, so only the dims after it need a size-1 dim.
        trailing = out.meta["val"].dim() - position - mask.meta["val"].dim()
        with graph.inserting_before(node):
            condition = mask
            for _ in range(trailing):
                condition_val = condition.meta["val"].unsqueeze(-1)
                condition = graph.call_function(
                    exir_ops.edge.aten.unsqueeze_copy.default, args=(condition, -1)
                )
                condition.meta["val"] = condition_val
            where = graph.call_function(
                exir_ops.edge.aten.where.self, args=(condition, src, out)
            )
        where.meta = node.meta.copy()
        node.replace_all_uses_with(where)
        graph.erase_node(node)
        dead = self._only_range_checked(values, mask, position)
        if dead is not None:
            for user in reversed(dead):
                graph.erase_node(user)

    @staticmethod
    def _only_range_checked(
        index: Node, mask: Node, position: int
    ) -> Optional[List[Node]]:
        """
        The index and everything that reads it, in graph order, if all that
        reads it is symbolic math on its sizes and checks that hold for every
        count, its size at ``position``, in [0, mask.numel()], else None.
        """
        count = index.meta["val"].shape[position]
        if not isinstance(count, torch.SymInt):
            return None
        numel = mask.meta["val"].numel()
        if isinstance(numel, torch.SymInt):
            numel = numel.node.shape_env.bound_sympy(numel.node.expr).upper
        within = ValueRanges(0, numel)
        found = [index]
        pending = list(index.users)
        while pending:
            user = pending.pop()
            if user in found:
                continue
            if user.target == _ASSERT_SCALAR:
                condition = user.args[0]
                expr = (
                    condition.meta.get("val").node.expr
                    if isinstance(condition, Node)
                    and isinstance(condition.meta.get("val"), torch.SymBool)
                    else None
                )
                if expr is None or expr.free_symbols != {count.node.expr}:
                    return None
                bounds = bound_sympy(expr, {count.node.expr: within})
                if not (bounds.is_singleton() and bounds.lower == sympy.true):
                    return None
            # Computing a symbolic value has no effect of its own, and its
            # readers are visited in turn.
            elif not isinstance(user.meta.get("val"), (torch.SymInt, torch.SymBool)):
                return None
            found.append(user)
            pending.extend(user.users)
        order = {n: i for i, n in enumerate(index.graph.nodes)}
        return sorted(found, key=order.__getitem__)
