# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import operator
from typing import Any, Dict, List, Optional, Set

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils import _pytree as pytree

logger: logging.Logger = logging.getLogger(__name__)

aten = torch.ops.aten

# Views a write can go through, mapped to the op that writes the result back
# into the viewed tensor.
_VIEW_TO_SCATTER: Dict[torch._ops.OpOverload, torch._ops.OpOverload] = {
    aten.select.int: aten.select_scatter.default,
    aten.slice.Tensor: aten.slice_scatter.default,
    aten.diagonal.default: aten.diagonal_scatter.default,
}

# In-place ops that are rewritten, mapped to their functional variants.
_INPLACE_TO_FUNCTIONAL: Dict[torch._ops.OpOverload, torch._ops.OpOverload] = {
    aten.copy_.default: aten.copy.default,
    aten.fill_.Scalar: aten.fill.Scalar,
    aten.fill_.Tensor: aten.fill.Tensor,
    aten.zero_.default: aten.zero.default,
    aten.add_.Tensor: aten.add.Tensor,
    aten.sub_.Tensor: aten.sub.Tensor,
    aten.mul_.Tensor: aten.mul.Tensor,
    aten.div_.Tensor: aten.div.Tensor,
}

# Functional ops used instead when the other operand is a Python scalar, so a
# quantizer does not lift and quantize the scalar the in-place op kept.
_TENSOR_TO_SCALAR: Dict[torch._ops.OpOverload, torch._ops.OpOverload] = {
    aten.add.Tensor: aten.add.Scalar,
    aten.sub.Tensor: aten.sub.Scalar,
    aten.mul.Tensor: aten.mul.Scalar,
    aten.div.Tensor: aten.div.Scalar,
}

_PROVENANCE_KEYS = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "custom",
    "from_node",
)


def _returns_alias(node: Any) -> bool:
    return (
        isinstance(node, Node)
        and isinstance(node.target, torch._ops.OpOverload)
        and any(r.alias_info is not None for r in node.target._schema.returns)
    )


def _is_alias(node: Any) -> bool:
    if isinstance(node, Node) and node.target is operator.getitem:
        return _returns_alias(node.args[0])
    return _returns_alias(node)


def _is_write(node: Node) -> bool:
    if not isinstance(node.target, torch._ops.OpOverload) or not node.args:
        return False
    alias_info = node.target._schema.arguments[0].alias_info
    return alias_info is not None and alias_info.is_write


def _aliases(base: Node) -> Set[Node]:
    """Views of `base`, in-place ops on it, and getitems of its multi-output
    views such as unbind or split."""
    aliases: Set[Node] = set()
    to_visit = [base]
    while to_visit:
        n = to_visit.pop()
        for user in n.users:
            if user not in aliases and (
                (_returns_alias(user) and user.args[0] is n)
                or (user.target is operator.getitem and _returns_alias(n))
            ):
                aliases.add(user)
                to_visit.append(user)
    return aliases


def _fake_run(op: torch._ops.OpOverload, args: Any, kwargs: Any) -> Any:
    args, kwargs = pytree.tree_map_only(
        Node, lambda n: n.meta.get("val"), (args, kwargs)
    )
    fake_mode = torch._guards.detect_fake_mode(pytree.tree_leaves((args, kwargs)))
    if fake_mode is None:
        return None
    with fake_mode:
        return op(*args, **kwargs)


def _add_node(
    graph: Graph, op: torch._ops.OpOverload, args: Any, kwargs: Any, like: Node
) -> Node:
    node = graph.call_function(op, args, kwargs)
    node.meta = {k: copy.copy(like.meta[k]) for k in _PROVENANCE_KEYS if k in like.meta}
    node.meta["val"] = _fake_run(op, args, kwargs)
    return node


def _rewrite(graph: Graph, node: Node) -> Optional[str]:
    """Rewrite `node`, an in-place write through a view, or return why it is
    kept."""
    views: List[Node] = []
    base = node.args[0]
    while isinstance(base, Node) and base.target in _VIEW_TO_SCATTER:
        views.append(base)
        base = base.args[0]
    if not views:
        return "the write does not go through select, slice or diagonal"
    if base.op != "call_function" or _is_alias(base):
        return "the written tensor is an input, get_attr, view or in-place result"
    if node.target not in _INPLACE_TO_FUNCTIONAL:
        return "the op has no functional replacement here"
    # Only `base` is redirected to the new value below, so every later access
    # must go through it. A view, or the in-place op's own result, that is read
    # after the write would still see the old value.
    if node.users or any(
        user > node for alias in _aliases(base) if alias < node for user in alias.users
    ):
        return "an alias of the written tensor is read after the write"
    functional_op = _INPLACE_TO_FUNCTIONAL[node.target]
    if functional_op in _TENSOR_TO_SCALAR and not isinstance(node.args[1], Node):
        functional_op = _TENSOR_TO_SCALAR[functional_op]
    value = _fake_run(functional_op, node.args, node.kwargs)
    written = node.args[0].meta.get("val")
    if not (
        isinstance(value, torch.Tensor)
        and isinstance(written, torch.Tensor)
        and value.dtype == written.dtype
        and value.shape == written.shape
    ):
        return "the functional op changes the dtype or shape"

    with graph.inserting_before(node):
        new_value = _add_node(graph, functional_op, node.args, node.kwargs, node)
        for view in views:
            scatter = _VIEW_TO_SCATTER[view.target]
            scatter_args = (view.args[0], new_value, *view.args[1:])
            new_value = _add_node(graph, scatter, scatter_args, view.kwargs, node)
    base.replace_all_uses_with(new_value, delete_user_cb=lambda user: user > node)
    graph.erase_node(node)
    return None


class ReplaceInplaceOpsPass(PassBase):
    """Replace in-place writes through views of tensors created in the graph
    with functional ops, e.g. `buf.select(0, 0).copy_(x)` with
    `select_scatter(buf, copy(buf.select(0, 0), x), 0, 0)`.

    Meant for a quantizer's `transform_for_annotation`: an observer reads its
    input where it is inserted, so it misses a later in-place write through a
    view of that input, and the quantized model ignores whatever was written
    (pytorch/ao#5040, e.g. torchvision's `batch_images`).

    Only `copy_`, `fill_`, `zero_`, `add_`, `sub_`, `mul_` and `div_` through
    chains of `select`, `slice` and `diagonal` are rewritten. These writes are
    kept, logged at debug level, and still hit the issue above:

    - writes through other views (`view`, `permute`, `unbind`, ...);
    - writes to graph inputs, to `get_attr` parameters, buffers and constants,
      and to tensors that are themselves views or in-place results, such as
      `torch.zeros(n * m).view(n, m)` or `torch.empty(n).fill_(0)`;
    - writes after which a view of the same tensor, or the in-place op's own
      result, is read;
    - writes whose functional op would change the dtype or shape;
    - writes inside control-flow subgraphs (`cond`, `while_loop`, `scan`),
      since only the top-level graph is visited.

    Direct in-place ops such as `relu_` are left alone: later reads already use
    the in-place node, so an observer sees the written value.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or not _is_write(node):
                continue
            if not _is_alias(node.args[0]):
                continue
            reason = _rewrite(graph_module.graph, node)
            if reason is None:
                modified = True
            else:
                logger.debug("Keeping %s: %s", node.format_node(), reason)
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
