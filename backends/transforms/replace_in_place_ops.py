# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

# Views that an in-place write can go through, mapped to the op that writes the
# result back into the viewed tensor.
_VIEW_TO_SCATTER = {
    torch.ops.aten.select.int: torch.ops.aten.select_scatter.default,
    torch.ops.aten.slice.Tensor: torch.ops.aten.slice_scatter.default,
    torch.ops.aten.diagonal.default: torch.ops.aten.diagonal_scatter.default,
    torch.ops.aten.as_strided.default: torch.ops.aten.as_strided_scatter.default,
}


def _get_functional_op(op: torch._ops.OpOverload) -> Optional[torch._ops.OpOverload]:
    name = op._schema.name.split("::")[-1]
    if not name.endswith("_"):
        return None
    packet = getattr(getattr(torch.ops, op.namespace), name[:-1], None)
    if packet is None or op._overloadname not in packet.overloads():
        return None
    return getattr(packet, op._overloadname)


def _get_arg0_alias_info(node: Node) -> Optional[Any]:
    if not isinstance(node.target, torch._ops.OpOverload) or not node.args:
        return None
    return node.target._schema.arguments[0].alias_info


def replace_in_place_ops(graph_module: GraphModule) -> GraphModule:
    """Replace in-place ops on tensors created in the graph with their functional
    variants, e.g. `buf.select(0, 0).copy_(x)` with
    `select_scatter(buf, copy(buf.select(0, 0), x), 0, 0)`.

    Meant for a quantizer's `transform_for_annotation`: an observer reads its
    input where it is inserted, so it misses a later in-place write to that
    input, and the quantized model ignores whatever was written (for example
    torchvision's `batch_images`, which fills `new_full` with `copy_`). Writes
    to graph inputs and buffers are kept, as they have to stay visible outside
    the graph.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        alias_info = _get_arg0_alias_info(node)
        if alias_info is None or not alias_info.is_write:
            continue
        if not isinstance(node.args[0], Node):
            continue
        functional_op = _get_functional_op(node.target)
        views: List[Node] = []
        base = node.args[0]
        while base.target in _VIEW_TO_SCATTER:
            views.append(base)
            base = base.args[0]
        if (
            functional_op is None
            or not isinstance(base.target, torch._ops.OpOverload)
            or any(r.alias_info for r in base.target._schema.returns)
        ):
            continue
        # Aliases of `base` created before the write and used after it would
        # need to be recomputed from the new value; leave those writes in place.
        aliases = []
        to_visit = [base]
        while to_visit:
            n = to_visit.pop()
            for user in n.users:
                if _get_arg0_alias_info(user) is not None and user.args[0] is n:
                    aliases.append(user)
                    to_visit.append(user)
        if any(
            user > node
            for alias in aliases
            if alias < node and alias not in views
            for user in alias.users
        ):
            continue

        with graph.inserting_before(node):
            new_value = graph.call_function(functional_op, node.args, node.kwargs)
            new_value.meta = dict(node.meta)
            replacements: List[Tuple[Node, Node]] = [
                (node, new_value),
                (node.args[0], new_value),
            ]
            for view in views:
                parent = view.args[0]
                new_value = graph.call_function(
                    _VIEW_TO_SCATTER[view.target],
                    (parent, new_value, *view.args[1:]),
                    view.kwargs,
                )
                new_value.meta = {**node.meta, "val": parent.meta.get("val")}
                replacements.append((parent, new_value))
        for old, new in replacements:
            old.replace_all_uses_with(
                new, delete_user_cb=lambda user, write=node: user > write
            )
        graph.erase_node(node)
    graph_module.recompile()
    return graph_module
