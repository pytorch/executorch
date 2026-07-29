# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_manager import PassResult


def _contiguous_dim_order(ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim))


def _is_contiguous(dim_order: Sequence[int]) -> bool:
    return tuple(dim_order) == _contiguous_dim_order(len(dim_order))


def _node_is_boundary_clone(node: torch.fx.Node) -> bool:
    """Return True if `node` is a boundary clone inserted by this pass.

    A boundary clone is a `_clone_dim_order` node that consumes a model input.
    """
    if (
        node.target != exir_ops.edge.dim_order_ops._clone_dim_order.default
    ):  # noqa: F405
        return False
    if not node.args:
        return False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node) or src.op != "placeholder":
        return False
    val = src.meta.get("val", None)
    return isinstance(val, torch.Tensor) and not val.is_contiguous()


# Edge-dialect ops whose job is to copy/clone a tensor into a (possibly different) dim order.  After this pass,
#  everything is contiguous, so these are identity ops and can be replaced with their first argument.
_DIM_ORDER_CHANGING_OPS: frozenset = frozenset(
    {
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }
)

# Edge-dialect allocation ops that carry a `dim_order` keyword argument. The argument must be rewritten to the
#  contiguous order so that newly allocated tensors match the rest of the graph.
_ALLOC_DIM_ORDER_OPS: frozenset = frozenset(
    {
        exir_ops.edge.dim_order_ops._empty_dim_order.default,
    }
)


class EnforceContiguousDimOrder(ExportPass):
    """
    Edge-dialect pass that enforces contiguous dim order throughout the graph.

    What the pass does:
    1. Removes every `dim_order_ops._clone_dim_order` and `dim_order_ops._to_dim_order_copy` node.
    2. Inserts `dim_order_ops._clone_dim_order` nodes immediately after each placeholder (model input) whose dim order
        is not contiguous (e.g. channels last). This for example handles models that were exported with channels-last
        example inputs.
    3. Rewrites the `dim_order` keyword argument of `dim_order_ops._empty_dim_order` to the contiguous order, so that
        newly allocated tensors also get the right layout.
    4. Calls `super().call()` to let `ExportPass` re-propagate all node metadata from scratch after structural graph
        changes.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        graph = graph_module.graph

        # Step 1 — Remove existing `_clone_dim_order` / `_to_dim_order_copy` nodes.
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in _DIM_ORDER_CHANGING_OPS:
                continue
            if _node_is_boundary_clone(node):
                continue  # Preserve boundary clones for idempotency.

            src = node.args[0]
            node.replace_all_uses_with(src)
            graph.erase_node(node)
            modified = True

        # Step 2 — Insert `_clone_dim_order` after non-contiguous placeholders.
        # If the model was exported with channels last (or other non-contiguous) example inputs, the placeholder nodes
        #  carry a non-contiguous meta['val']. Without this step, all downstream ops would receive a non-contiguous
        #  tensor at runtime even though we removed all internal layout conversions in Step 1.
        # For each such placeholder we insert a `_clone_dim_order` that converts the runtime input to the contiguous dim
        #  order. (this operator can later be replaced with the equivalent `permute_copy ->  channels_last.permute_copy`
        #  sequence, for the purposes of https://github.com/pytorch/executorch/issues/19299)
        for node in list(graph.nodes):
            if node.op != "placeholder":
                continue
            if not isinstance(val := node.meta.get("val", None), torch.Tensor):
                continue
            if val.is_contiguous():
                continue
            if len(node.users) == 1 and _node_is_boundary_clone(next(iter(node.users))):
                # The clone is already in place.
                continue

            # Found a model input with non-contiguous layout and no following clone.

            contiguous_dim_order = list(_contiguous_dim_order(val.dim()))

            with graph.inserting_after(node):
                clone_node = graph.call_function(
                    exir_ops.edge.dim_order_ops._clone_dim_order.default,  # noqa: F405
                    args=(node,),
                    kwargs={"dim_order": contiguous_dim_order},
                )
                clone_node.meta["val"] = val.contiguous()

            # Redirect all downstream consumers to the contiguous clone. This also replaces `clone_node.args[0]`.
            #  Restore it immediately to break the cycle.
            node.replace_all_uses_with(clone_node)
            clone_node.update_arg(0, node)

            modified = True

        # Step 3 — Fix dim_order kwargs on allocation ops.
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in _ALLOC_DIM_ORDER_OPS:
                continue
            if "dim_order" not in node.kwargs:
                continue

            dim_order = node.kwargs["dim_order"]
            if not _is_contiguous(dim_order):
                new_kwargs = dict(node.kwargs)
                new_kwargs["dim_order"] = list(_contiguous_dim_order(len(dim_order)))
                node.kwargs = new_kwargs
                modified = True

        # Step 4 - Re-compute the metadata.
        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            # Let ExportPass re-propagate metadata through op semantics first.
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
