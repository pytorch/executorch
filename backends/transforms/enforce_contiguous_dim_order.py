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


def _node_is_input_boundary_clone(node: torch.fx.Node) -> bool:
    """Return True if `node` is an input boundary clone.

    An input boundary clone is a `_clone_dim_order` node whose source is a non-contiguous
    placeholder and whose own `dim_order` kwarg is contiguous. It converts the runtime input
    to contiguous layout so that the rest of the graph operates on contiguous tensors.
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
    if not (isinstance(val, torch.Tensor) and not val.is_contiguous()):
        return False
    dim_order = node.kwargs.get("dim_order")
    return dim_order is not None and _is_contiguous(dim_order)


def _node_is_output_boundary_clone(node: torch.fx.Node) -> bool:
    """Return True if `node` is an output boundary clone.

    An output boundary clone is a `_clone_dim_order` node with a non-contiguous `dim_order`
    kwarg, fed by a contiguous source, that feeds directly into the graph output node. It
    restores the original non-contiguous output dim order that callers expect, converting the
    contiguous internal result back to the required layout at the graph boundary.
    """
    if (
        node.target != exir_ops.edge.dim_order_ops._clone_dim_order.default
    ):  # noqa: F405
        return False
    dim_order = node.kwargs.get("dim_order")
    if dim_order is None or _is_contiguous(dim_order):
        return False
    if not node.users or any(user.op != "output" for user in node.users):
        return False
    if not node.args:
        return False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node):
        return False
    val = src.meta.get("val", None)
    return isinstance(val, torch.Tensor) and val.is_contiguous()


# Edge-dialect ops whose job is to copy/clone a tensor into a (possibly different) dim order.  After this pass,
#  everything is contiguous internally, so these are identity ops and can be replaced with their first argument.
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
    Edge-dialect pass that enforces contiguous dim order throughout the graph while
    preserving the dim order of the model's inputs and outputs.

    What the pass does:
    1. Removes every `dim_order_ops._clone_dim_order` and `dim_order_ops._to_dim_order_copy`
       node that is not an IO boundary clone (see steps 2 and 3 below).
    2. Inserts `dim_order_ops._clone_dim_order` nodes immediately after each placeholder
       (model input) whose dim order is not contiguous (e.g. channels last). This converts
       the runtime input to contiguous layout for all downstream ops.
    3. Inserts `dim_order_ops._clone_dim_order` nodes immediately before the output node for
       each output whose original dim order was not contiguous, restoring the output dim order
       that callers expect.
    4. Rewrites the `dim_order` keyword argument of `dim_order_ops._empty_dim_order` to the
       contiguous order so that newly allocated tensors also get the right layout.
    5. Calls `super().call()` to let `ExportPass` re-propagate all node metadata from scratch
       after structural graph changes.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        graph = graph_module.graph

        # Pre-scan: record the original non-contiguous output positions and their vals before
        # any structural changes. Step 1 below may remove internal _clone_dim_order nodes that
        # are direct predecessors of the output, which would otherwise make it impossible to
        # detect which outputs were originally non-contiguous.
        output_node = next((n for n in graph.nodes if n.op == "output"), None)
        # Map from output index to a (dim_order_snapshot, original_val) pair. The dim_order
        # is captured as an immutable tuple NOW so that Step 4 does not depend on step
        # ordering or on whether the FakeTensor objects are mutated later (e.g. by metadata
        # re-propagation in super().call()).
        output_restore: dict[int, tuple[tuple[int, ...], torch.Tensor]] = {}
        if output_node is not None:
            return_values = output_node.args[0]
            if isinstance(return_values, (list, tuple)):
                for i, arg in enumerate(return_values):
                    if isinstance(
                        arg, torch.fx.Node
                    ) and not _node_is_output_boundary_clone(arg):
                        val = arg.meta.get("val")
                        if isinstance(val, torch.Tensor) and not val.is_contiguous():
                            output_restore[i] = (tuple(val.dim_order()), val)

        # Steps 1-3 in a single traversal over a snapshot of the graph nodes.
        for node in list(graph.nodes):
            if node.op == "call_function":
                if node.target in _DIM_ORDER_CHANGING_OPS:
                    # Preserve boundary clones; they perform the required IO dim-order
                    # conversion and must not be removed.
                    if _node_is_input_boundary_clone(
                        node
                    ) or _node_is_output_boundary_clone(node):
                        continue
                    src = node.args[0]
                    node.replace_all_uses_with(src)
                    graph.erase_node(node)
                    modified = True
                elif node.target in _ALLOC_DIM_ORDER_OPS:
                    if "dim_order" not in node.kwargs:
                        continue
                    dim_order = node.kwargs["dim_order"]
                    if not _is_contiguous(dim_order):
                        new_kwargs = dict(node.kwargs)
                        new_kwargs["dim_order"] = list(
                            _contiguous_dim_order(len(dim_order))
                        )
                        node.kwargs = new_kwargs
                        modified = True
            elif node.op == "placeholder":
                val = node.meta.get("val")
                if not isinstance(val, torch.Tensor) or val.is_contiguous():
                    continue
                if len(node.users) == 1 and _node_is_input_boundary_clone(
                    next(iter(node.users))
                ):
                    continue  # Input boundary clone already present.

                # Found a model input with non-contiguous layout and no following clone.
                contiguous_dim_order = list(_contiguous_dim_order(val.dim()))
                with graph.inserting_after(node):
                    clone_node = graph.call_function(
                        exir_ops.edge.dim_order_ops._clone_dim_order.default,  # noqa: F405
                        args=(node,),
                        kwargs={"dim_order": contiguous_dim_order},
                    )
                    clone_node.meta["val"] = val.contiguous()

                # Redirect all downstream consumers to the contiguous clone. This also replaces
                #  `clone_node.args[0]`. Restore it immediately to break the cycle.
                node.replace_all_uses_with(clone_node)
                clone_node.update_arg(0, node)
                modified = True

        # Step 4: Insert output boundary clones to restore non-contiguous output dim orders.
        # This is the output-side counterpart of the input boundary clone insertion above:
        # the contiguous internal result is converted back to the original output dim order
        # that callers expect.
        if output_node is not None and output_restore:
            current_return_values = list(output_node.args[0])
            for i, (dim_order_snapshot, original_val) in output_restore.items():
                src_node = current_return_values[i]
                # After Step 1 removals, src_node may already be a valid output boundary
                # clone (e.g. the second of two chained non-contiguous clones whose first
                # was removed, leaving the second with a contiguous source). Skip insertion
                # in that case to avoid a duplicate output boundary clone.
                if _node_is_output_boundary_clone(src_node):
                    continue
                with graph.inserting_before(output_node):
                    clone_node = graph.call_function(
                        exir_ops.edge.dim_order_ops._clone_dim_order.default,  # noqa: F405
                        args=(src_node,),
                        kwargs={"dim_order": list(dim_order_snapshot)},
                    )
                    clone_node.meta["val"] = original_val
                current_return_values[i] = clone_node
                modified = True
            output_node.args = (tuple(current_return_values),)

        # Step 5: Re-compute the metadata.
        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
