# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass

# noinspection PyProtectedMember
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class RemoveMaxPoolGetItemPass(NeutronEdgePass):
    """Replace nodes in the following pattern:

                     │
    ┌────────────────▼────────────────┐
    │ max_pool2d_with_indices.default │
    └────────────────┬────────────────┘                                        │
                     │                       replace with           ┌──────────▼─────────┐
                     │                      ──────────────►         │ max_pool2d.default │
              ┌──────▼─────┐                                        └──────────┬─────────┘
              │ getitem[0] │  (extract max values only)                        ▼
              └──────┬─────┘
                     │
                     ▼

    This transformation is necessary because Neutron does not support returning the indices of the maximum values.
    """

    def run(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if not (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.max_pool2d_with_indices.default
            ):
                continue

            if len(users := list(node.users)) != 1:
                continue  # Unexpected case.

            if (getitem_node := users[0]).target != operator.getitem:
                continue  # Unexpected case.

            if getitem_node.args[1] != 0:
                # The index of the output tensor. Only `0` is supported as index `1` holds the indices from which the
                #  max values were selected, which cannot be done on Neutron.
                continue

            with graph_module.graph.inserting_before(node):
                new_max_pool_2d = graph_module.graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.max_pool2d.default,
                    args=node.args,
                    kwargs=node.kwargs,
                )

            # Attach the rest of the model to the `aten.max_pool2d.default`.
            getitem_node.replace_all_uses_with(new_max_pool_2d)

            # Remove the old nodes.
            graph_module.graph.erase_node(getitem_node)
            graph_module.graph.erase_node(node)

            # Recompile the graph.
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

            # Return now to avoid traversing a modified graph. The parent class will call this pass again if needed.
            return PassResult(graph_module, True)

        # No changes were made.
        return PassResult(graph_module, False)
