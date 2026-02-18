# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class ConvertReshapingNodesToViewPass(NeutronEdgePass):
    """Replaces:
        - 'aten.squeeze.default', 'aten.squeeze.dims' and 'aten.squeeze.dim' with 'aten.view_copy.default'.

                   x                                                  x
                   │                                                  │
    ┌──────────────▼──────────────┐    replace with   ┌───────────────▼────────────────┐
    │   aten.[un]squeeze(x, dim)  │  ──────────────►  │  aten.view_copy.default(x, S)  │
    └──────────────┬──────────────┘                   └───────────────┬────────────────┘
                   │                                                  │
                   ▼                                                  ▼
                  out                                                out

        - 'aten.unsqueeze.default' with 'aten.view_copy.default'.

                  x                                                 x
                  │                                                 │
    ┌─────────────▼─────────────┐    replace with   ┌───────────────▼────────────────┐
    │   aten.unsqueeze(x, dim)  │  ──────────────►  │  aten.view_copy.default(x, S)  │
    └─────────────┬─────────────┘                   └───────────────┬────────────────┘
                  │                                                 │
                  ▼                                                 ▼
                 out                                               out
    """

    graph_module: GraphModule

    @staticmethod
    def _is_squeeze(node_: Node) -> bool:
        return node_.op == "call_function" and (
            node_.target == exir_ops.edge.aten.squeeze_copy.dim
            or node_.target == exir_ops.edge.aten.squeeze_copy.dims
            or node_.target == exir_ops.edge.aten.squeeze_copy.default
        )

    @staticmethod
    def _is_unsqueeze(node_: Node) -> bool:
        return (
            node_.op == "call_function"
            and node_.target == exir_ops.edge.aten.unsqueeze_copy.default
        )

    def _create_view_copy_node(self, *view_args) -> Node:
        view_target = exir_ops.edge.aten.view_copy.default
        view_node = self.graph_module.graph.call_function(view_target, view_args)

        view_node.meta["source_fn_stack"] = [
            (view_node.name, exir_ops.edge.aten.view_copy.default)
        ]

        x_val = view_args[0].meta["val"]
        with FakeTensorMode() as mode:
            fake_input = FakeTensor.from_tensor(
                torch.empty(x_val.shape, dtype=x_val.dtype), mode
            )
            output_shape = view_target(fake_input, *view_args[1:]).shape
            view_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output_shape, dtype=x_val.dtype), mode
            )

        return view_node

    def run(self, graph_module: GraphModule) -> Optional[PassResult]:
        self.graph_module = graph_module

        for node in list(graph_module.graph.nodes):
            if not (self._is_squeeze(node) or self._is_unsqueeze(node)):
                continue

            input_node = node.all_input_nodes[0]
            target_shape = node.meta["val"].shape

            with self.graph_module.graph.inserting_after(node):
                view_copy_node = self._create_view_copy_node(input_node, target_shape)

            node.replace_all_uses_with(view_copy_node)
            self.graph_module.graph.erase_node(node)

            self.graph_module.graph.eliminate_dead_code()
            self.graph_module.recompile()

            # Return immediately to avoid traversing a modified graph.
            # The parent class will call this pass again.
            return PassResult(graph_module, True)

        # The graph was not modified.
        return PassResult(graph_module, False)
