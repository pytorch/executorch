# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ConvertUnsqueezeToViewPass(PassBase):
    """Replace 'aten.unsqueeze.default' with 'aten.view.default'.

                  x                                               x
                  │                                               │
    ┌─────────────▼─────────────┐    replace with   ┌─────────────▼─────────────┐
    │   aten.unsqueeze(x, dim)  │  ──────────────►  │  aten.view.default(x, S)  │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  ▼                                               ▼
                 out                                             out
    """

    @staticmethod
    def _is_unsqueeze(node_: Node) -> bool:
        return (
            node_.op == "call_function"
            and node_.target == torch.ops.aten.unsqueeze.default
        )

    def _create_view_node(self, *view_args) -> Node:
        view_target = torch.ops.aten.view.default
        view_node = self.graph_module.graph.call_function(view_target, view_args)

        view_node.meta["source_fn_stack"] = [
            (view_node.name, torch.ops.aten.view.default)
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

    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        self.graph_module = graph_module
        made_changes = False

        if not any(self._is_unsqueeze(n) for n in graph_module.graph.nodes):
            return PassResult(graph_module, made_changes)

        for node in list(graph_module.graph.nodes):
            if not self._is_unsqueeze(node):
                continue

            input_node = node.all_input_nodes[0]
            target_size = node.meta["val"].shape

            with self.graph_module.graph.inserting_after(node):
                view_node = self._create_view_node(input_node, target_size)

            node.replace_all_uses_with(view_node)
            self.graph_module.graph.erase_node(node)

            made_changes = True

        self.graph_module.recompile()
        self.graph_module.graph.eliminate_dead_code()

        return PassResult(graph_module, made_changes)
