# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx.experimental.symbolic_shapes import statically_known_true


class Remove0DTensor(ExportPass):
    """
    QNN does not allow 0D tensor, so a select whose 0D output is the only element of
    its input is replaced by that input, when every user broadcasts it into an
    output of rank one or more and so gets the same result.
    Before adding operations to the list of nodes to be removed, please ensure that it will not change the logic.
    """

    remove_ops = {
        exir_ops.edge.aten.select.int,
        exir_ops.edge.aten.select_copy.int,
    }

    broadcast_ops = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
    }

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()

    def _only_broadcast(self, node: torch.fx.Node) -> bool:
        return all(
            (
                self._only_broadcast(user)
                if user.target in q_ops or user.target in dq_ops
                else user.target in self.broadcast_ops and user.meta["val"].dim() > 0
            )
            for user in node.users
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if (
                node.target in self.remove_ops
                and len(node.meta["val"].shape) == 0
                and statically_known_true(node.args[0].meta["val"].numel() == 1)
                and self._only_broadcast(node)
            ):
                for user_n in list(node.users.keys()):
                    user_n.replace_input_with(node, node.args[0])
                graph.erase_node(node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
