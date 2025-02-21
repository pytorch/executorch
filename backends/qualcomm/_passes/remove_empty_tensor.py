# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemoveEmptyTensor(ExportPass):
    """
    QNN does not allow 0D tensor, we remove the node that will output an empty tensor.
    Before adding operations to the list of nodes to be removed, please ensure that it will not change the logic.
    """

    remove_ops = {
        exir_ops.edge.aten.select.int,
    }

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target in self.remove_ops and len(node.meta["val"].shape) == 0:
                for user_n in list(node.users.keys()):
                    user_n.replace_input_with(node, node.args[0])
                graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
