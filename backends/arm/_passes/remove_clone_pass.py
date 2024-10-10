# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemoveClonePass(ExportPass):

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == exir_ops.edge.aten.clone.default:
                for user in list(node.users):
                    # TODO remove dq/q-ops around removed clone-op
                    user.replace_input_with(node, node.args[0])
                graph_module.graph.erase_node(node)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
