# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewriteFp16SiluPass(ExportPass):
    """Rewrite preserved FP16 SiLU into FP16 sigmoid and multiply."""

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.target != exir_ops.edge.aten.silu.default:
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue

            with graph.inserting_before(node):
                sigmoid = graph.call_function(
                    exir_ops.edge.aten.sigmoid.default, (input_node,)
                )
                mul = graph.call_function(
                    exir_ops.edge.aten.mul.Tensor, (input_node, sigmoid)
                )
            node.replace_all_uses_with(mul)
            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
