# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertELUParamsPass(ExportPass):
    """
    Pass to convert the input_scale kwarg of ELU operator from float to
    int.

    It has been set to 2 as the outputs seem to stay the same regardless of what
    the value of input_scale is, as long as that value is not 1.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        modified_graph = False
        graph = graph_module.graph
        node_list = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.elu.default
        )
        for node in node_list:
            with graph.inserting_after(node):
                replace_node = create_node(graph, exir_ops.edge.aten.elu.default)
                replace_node.args = (
                    node.args[0],
                    int(node.args[1]) if len(node.args) > 1 else 1,
                )
                updated_kwargs = dict(node.kwargs)
                updated_kwargs["input_scale"] = int(2)
                replace_node.kwargs = updated_kwargs

                node.replace_all_uses_with(replace_node)
                graph.erase_node(node)

                modified_graph = True
        if modified_graph:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified_graph)
