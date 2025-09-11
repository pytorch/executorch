# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import get_param


class Conv1dToConv2d(ExportPass):

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        node_list = list(graph.nodes)
        for node in node_list:
            if node.op == "call_function":
                if node.target == exir_ops.edge.aten.convolution.default:
                    stride = list(node.args[3])
                    if len(stride) != 1:
                        continue

                    # convert 3dim weight to 4dim
                    weight_node = node.args[1]
                    weight_3dim = get_param(self.edge_program, weight_node)
                    weight_4dim = torch.nn.Parameter(
                        data=weight_3dim.data.contiguous().unsqueeze(dim=-1),
                        requires_grad=False,
                    )
                    parameter_name = (
                        self.edge_program.graph_signature.inputs_to_parameters[
                            weight_node.name
                        ]
                    )
                    self.edge_program.state_dict[parameter_name] = weight_4dim
                    weight_node.meta["val"] = weight_node.meta["val"].data.unsqueeze(
                        dim=-1
                    )

                    # Extend stride, padding, and dilation
                    node.args = (
                        node.args[0],
                        node.args[1],
                        node.args[2],
                        node.args[3] + [1],  # stride
                        node.args[4] + [0],  # padding
                        node.args[5] + [1],  # dilation
                        node.args[6],
                        node.args[7],
                        node.args[8],
                    )

                    # unsqueeze -> conv2d -> squeeze
                    with graph.inserting_before(node):
                        input_node = node.args[0]
                        unsqueeze_before = graph.create_node(
                            "call_function", exir_ops.edge.aten.unsqueeze_copy.default
                        )
                        unsqueeze_before.args = (
                            input_node,
                            -1,
                        )
                        node.replace_input_with(input_node, unsqueeze_before)

                    with graph.inserting_after(node):
                        squeeze_after = graph.create_node(
                            "call_function", exir_ops.edge.aten.squeeze_copy.dims
                        )
                        squeeze_after.args = (
                            node,
                            [-1],
                        )
                        original_users = [
                            user for user in node.users if user != squeeze_after
                        ]
                        for user in original_users:
                            user.replace_input_with(node, squeeze_after)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
