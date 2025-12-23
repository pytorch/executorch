# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class Conv1dToConv2d(ExportPass):
    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def update_kernel(self, weight_node: torch.Tensor):
        # lifted tensor in tensor constant
        weight_3d = get_param_tensor(self.edge_program, weight_node)
        if param_name := self.edge_program.graph_signature.inputs_to_parameters.get(
            weight_node.name
        ):
            new_weight_param = torch.nn.Parameter(
                data=weight_3d.data.contiguous().unsqueeze(dim=-1), requires_grad=False
            )
            self.edge_program.state_dict[param_name] = new_weight_param
        elif tensor_name := self.edge_program.graph_signature.inputs_to_lifted_tensor_constants.get(
            weight_node.name
        ):
            self.edge_program.constants[tensor_name] = torch.unsqueeze(weight_3d, -1)
        else:
            RuntimeError("Weight of 1d conv should be constant tensor or Parameter obj")
        weight_node.meta["val"] = weight_node.meta["val"].data.unsqueeze(dim=-1)

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        node_list = list(graph.nodes)
        for node in node_list:
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.convolution.default:
                continue
            stride = list(node.args[3])
            if len(stride) != 1:
                continue

            # convert 3dim weight to 4dim
            weight_node = node.args[1]
            self.update_kernel(weight_node)

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
                prev_qparams = input_node.meta.get("quantize_attrs")
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
                original_users = [user for user in node.users if user != squeeze_after]
                for user in original_users:
                    user.replace_input_with(node, squeeze_after)
            if quant_attr := node.meta.get("quantize_attrs"):
                squeeze_after.meta["quantize_attrs"] = quant_attr
            if prev_qparams is not None:
                unsqueeze_before.meta["quantize_attrs"] = prev_qparams

        graph_module.recompile()
        _ = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
