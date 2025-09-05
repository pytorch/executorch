# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FixedLinearKeepDim(ExportPass):
    """
    Add squeeze and unsqueeze around linear node since QNN has no keep dims for linear op.
    """

    view_copy = exir_ops.edge.aten.view_copy.default
    linear = exir_ops.edge.aten.linear.default

    def __init__(self):
        super(FixedLinearKeepDim, self).__init__()

    def _fixed_keep_dim(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.target != self.linear:
                continue

            linear_node = node
            input_node = linear_node.args[0]
            # Since QNN has no keep dims for linear op, we will need to add squeeze and unsqueeze around linear node
            # TODO: Find a more general conditional statement.
            linear_output = linear_node.meta["val"]
            if linear_output.dim() >= 3:
                with graph_module.graph.inserting_after(input_node):
                    input_users = list(input_node.users.keys())
                    input_tensor = input_node.meta["val"]
                    squeeze_dim = (-1, input_tensor.shape[-1])
                    squeeze_node = graph_module.graph.create_node(
                        "call_function",
                        self.view_copy,
                        (
                            input_node,
                            squeeze_dim,
                        ),
                    )
                    # meta needs to be copied elementwisely for fake-tensor
                    # to be updated correctly and not affect meta of input_node
                    for k, v in input_node.meta.items():
                        squeeze_node.meta[k] = v
                    squeeze_node.meta["val"] = input_tensor.reshape(squeeze_dim)
                    for user in input_users:
                        if user == linear_node:
                            user.replace_input_with(input_node, squeeze_node)

                with graph_module.graph.inserting_after(linear_node):
                    output_users = list(linear_node.users.keys())
                    unsqueeze_dim = linear_output.shape
                    unsqueeze_node = graph_module.graph.create_node(
                        "call_function",
                        self.view_copy,
                        (
                            linear_node,
                            unsqueeze_dim,
                        ),
                    )
                    # meta needs to be copied elementwisely for fake-tensor
                    # to be updated correctly and not affect meta of unsqueeze_node
                    for k, v in linear_node.meta.items():
                        unsqueeze_node.meta[k] = v
                    # update linear node's shape
                    linear_node.meta["val"] = linear_output.reshape(
                        (squeeze_node.meta["val"].shape[0], linear_output.shape[-1])
                    )
                    for user in output_users:
                        user.replace_input_with(linear_node, unsqueeze_node)

    def call(self, graph_module: torch.fx.GraphModule):
        self._fixed_keep_dim(graph_module)
        dead_code_elimination_pass(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
