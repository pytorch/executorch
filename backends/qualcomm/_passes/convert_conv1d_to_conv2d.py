# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from executorch.backends.qualcomm.builders.utils import get_parameter, set_parameter
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class ConvertConv1dToConv2d(ExportPass):
    """
    Conv1d is not supported by QNN.
    Change it to input -> unsqueeze -> conv2d -> squeeze -> output
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(ConvertConv1dToConv2d, self).__init__()
        self.edge_program = edge_program

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        conv_op = exir_ops.edge.aten.convolution.default
        for node in graph.nodes:
            if node.target == conv_op and node.meta["val"].dim() == 3:

                input_node = node.args[0]
                with graph_module.graph.inserting_after(input_node):
                    unsqueeze_op = exir_ops.edge.aten.unsqueeze_copy.default
                    unsqueeze_node = graph.create_node(
                        "call_function",
                        unsqueeze_op,
                        (
                            input_node,
                            2,
                        ),
                    )
                    unsqueeze_node.meta = copy_meta(input_node.meta)
                    unsqueeze_node.meta["val"] = unsqueeze_node.meta["val"].unsqueeze(2)
                    with graph_module.graph.inserting_after(unsqueeze_node):

                        filter_node = node.args[1]
                        filter_node.meta["val"] = (
                            filter_node.meta["val"].unsqueeze(2).contiguous()
                        )
                        filter_tensor = get_parameter(filter_node, self.edge_program)
                        # Wrap with nn.Parameter. In FP mode, unsqueeze will make output not a nn.Parameter, which makes program to fail during edge_program._validate()
                        filter_tensor = nn.Parameter(filter_tensor.unsqueeze(2))
                        set_parameter(filter_tensor, filter_node, self.edge_program)

                        bias_node = node.args[2]
                        stride = [1] + node.args[3]
                        padding = [0] + node.args[4]
                        dilation = [1] + node.args[5]
                        transpose = node.args[6]
                        output_padding = [0] + node.args[7]
                        groups = node.args[8]

                        conv2d_node = graph.create_node(
                            "call_function",
                            conv_op,
                            (
                                unsqueeze_node,
                                filter_node,
                                bias_node,
                                stride,
                                padding,
                                dilation,
                                transpose,
                                output_padding,
                                groups,
                            ),
                        )
                        conv2d_node.meta = copy_meta(node.meta)
                        conv2d_node.meta["val"] = conv2d_node.meta["val"].unsqueeze(2)

                        with graph_module.graph.inserting_after(conv2d_node):
                            squeeze_op = exir_ops.edge.aten.squeeze_copy.dims
                            squeeze_node = graph.create_node(
                                "call_function",
                                squeeze_op,
                                (
                                    conv2d_node,
                                    [2],
                                ),
                            )
                            squeeze_node.meta = copy_meta(node.meta)
                for user in node.users.copy():
                    user.replace_input_with(node, squeeze_node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
