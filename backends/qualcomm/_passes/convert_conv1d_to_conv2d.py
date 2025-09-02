# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter, set_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_REQUANTIZE
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import append_qdq, copy_meta


class ConvertConv1dToConv2d(ExportPass):
    """
    Conv1d is not supported by QNN.
    Change it to input -> unsqueeze -> conv2d -> squeeze -> output
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(ConvertConv1dToConv2d, self).__init__()
        self.edge_program = edge_program
        self.conv_op_map = {
            torch.ops.aten.conv1d.default: torch.ops.aten.conv2d.default,
            torch.ops.aten.conv_transpose1d.default: torch.ops.aten.conv_transpose2d.input,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target in self.conv_op_map:
                input_node = node.args[0]
                with graph_module.graph.inserting_after(input_node):
                    unsqueeze_op = torch.ops.aten.unsqueeze_copy.default
                    unsqueeze_node = graph.create_node(
                        "call_function",
                        unsqueeze_op,
                        (
                            input_node,
                            2,
                        ),
                    )
                    unsqueeze_node.meta = copy_meta(
                        input_node.meta, lambda m: {**m, "val": m["val"].unsqueeze(2)}
                    )
                    qdq_node_after_unsqueeze = append_qdq(
                        graph_module=graph_module,
                        node=unsqueeze_node,
                        qdq_node=input_node,
                    )

                    with graph_module.graph.inserting_after(qdq_node_after_unsqueeze):
                        filter_arg = node.args[1]
                        filter_node = (
                            filter_arg
                            if filter_arg.op == "placeholder"
                            else node.args[1].args[0]
                        )
                        filter_node.meta["val"] = (
                            filter_node.meta["val"].unsqueeze(2).contiguous()
                        )
                        filter_tensor = get_parameter(
                            filter_node, self.edge_program
                        ).unsqueeze(2)
                        set_parameter(
                            (
                                torch.nn.Parameter(filter_tensor)
                                if filter_tensor.dtype == torch.float
                                else filter_tensor
                            ),
                            filter_node,
                            self.edge_program,
                        )

                        num_args = len(node.args)

                        bias_node = node.args[2] if num_args > 2 else None
                        stride = [1] + node.args[3] if num_args > 3 else [1, 1]
                        padding = [0] + node.args[4] if num_args > 4 else [0, 0]
                        if node.target == torch.ops.aten.conv1d.default:
                            dilation = [1] + node.args[5] if num_args > 5 else [1, 1]
                            groups = node.args[6] if num_args > 6 else 1
                            conv_args = (
                                qdq_node_after_unsqueeze,
                                node.args[1],
                                bias_node,
                                stride,
                                padding,
                                dilation,
                                groups,
                            )
                        else:
                            output_padding = (
                                [0] + node.args[5] if num_args > 5 else [0, 0]
                            )
                            groups = node.args[6] if num_args > 6 else 1
                            dilation = [1] + node.args[7] if num_args > 7 else [1, 1]
                            conv_args = (
                                qdq_node_after_unsqueeze,
                                node.args[1],
                                bias_node,
                                stride,
                                padding,
                                output_padding,
                                groups,
                                dilation,
                            )
                        conv2d_node = graph.create_node(
                            "call_function",
                            self.conv_op_map[node.target],
                            conv_args,
                        )
                        conv2d_node.meta = copy_meta(
                            node.meta, lambda m: {**m, "val": m["val"].unsqueeze(2)}
                        )
                        qdq_node_after_conv2d = append_qdq(
                            graph_module=graph_module,
                            node=conv2d_node,
                            qdq_node=list(node.users)[0],
                        )

                        with graph_module.graph.inserting_after(qdq_node_after_conv2d):
                            squeeze_op = torch.ops.aten.squeeze_copy.dims
                            squeeze_node = graph.create_node(
                                "call_function",
                                squeeze_op,
                                (
                                    qdq_node_after_conv2d,
                                    [2],
                                ),
                            )
                            squeeze_node.meta = copy_meta(node.meta)

                            if QCOM_REQUANTIZE in input_node.meta:
                                input_node.meta.pop(QCOM_REQUANTIZE)
                            if QCOM_REQUANTIZE in node.meta:
                                squeeze_node.meta[QCOM_REQUANTIZE] = node.meta[
                                    QCOM_REQUANTIZE
                                ]
                                conv2d_node.meta.pop(QCOM_REQUANTIZE, None)

                for user in node.users.copy():
                    user.replace_input_with(node, squeeze_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
