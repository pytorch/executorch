# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceInfValues(ExportPass):
    """
    Due to limitation in QNN, change inf or -inf to arbitrary value in quantization.
    """

    def __init__(self):
        super(ReplaceInfValues, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):  # noqa: C901
        for buf_name, tensor in graph_module.named_buffers():
            if tensor.is_floating_point():
                # 255 here is mainly for attention_mask in Llama for reasonable quant scale
                tensor[tensor == float("inf")] = 255
                tensor[tensor == float("-inf")] = -255
                setattr(graph_module, buf_name, tensor)

        for node in graph_module.graph.nodes:
            arg_list = list(node.args)
            for index, arg in enumerate(arg_list):
                if arg == float("-inf"):
                    arg_list[index] = torch.finfo(torch.float32).min
                elif arg == float("inf"):
                    arg_list[index] = torch.finfo(torch.float32).max

            if node.target == torch.ops.aten.masked_fill.Scalar:
                if arg_list[2] == torch.finfo(torch.float32).max:
                    arg_list[2] = 255
                elif arg_list[2] == torch.finfo(torch.float32).min:
                    arg_list[2] = -255
            elif node.target == torch.ops.aten.scalar_tensor.default:
                if arg_list[0] == torch.finfo(torch.float32).max:
                    arg_list[0] = 255
                elif arg_list[0] == torch.finfo(torch.float32).min:
                    arg_list[0] = -255

            node.args = tuple(arg_list)

            if node.target in [
                torch.ops.aten.masked_fill.Tensor,
                torch.ops.aten.masked_fill.Scalar,
            ]:
                assert (
                    len(node.args) == 3
                ), f"Expecting {node.name} to have 3 arguments."
                val = node.args[2]
                if node.args[2] > torch.finfo(torch.float16).max:
                    val = 255
                elif node.args[2] < torch.finfo(torch.float16).min:
                    val = -255
                node.args = (
                    node.args[0],
                    node.args[1],
                    val,
                )

        graph_module.recompile()
        return PassResult(graph_module, True)
