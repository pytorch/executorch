# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceInfValues(ExportPass):
    """
    Due to limitation in Qnn, we need to change inf or -inf to arbitrary value in quantization.
    This could be a buffer or a node's argument.
    """

    def __init__(self):
        super(ReplaceInfValues, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
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
            node.args = tuple(arg_list)

        graph_module.recompile()
        return PassResult(graph_module, True)
