# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceInfBuffer(ExportPass):
    """
    Due to limitation in Qnn, we need to change inf or -inf to arbitrary value in quantization.
    """

    def __init__(self):
        super(ReplaceInfBuffer, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        for buf_name, tensor in graph_module.named_buffers():
            if tensor.is_floating_point():
                tensor[tensor == float("inf")] = 255
                tensor[tensor == float("-inf")] = -255
                setattr(graph_module, buf_name, tensor)

        graph_module.recompile()
        return PassResult(graph_module, True)
