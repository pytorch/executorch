# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class ReplaceArangeArgs(ExportPass):
    """
    During annotation, kwargs for arange will be removed due to restrictions by quantizer.
    This causes arange having no dtype, which means FP nodes might become an INT node during calibration.
    This can cause calibration to fail since QDQ can only be applied on FP nodes but not INT nodes.
    To hint the dtype, we provide step size as 1.0 instead of 1, which makes the node a fp node.
    """

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()
        self.quantization_capture = quantization_capture

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.arange.default:
                if torch.is_floating_point(node.meta["val"]) and len(node.args) == 1:
                    with graph_module.graph.inserting_after(node):
                        step_arange_op = torch.torch.ops.aten.arange.start_step
                        step_arange_node = graph.create_node(
                            "call_function",
                            step_arange_op,
                            (
                                0,
                                node.args[0],
                                1.0,
                            ),
                        )
                        step_arange_node.meta = copy_meta(node.meta)

                        for user in node.users.copy():
                            user.replace_input_with(node, step_arange_node)
                        graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
