# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.vulkan.utils as utils
import torch

from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FoldQDQPass(ExportPass):
    """
    Erase Q/DQ chain introduced by PT2E quantization workflow. It is assumed that all
    valid quant op patterns have already been fused before this pass.
    """

    def __init__(self):
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if utils.is_quant_node(node):
                original_node = node.args[0]
                assert isinstance(original_node, torch.fx.Node)
                # For each direct user that is a dequant node, connect the original
                # node to the users of the dequant node.
                for user in node.users:
                    if utils.is_dequant_node(user):
                        dq_node = user
                        dq_node.replace_all_uses_with(original_node)

        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        # Re-trace to validate everything is ok
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
