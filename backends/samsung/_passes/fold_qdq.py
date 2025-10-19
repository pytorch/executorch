# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx import GraphModule


class FoldQDQPass(ExportPass):
    def __init__(self):
        super().__init__()

    def _fold(
        self,
        graph_module: GraphModule,
    ):
        for node in graph_module.graph.nodes:
            if node.target not in (
                *QuantConstants.QUANT_OPS_KEY_MAP.keys(),
                *QuantConstants.DEQUANT_OPS_KEY_MAP.keys(),
            ):
                continue
            for user in [user for user in node.users.keys()]:  # noqa: C416
                user.replace_input_with(node, node.args[0])
            graph_module.graph.erase_node(node)

    def call(self, graph_module: GraphModule):
        self._fold(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        _ = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
