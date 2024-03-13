# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class DebugHandleGeneratorPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        """Lower a quantized reference model (with reference quantized operator patterns)
        to executorch backend, that has a canonical set of quantized operators
        """

        queue = [graph_module]
        index = 0
        # bfs to traverse all modules including control flow submodules to attached debug handle id
        while queue:
            current_graph_module = queue.pop(0)
            for node in current_graph_module.graph.nodes:
                node.meta["debug_handle"] = index
                index += 1
            control_flow_submodules = [
                submodule
                for _, submodule, _ in get_control_flow_submodules(current_graph_module)
            ]
            queue.extend(control_flow_submodules)
        return PassResult(graph_module, True)
