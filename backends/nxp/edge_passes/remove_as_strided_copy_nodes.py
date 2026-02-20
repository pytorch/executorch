# Copyright (c) 2025 Samsung Electronics Co. LTD
# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.edge_passes.neutron_edge_pass import NeutronEdgePass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx import GraphModule


class RemoveUselessAsStridedCopyNodes(NeutronEdgePass):
    def __init__(self):
        super().__init__()

    def gen_pattern_as_strided_copy(self, graph_module: GraphModule):
        # Unedited method taken from `backends/samsung/_passes/remove_useless_ops.py`.
        for node in list(graph_module.graph.nodes):  # noqa: C416
            if node.target != exir_ops.edge.aten.mean.dim:
                continue
            if len(node.users) != 1:
                continue
            successor = list(node.users.keys())[0]
            if successor.target != exir_ops.edge.aten.as_strided_copy.default:
                continue
            is_pattern = True
            count = 0
            for i, stride in enumerate(successor.args[2]):
                if stride < node.meta["val"].size()[i]:
                    if stride == 1:
                        count += 1
                    else:
                        is_pattern = False
                        break
                if count >= 2:
                    is_pattern = False
                    break
            if is_pattern:
                yield successor

    def _fold_as_strided_copy(
        self,
        graph_module: GraphModule,
    ) -> bool:
        # Method based on `_fold_as_strided_copy()` from `backends/samsung/_passes/remove_useless_ops.py`.
        made_changes = False
        for as_strided_copy_node in self.gen_pattern_as_strided_copy(graph_module):
            for user in list(as_strided_copy_node.users.keys()):
                user.replace_input_with(
                    as_strided_copy_node, as_strided_copy_node.args[0]
                )
            graph_module.graph.erase_node(as_strided_copy_node)

            made_changes = True

        return made_changes

    def run(self, graph_module: GraphModule):
        made_changes = self._fold_as_strided_copy(graph_module)

        graph_module.recompile()
        dead_code_elimination_pass(graph_module)

        return PassResult(graph_module, made_changes)
