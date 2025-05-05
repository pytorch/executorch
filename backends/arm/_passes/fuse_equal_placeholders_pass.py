# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    get_constant_placeholder_kind,
    get_param_tensor,
    is_param_node,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import ExportedProgram
from executorch.exir.pass_base import ExportPass, PassResult


class FuseEqualPlaceholdersPass(ExportPass):
    """
    This pass optimizes memory usage by finding constant placeholders
    pointing to identical tensors and fusing them to one single placeholder
    with multiple users.
    """

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        const_placeholder_nodes = []
        for node in graph_module.graph.nodes:
            if is_param_node(self.exported_program, node):
                const_placeholder_nodes.append(node)

        while const_placeholder_nodes:

            # Find equal tensors
            node1 = const_placeholder_nodes.pop()
            eq_nodes = [node1]
            tensor1 = get_param_tensor(self.exported_program, node1)
            if tensor1 is None:
                continue

            for node2 in const_placeholder_nodes:
                tensor2 = get_param_tensor(self.exported_program, node2)
                if tensor2 is None:
                    continue

                if torch.equal(tensor1, tensor2):
                    eq_nodes.append(node2)

            if len(eq_nodes) > 1:
                common_name = node1.name + "_common"
                common_kind = get_constant_placeholder_kind(
                    self.exported_program, node1
                )
                common_persisten_buffer = True

                with graph_module.graph.inserting_before(node1):
                    common_node = create_constant_placeholder(
                        self.exported_program,
                        graph_module.graph,
                        common_name,
                        common_kind,
                        tensor1,
                        common_persisten_buffer,
                    )

                for eq_node in eq_nodes:
                    eq_node.replace_all_uses_with(common_node)
                    delete_constant_placeholder(self.exported_program, eq_node)
                    if eq_node != node1:
                        const_placeholder_nodes.remove(eq_node)

                modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module=graph_module, modified=modified)
