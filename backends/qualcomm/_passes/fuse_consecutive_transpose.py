# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.qualcomm.utils.constants import QCOM_INSERTED_PERMUTE

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FuseConsecutiveTranspose(ExportPass):
    """
    This pass fuses consecutive transpose / permute into one or none to reduce runtime
    overhead.
    To simplify the fuse logic, we ensure each permute node's output has at most 1 permute node
    by cloning transpose.
    Example:
    Before clone transpose:
    relu -> permute1 ─> permute2
               |──────> permute3

    After clone transpose:
    relu ─> permute1 ──────> permute2
      |───> permute4(new) ─> permute3
    """

    def __init__(self):
        super().__init__()
        self.op_map = {
            exir_ops.edge.aten.permute_copy.default,
        }
        self.visited = set()
        self.nodes = []

    def _clone_transpose(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        graph = graph_module.graph
        for n in graph_module.graph.nodes:
            if n.target in self.op_map:
                users = [user for user in list(n.users) if user.target in self.op_map]
                if len(users) > 1:
                    for i in range(1, len(users)):
                        with graph.inserting_after(n):
                            clone_permute_node = graph.create_node(
                                "call_function",
                                exir_ops.edge.aten.permute_copy.default,
                                (n.args[0], n.args[1]),
                            )
                            clone_permute_node.meta = n.meta
                            users[i].replace_input_with(n, clone_permute_node)

    def _is_dispensable(self, axis_order):
        for index, value in enumerate(axis_order):
            if index != value:
                return False
        return True

    def _traverse(self, node):
        if node in self.visited or node.target not in self.op_map:
            return

        self.nodes.append(node)
        self.visited.add(node)
        next_users = [n for n in list(node.users) if n.target in self.op_map]

        assert (
            len(next_users) <= 1
        ), "Each permute node should have at most 1 permute output node after _clone_transpose"
        if not next_users:
            return
        else:
            self._traverse(list(node.users)[0])

    def _fuse(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph = graph_module.graph
        for n in graph_module.graph.nodes:
            self._traverse(n)
            if len(self.nodes) > 1:
                input_node, output_node = self.nodes[0].args[0], self.nodes[-1]
                input_shape = input_node.meta["val"].shape
                axis_order = torch.arange(len(input_shape)).tolist()
                for node in self.nodes:
                    axis_order = [axis_order[i] for i in node.args[1]]
                # If axis order is just [0,1,2,3], we ignore permute node
                if self._is_dispensable(axis_order):
                    for user in output_node.users.copy():
                        user.replace_input_with(output_node, n.args[0])
                else:
                    with graph.inserting_after(input_node):
                        permute_op = exir_ops.edge.aten.permute_copy.default
                        permute_node = graph.create_node(
                            "call_function", permute_op, (input_node, axis_order)
                        )
                        users = output_node.users.copy()
                        for user in users:
                            user.replace_input_with(output_node, permute_node)

                        # copy metadata
                        permute_node.meta = output_node.meta
                        # Without "qnn_permute", we might obtain wrong input shape
                        if [pn.meta.get(QCOM_INSERTED_PERMUTE) for pn in self.nodes]:
                            permute_node.meta[QCOM_INSERTED_PERMUTE] = True

            # clear current stack
            self.nodes = []

    def call(self, graph_module: torch.fx.GraphModule):
        self._clone_transpose(graph_module)
        self._fuse(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
