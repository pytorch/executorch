# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FuseConsecutiveCast(ExportPass):
    """
    This pass fuses consecutive cast into one or none to reduce runtime
    overhead.
    To simplify the fuse logic, we ensure each cast node's output has at most 1 cast node
    by cloning cast.
    Example:
    Before clone cast:
    relu -> cast1 ─> cast2
            |──────> cast3

    After clone cast:
    relu ─> cast1 ──────> cast2
      |───> cast4(new) ─> cast3
    """

    def __init__(self):
        super().__init__()
        self.op_map = {
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
            exir_ops.edge.aten._to_copy.default,
        }
        self.visited = set()
        self.nodes = []

    def _canonicalize_cast(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # replace all i64 cast nodes with i32 version
        graph = graph_module.graph
        for n in graph_module.graph.nodes:
            if n.target in self.op_map and n.meta["val"].dtype == torch.int64:
                users = list(n.users)
                for user in users:
                    # bypass graph output node to meet original convention
                    if user.op == "output":
                        continue

                    with graph.inserting_after(n):
                        cast_node = graph.create_node(
                            "call_function",
                            exir_ops.edge.aten._to_copy.default,
                            n.args,
                            kwargs={"dtype": torch.int32},
                        )
                        cast_node.meta = n.meta
                        cast_node.meta["val"] = cast_node.meta["val"].to(torch.int32)
                        user.replace_input_with(n, cast_node)

        graph.eliminate_dead_code()

        # clone nodes for future fusion
        for n in graph_module.graph.nodes:
            # make sure we're handling cast node instead of convert node
            if n.target in self.op_map and n.kwargs.get("dtype", None) is not None:
                users = [user for user in list(n.users) if user.target in self.op_map]
                if len(users) > 1:
                    for i in range(1, len(users)):
                        with graph.inserting_after(n):
                            clone_cast_node = graph.create_node(
                                "call_function",
                                exir_ops.edge.aten._to_copy.default,
                                n.args,
                                kwargs=n.kwargs,
                            )
                            clone_cast_node.meta = n.meta
                            users[i].replace_input_with(n, clone_cast_node)

    def _traverse(self, node):
        if node in self.visited or node.target not in self.op_map:
            return

        self.nodes.append(node)
        self.visited.add(node)
        next_users = [n for n in list(node.users) if n.target in self.op_map]

        assert (
            len(next_users) <= 1
        ), "Each cast node should have at most 1 cast output node after _clone_cast"
        if not next_users:
            return
        else:
            self._traverse(list(node.users)[0])

    def _fuse(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            self._traverse(n)
            # TODO: how to handle following scenario (won't happen for quantized graph)
            #       fp -> to(i32) -> to(fp)
            if len(self.nodes) > 1:
                input_node, output_node = self.nodes[0], self.nodes[-1]
                output_node.replace_input_with(output_node.args[0], input_node.args[0])

            # clear current stack
            self.nodes = []

    def call(self, graph_module: torch.fx.GraphModule):
        self._canonicalize_cast(graph_module)
        self._fuse(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
