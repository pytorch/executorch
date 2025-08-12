# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_BYPASS_NODE
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FoldQDQ(ExportPass):
    """
    Erase QDQ pattern.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram, force_fold=False):
        super(FoldQDQ, self).__init__()
        self.edge_program = edge_program
        self.force_fold = force_fold

    def _annotate_bypass(self, node):
        node.meta[QCOM_BYPASS_NODE] = True
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.op == "call_function":
                self._annotate_bypass(arg)

    def _fold_dq(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # remove dq
        for n in graph_module.graph.nodes:
            user_list = list(n.users.keys())
            if n.target not in dq_ops:
                continue

            # skip parameters & buffers
            if not self.force_fold and is_parameter(n.args[0], self.edge_program):
                self._annotate_bypass(n)
            else:
                for user_n in user_list:
                    user_n.replace_input_with(n, n.args[0])
                graph_module.graph.erase_node(n)

    def _fold_q(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # remove q
        for n in graph_module.graph.nodes:
            if n.target not in q_ops:
                continue

            to_be_removed = [n]
            source_n = n.args[0]

            # TODO: remove this hack as source_fn_stack is internal implementation detail of torch.export.
            # To make constant value/tensor be tagged as delegatable during partition
            if source_n.op == "get_attr":
                source_n.meta["source_fn_stack"] = list(n.users.keys())[0].meta.get(
                    "source_fn_stack"
                )

            # collecting quant nodes to be removed
            for i in range(1, len(n.args)):
                if isinstance(n.args[i], torch.fx.node.Node):
                    to_be_removed.append(n.args[i])
                    # could be a commonly shared attribute between q & dq
                    if n.args[i].target == exir_ops.edge.aten._to_copy.default:
                        to_be_removed.append(n.args[i].args[0])
            # connect source node to quant users and remove quant node
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            for n in to_be_removed:
                graph_module.graph.erase_node(n)

    def call(self, graph_module: torch.fx.GraphModule):
        self._fold_dq(graph_module)
        self._fold_q(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
