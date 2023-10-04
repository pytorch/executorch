# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class FoldQDQ(ExportPass):
    """
    Erase QDQ pattern.
    """

    q_ops = {
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    }
    dq_ops = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    }

    def __init__(self):
        super(FoldQDQ, self).__init__()

    def _fold(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # remove dq
        for n in graph_module.graph.nodes:
            user_list = list(n.users.keys())
            if n.target not in self.dq_ops:
                continue
            for user_n in user_list:
                user_n.replace_input_with(n, n.args[0])
            graph_module.graph.erase_node(n)

        # remove q
        for n in graph_module.graph.nodes:
            if n.target not in self.q_ops:
                continue
            to_be_removed = [n]
            source_n = n.args[0]

            # TODO: remove this hack as source_fn_stack is internal implementation detail of torch.export.
            # To make constant value/tensor be tagged as delegatable during partition
            if source_n.op == "get_attr":
                source_n.meta["source_fn_stack"] = list(n.users.keys())[0].meta.get(
                    "source_fn_stack"
                )

            # add quantization attributes to the meta of source node
            for i in range(1, len(n.args)):
                if type(n.args[i]) == torch.fx.node.Node:
                    to_be_removed.append(n.args[i])
            # connect source node to quant users and remove quant node
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            for n in to_be_removed:
                graph_module.graph.erase_node(n)

    def call(self, graph_module: torch.fx.GraphModule):
        self._fold(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
