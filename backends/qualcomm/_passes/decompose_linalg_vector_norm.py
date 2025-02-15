# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import to_edge
from executorch.exir.pass_base import ExportPass, PassResult


class LinalgVectorNorm(torch.nn.Module):
    def __init__(self, exp, dim, keepdim):
        super().__init__()
        self.exp = exp
        self.dim = tuple(dim) if dim is not None else None
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            x = torch.flatten(x)
            self.dim = 0

        x = torch.abs(x)
        x = torch.pow(x, self.exp)
        x = torch.sum(x, dim=self.dim, keepdim=self.keepdim)
        return torch.pow(x, 1.0 / self.exp)


class DecomposeLinalgVectorNorm(ExportPass):
    """
    Decompose for math equivalent op.
    """

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()
        self.quantization_capture = quantization_capture

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if "linalg_vector_norm" in str(node.target):
                ord = node.args[1] if len(node.args) > 1 else 2.0
                dim = node.args[2] if len(node.args) > 2 else None
                keepdim = node.args[3] if len(node.args) > 3 else False
                model = LinalgVectorNorm(ord, dim, keepdim)
                if self.quantization_capture:
                    decomposed_module = torch.export.export(
                        model, (node.args[0].meta["val"],)
                    ).module()
                else:
                    edge_mgr = to_edge(
                        torch.export.export(model, (node.args[0].meta["val"],))
                    )
                    decomposed_module = edge_mgr.exported_program()

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": node.args[0]}

                    for decomposed_node in decomposed_module.graph.nodes:
                        # no need to copy existent 'output'
                        if decomposed_node.op == "output":
                            for user in node.users.copy():
                                # remap
                                user.replace_input_with(
                                    node,
                                    remap[decomposed_node.args[0][0]],
                                )
                        # no need to copy existent placeholders
                        elif decomposed_node.op == "placeholder":
                            # replace node map from string to graph node
                            remap[decomposed_node] = remap.pop(decomposed_node.name)
                        else:
                            remap[decomposed_node] = graph.node_copy(
                                decomposed_node,
                                arg_transform=lambda x, remap=remap: remap[x],
                            )

                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
