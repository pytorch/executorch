# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import to_edge
from executorch.exir.pass_base import ExportPass, PassResult


class Any(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = tuple(dim) if isinstance(dim, list) else dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            x = torch.flatten(x)
            self.dim = 0

        x = x.to(torch.bool).to(torch.int32)
        x = torch.sum(x, dim=self.dim, keepdim=self.keepdim, dtype=torch.int32)
        return torch.not_equal(x, torch.zeros(1, dtype=torch.int32))


class DecomposeAny(ExportPass):
    """
    Decompose for math equivalent op.
    """

    def __init__(self, quantization_capture=False) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if "any.dim" in str(node.target):
                dim = node.args[1] if len(node.args) > 1 else None
                keepdim = node.args[2] if len(node.args) > 2 else False
                model = Any(dim, keepdim)
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
