# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import _operator
from typing import Dict, Tuple

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_nn_module_stack


class DecomposeWrapWithAutocast(ExportPass):
    """
    Decompose the _higher_order_ops WrapWithAutocast
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_submod(
        self, gm: torch.fx.GraphModule, node: torch.fx.Node
    ) -> Tuple[torch.fx.GraphModule, str]:
        for a in node.args:
            if isinstance(a, torch.fx.Node) and "submod" in a.target:
                return getattr(gm, a.target), a.target

    def _replace_output(
        self, wwac_node: torch.fx.Node, output_node: torch.fx.Node, remap: Dict
    ):
        for user in wwac_node.users.copy():
            arg_idx = 0
            is_user_getitem = False

            if user.target == _operator.getitem:
                arg_idx = user.args[1]
                is_user_getitem = True

            user.replace_input_with(
                wwac_node,
                remap[output_node.args[0][arg_idx]],
            )

            if is_user_getitem:
                for user_user in user.users.copy():
                    user_user.replace_input_with(user, user.args[0])

    def _replace(self, gm: torch.fx.GraphModule) -> None:
        graph = gm.graph
        for node in graph.nodes:
            if isinstance(node.target, torch._higher_order_ops.wrap.WrapWithAutocast):
                submod, submod_name = self._get_submod(gm, node)
                n_args = node.args
                input_submod = n_args[4]
                decomposed_module = submod
                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    # remap = {"expand_1": node.args[5], "to_4": node.args[6]}
                    remap = {n_args[i].name: n_args[i] for i in range(5, len(n_args))}

                    for decomposed_node in decomposed_module.graph.nodes:
                        copy_nn_module_stack(node, decomposed_node)
                        # no need to copy existent 'output'
                        if decomposed_node.op == "output":
                            self._replace_output(node, decomposed_node, remap)
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

                graph.erase_node(input_submod)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._replace(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
