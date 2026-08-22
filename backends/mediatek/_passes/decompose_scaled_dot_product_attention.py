# Copyright (c) 2024 MediaTek Inc.
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import torch

from executorch.exir.pass_base import ExportPass, PassResult
from torch._decomp import get_decompositions
from torch.fx import Graph
from torch.fx.experimental.proxy_tensor import make_fx


def _get_input_node_names(graph: Graph):
    input_names = []
    for node in graph.nodes:
        if node.op == "placeholder":
            input_names.append(node.name)
    return input_names


class DecomposeScaledDotProductAttention(ExportPass):
    """Decompose the single SDPA operator."""

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target != torch.ops.aten.scaled_dot_product_attention.default:
                continue

            decom_mappings = get_decompositions(
                [torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default]
            )
            input_tensors = (arg.meta["val"] for arg in node.args)
            decomposed_module = make_fx(node.target, decom_mappings, "fake", True)(
                *input_tensors
            )
            decomposed_input_names = _get_input_node_names(decomposed_module.graph)
            with graph.inserting_before(node):
                name_to_input_tensor_map = {}
                for idx, arg in enumerate(node.args):
                    name_to_input_tensor_map[decomposed_input_names[idx]] = arg

                decomposed_node_to_subgraph_node = {}
                for decomposed_node in decomposed_module.graph.nodes:
                    if decomposed_node.op == "placeholder":
                        decomposed_node_to_subgraph_node[decomposed_node] = (
                            name_to_input_tensor_map[decomposed_node.name]
                        )

                # Copy node from decompose graph module
                for decomposed_node in decomposed_module.graph.nodes:
                    if decomposed_node.op == "placeholder":
                        continue
                    if decomposed_node.op == "output":
                        for user in node.users.copy():
                            new_node = decomposed_node_to_subgraph_node[
                                decomposed_node.args[0]
                            ]
                            user.replace_input_with(node, new_node)
                        continue

                    subgraph_node = graph.node_copy(
                        decomposed_node,
                        arg_transform=lambda x, d=decomposed_node_to_subgraph_node: d[
                            x
                        ],
                    )
                    subgraph_node.meta["source_fn_stack"] = [
                        (subgraph_node, subgraph_node.target)
                    ]
                    decomposed_node_to_subgraph_node[decomposed_node] = subgraph_node

                graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
