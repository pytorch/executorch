# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


class DecomposeScaledDotProductAttention(ExportPass):
    """
    Decompose from scaled_dot_product_attention to multiple nodes.
    """

    def __init__(self, allow_non_fake_inputs: bool = True) -> None:
        super().__init__()
        # With allow_non_fake_inputs=False, we don't get _unsafe_view ops
        # in the graph, we allow disabling it here.
        self._allow_non_fake_inputs = allow_non_fake_inputs

    def call(
        self, graph_module: torch.fx.GraphModule, allow_non_fake_inputs: bool = True
    ) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == torch.ops.aten.scaled_dot_product_attention.default:
                input_tensors = (arg.meta["val"] for arg in node.args)

                # refer to pytorch/test/test_decomp.py
                decomposed_module = make_fx(
                    node.target,
                    decomposition_table=get_decompositions(  # pyre-fixme[6]
                        [
                            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                        ]
                    ),
                    tracing_mode="fake",
                    _allow_non_fake_inputs=allow_non_fake_inputs,
                )(*input_tensors)
                with graph.inserting_before(node):
                    name_to_input_tensor_map = {}
                    for i, arg in enumerate(node.args):
                        name_to_input_tensor_map[f"arg{i}_1"] = arg

                    decomposed_node_to_subgraph_node = {}
                    last_decomposed_node = None
                    # Create a mapping from input nodes in decomposed module to original nodes.
                    # In decomposed module, there are only input tensors for placeholder op.
                    for decomposed_node in decomposed_module.graph.nodes:
                        if decomposed_node.op == "placeholder":
                            decomposed_node_to_subgraph_node[decomposed_node] = (
                                name_to_input_tensor_map[decomposed_node.name]
                            )

                        if decomposed_node.op == "output":
                            last_decomposed_node = decomposed_node.args[0]

                    # Copy node from decompose graph module
                    for decomposed_node in decomposed_module.graph.nodes:
                        if decomposed_node.op == "placeholder":
                            continue

                        if (
                            decomposed_node.op == "output"
                            and last_decomposed_node is not None
                        ):
                            for user in node.users.copy():
                                user.replace_input_with(
                                    node,
                                    decomposed_node_to_subgraph_node[
                                        last_decomposed_node
                                    ],
                                )
                            continue

                        subgraph_node = graph.node_copy(
                            decomposed_node,
                            arg_transform=lambda x: decomposed_node_to_subgraph_node[  # noqa: B023
                                x
                            ],
                        )
                        subgraph_node.meta["source_fn_stack"] = [
                            (subgraph_node, subgraph_node.target)
                        ]
                        decomposed_node_to_subgraph_node[decomposed_node] = (
                            subgraph_node
                        )

                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
