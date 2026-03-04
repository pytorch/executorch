# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult
from torch._decomp import get_decompositions
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx


class GetDecompositionPass(ArmPass):

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops: list[OpOverload] = []

    def __init__(self, tfa_pass=False, *args, **kwargs):
        super().__init__(tfa_pass, *args, **kwargs)

        if type(self) is GetDecompositionPass:
            raise TypeError("Base class DecomposePass cannot be instantiated directly.")

    def _skip_pass(self, input_tensors: list):
        return False

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target not in self.targeted_ops
                or not self.allowed_to_transform(node.meta)
            ):
                continue

            input_tensors = []
            for arg in node.args:
                if hasattr(arg, "meta"):
                    input_tensors.append(arg.meta["val"])

                elif isinstance(arg, int):
                    input_tensors.append(arg)

            if self._skip_pass(input_tensors):
                continue

            # refer to pytorch/test/test_decomp.py
            decomposed_module = make_fx(
                node.target,
                decomposition_table=get_decompositions(self.targeted_ops),  # type: ignore[arg-type]
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*input_tensors)

            with graph_module.graph.inserting_before(node):
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
                    decomposed_node.meta["nn_module_stack"] = node.meta.get(
                        "nn_module_stack"
                    )
                    if decomposed_node.op == "placeholder":
                        continue

                    if (
                        decomposed_node.op == "output"
                        and last_decomposed_node is not None
                    ):
                        for user in node.users.copy():
                            user.replace_input_with(
                                node,
                                decomposed_node_to_subgraph_node[last_decomposed_node],
                            )
                        continue

                    subgraph_node = graph_module.graph.node_copy(
                        decomposed_node,
                        arg_transform=lambda x: decomposed_node_to_subgraph_node[  # noqa: B023
                            x
                        ],
                    )
                    subgraph_node.meta["source_fn_stack"] = [
                        (subgraph_node, subgraph_node.target)
                    ]
                    decomposed_node_to_subgraph_node[decomposed_node] = subgraph_node

                graph_module.graph.erase_node(node)

                modified = True
        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)
