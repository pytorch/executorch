# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import cast, List

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

# This is a list of ops that are No Ops if used with an empty tensor.
# Which means that if we remove the empty tensor as input to this op,
# The result of the operation will stay the same


class PruneEmptyTensorsPass(ExportPass):
    """
    Removes Any empty tensors from the graph that can safely be removed
    without affecting the results of the graph. Currently we remove empty
    tensor operations from the following ops:
    - aten.cat.default
    """

    def remove_empty_tensors_from_cat(
        self, graph_module: GraphModule, cat_node: Node
    ) -> None:
        """
        Removes empty tensors from the graph that are inputs to aten.cat.default
        """
        concat_list = cast(List[Node], cat_node.args[0])
        pruned_concat_list = []
        for input_arg in concat_list:
            input_arg_tensor = input_arg.meta["val"]
            if input_arg_tensor.numel() != 0:
                pruned_concat_list.append(input_arg)

        cat_node.args = (pruned_concat_list,) + cat_node.args[1:]
        if len(pruned_concat_list) == 0:
            # if all the inputs to the cat are empty tensors, then we can replace
            # this concat node with an aten full like
            cat_tensor = cat_node.meta["val"]
            with graph_module.graph.inserting_after(cat_node):
                full_like = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.aten.full.default,
                    args=(tuple(cat_tensor.shape), 0),
                    kwargs={"dtype": cat_tensor.dtype},
                )
                full_like.meta = cat_node.meta
                cat_node.replace_all_uses_with(full_like)

    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            if node.target == torch.ops.aten.cat.default:
                self.remove_empty_tensors_from_cat(graph_module, node)

        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()

        return PassResult(graph_module, True)
