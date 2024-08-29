# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import re
from typing import List

import torch

from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.exported_program import ExportedProgram
from torch.library import impl, Library


fallback_op_lib = Library("llama", "DEF")
# registering an operator.
fallback_op_lib.define("fallback(Tensor input) -> Tensor")


@impl(fallback_op_lib, "fallback")
def fallback_impl(a: torch.Tensor) -> torch.Tensor:
    return a


# registering the out variant.
fallback_op_lib.define("fallback.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)")


@impl(fallback_op_lib, "fallback.out")
def fallback_out_impl(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(a)
    return out


class SplitGraph(ExportPass):
    """
    Class to split the model to multiple partitions.
    Because there is limited memory on the device, it could
    not load all llama model in one pte.
    """

    def __init__(self, shard_layers: List[int]):
        super().__init__()
        self.shard_layers = shard_layers

    def _insert_fallback_op(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """
        Insert fallback op before layer that needs to be shard.
        Example:
            There is 12 layers llama model and num_sharding is 3.
            The first partition will contain layers [0, 4) and embedding.
            The second partition will contain layers [4, 8).
            The third partition will contain layers [8, 12) and output.
        """
        pattern = r"layers.(\d+)"
        prev_node = None
        prev_layer = None
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or "nn_module_stack" not in node.meta:
                continue

            module_values_list = list(node.meta["nn_module_stack"].values())
            full_qualified_name = module_values_list[-1][0]
            # Search which layer this node belongs to
            match = re.search(pattern, full_qualified_name)
            if match is None:
                continue

            cur_layer = int(match.group(1))
            # Check the current node which is the last node of the layer
            if cur_layer in self.shard_layers and prev_layer == cur_layer - 1:
                with graph_module.graph.inserting_after(prev_node):
                    users = list(prev_node.users.keys())
                    inserted_node = graph_module.graph.create_node(
                        "call_function",
                        exir_ops.edge.llama.fallback.default,
                        (prev_node,),
                    )
                    inserted_node.meta["val"] = prev_node.meta["val"]
                    if prev_node.meta.get(QCOM_QUANT_ATTRS, None):
                        inserted_node.meta[QCOM_QUANT_ATTRS] = prev_node.meta[
                            QCOM_QUANT_ATTRS
                        ]
                    for user in users:
                        user.replace_input_with(prev_node, inserted_node)

            prev_layer = cur_layer
            prev_node = node

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert_fallback_op(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)


def split_graph(edge_program: ExportedProgram, num_layers: int, shares: int):
    graph_module = edge_program.graph_module
    shard_layers = list(range(0, num_layers, int(num_layers / shares)))
    return SplitGraph(shard_layers)(graph_module)
