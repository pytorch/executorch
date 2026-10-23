# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import merge_decomposed_graph


class FloorDivide(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x, y):
        result = torch.div(x, y)
        result = torch.floor(result)
        return result.to(self.dtype)


class DecomposeFloorDivide(ExportPass):
    """
    Decompose for math equivalent op.
    Since QNN does not support floor_divide operations for int32 or int64 inputs,
    it is necessary to decompose the operation into a division using floating-point precision,
    followed by applying the floor function.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if (
                torch.ops.aten.floor_divide.default == node.target
                and not torch.is_floating_point(node.meta["val"])
            ):
                # The divisor may be a Python scalar (e.g. `indices // num_classes`):
                # trace it as a 0-dim tensor, then remap it back to the scalar.
                model = FloorDivide(node.meta["val"].dtype)
                decomposed_module = torch.export.export(
                    model,
                    tuple(
                        (
                            arg.meta["val"]
                            if isinstance(arg, torch.fx.Node)
                            else torch.tensor(arg)
                        )
                        for arg in node.args[:2]
                    ),
                    strict=True,
                ).module()
                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": node.args[0], "y": node.args[1]}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)

        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
