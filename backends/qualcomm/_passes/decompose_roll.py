# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from executorch.exir.pass_base import ExportPass, PassResult

from .utils import merge_decomposed_graph


class SliceCopy(torch.nn.Module):
    def __init__(self, val_shape, shifts, dims):
        super().__init__()
        self.val_shape = val_shape
        if dims[0] is None:
            self.shifts = [shifts[0] % torch.numel(torch.tensor(val_shape))]
        else:
            self.shifts = [shift % val_shape[dim] for shift, dim in zip(shifts, dims)]
        self.dims = dims

    def forward(self, x):
        if self.dims[0] is None:
            y = x.flatten()
            y = torch.cat((y[-self.shifts[0] :], y[: -self.shifts[0]]))
            return y.view(self.val_shape)

        for shift, dim in zip(self.shifts, self.dims):
            x = torch.cat(
                (
                    x[(slice(None),) * dim + (slice(-shift, None),)],
                    x[(slice(None),) * dim + (slice(0, -shift),)],
                ),
                dim=dim,
            )
        return x


class DecomposeRoll(ExportPass):
    """
    Decompose roll into slice and cat.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in graph.nodes:
            if "roll" in str(node.target):
                input_node, shifts = node.args[0], node.args[1]
                dims = node.args[2] if len(node.args) == 3 else None

                # Normalize shifts and dims to lists
                shifts = shifts if isinstance(shifts, (list, tuple)) else [shifts]
                dims = dims if isinstance(dims, (list, tuple)) else [dims]

                model = SliceCopy(input_node.meta["val"].shape, shifts, dims)
                decomposed_module = torch.export.export(
                    model, (input_node.meta["val"],), strict=True
                ).module()

                with graph.inserting_before(node):
                    # remap is used to map original node values to new node values,
                    # which ensures that reference to nodes are correctly updated in the new graph
                    remap = {"x": input_node}
                    merge_decomposed_graph(
                        remap=remap,
                        target_node=node,
                        target_graph=graph,
                        decomposed_graph_module=decomposed_module,
                    )
                    graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
