# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.symbolic_shape_utils import materialize_symints
from torch.export import Dim, export
from torch.fx import Node


class Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_materialize_symints_materializes_dynamic_shape() -> None:
    exported = export(
        Identity(),
        (torch.randn(2, 3),),
        dynamic_shapes={"x": {0: Dim("batch", min=1, max=8)}},
    )
    graph = exported.graph_module.graph
    input_node = next(node for node in graph.nodes if node.op == "placeholder")
    batch = input_node.meta["val"].shape[0]
    output_node = next(node for node in graph.nodes if node.op == "output")

    with graph.inserting_before(output_node):
        materialized = materialize_symints(graph, [1, batch])

    assert materialized[0] == 1
    assert isinstance(materialized[1], Node)
    assert materialized[1].meta["val"] == batch
    graph.lint()
