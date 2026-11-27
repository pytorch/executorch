# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import RewriteHighRankSingletonPermutePass
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export
from torch.fx import Node


class _HighRankPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 4, 1, 2, 3)


def test_rewrite_high_rank_permute_materializes_dynamic_view_shapes() -> None:
    model = _HighRankPermute()
    example = torch.randn(2, 1, 2, 1, 3)
    exported = export(
        model,
        (example,),
        dynamic_shapes={"x": {0: Dim("batch", min=1, max=8)}},
    )
    graph_module = to_edge(exported).exported_program().graph_module

    result = RewriteHighRankSingletonPermutePass()(graph_module)
    assert result is not None
    rewritten = result.graph_module
    rewritten.graph.lint()

    views = [
        node
        for node in rewritten.graph.nodes
        if node.target == exir_ops.edge.aten.view_copy.default
    ]
    permutes = [
        node
        for node in rewritten.graph.nodes
        if node.target == exir_ops.edge.aten.permute_copy.default
    ]
    assert len(views) == 2
    assert all(any(isinstance(dim, Node) for dim in view.args[1]) for view in views)
    assert len(permutes) == 1
    assert len(permutes[0].args[1]) == 3

    for batch in (2, 4):
        inputs = torch.randn(batch, 1, 2, 1, 3)
        torch.testing.assert_close(rewritten(inputs)[0], model(inputs))
