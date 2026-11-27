# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.get_decomposition_pass import GetDecompositionPass
from executorch.backends.cortex_m.quantizer.node_finders import SDPAFinder


@pytest.mark.parametrize("strict", [False, True])
def test_nested_decomposition_preserves_sdpa_provenance(strict):
    class DecomposeSilu(GetDecompositionPass):
        targeted_ops = [torch.ops.aten.silu.default]

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    inputs = (torch.randn(1, 2, 3, 4),)
    graph_module = torch.export.export(Model(), inputs, strict=strict).module()
    parent = next(
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.silu.default
    )
    ancestry = [("attention", torch.ops.aten.scaled_dot_product_attention.default)]
    parent.meta["source_fn_stack"] = ancestry
    parent_identity = (parent.name, parent.target)
    expected = graph_module(*inputs)
    DecomposeSilu()(graph_module)
    children = list(SDPAFinder().find_nodes(graph_module))
    assert len(children) >= 2
    for child in children:
        assert child.meta["source_fn_stack"] == [
            *ancestry,
            parent_identity,
            (child.name, child.target),
        ]
    torch.testing.assert_close(graph_module(*inputs), expected)
