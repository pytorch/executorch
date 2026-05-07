# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.remove_permutes_around_elementwise_tosa_ops import (
    RemovePermutesAroundElementwiseTosaOps,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops

TOSA_INT_SPEC = TosaSpecification.create_from_string("TOSA-1.0+INT")
PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
RESCALE_TARGET = exir_ops.backend.tosa.RESCALE.default
TABLE_TARGET = exir_ops.backend.tosa.TABLE.default


def _count_nodes(graph_module: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target
    )


def test_remove_permutes_around_rescale_tosa_INT() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(1, 3, 4, 5)

    permute_in = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(x, [0, 2, 3, 1]),
    )
    rescale = graph.create_node(
        "call_function",
        RESCALE_TARGET,
        args=(permute_in, torch.int8, [1.0], 0, 0),
    )
    permute_out = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(rescale, [0, 3, 1, 2]),
    )
    graph.output(permute_out)

    graph_module = torch.fx.GraphModule({}, graph)

    with TosaLoweringContext(TOSA_INT_SPEC):
        result = RemovePermutesAroundElementwiseTosaOps().call(graph_module)

    assert result.modified
    assert _count_nodes(result.graph_module, PERMUTE_TARGET) == 0
    assert _count_nodes(result.graph_module, RESCALE_TARGET) == 1
