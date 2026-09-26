# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import (
    MoveDataMovementOpsToSmallerDtypePass,
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops


PERMUTE = exir_ops.edge.aten.permute_copy.default
RESCALE = exir_ops.backend.tosa.RESCALE.default


def _apply_pass(graph: torch.fx.Graph) -> torch.fx.GraphModule:
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        return MoveDataMovementOpsToSmallerDtypePass().call(graph_module).graph_module


def _call_nodes(graph_module: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [node for node in graph_module.graph.nodes if node.op == "call_function"]


def test_moves_permute_after_narrowing_rescale() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(permute, torch.int8, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output(rescale)

    graph_module = _apply_pass(graph)
    rescale, permute = _call_nodes(graph_module)

    assert rescale.target == RESCALE
    assert permute.target == PERMUTE
    assert rescale.meta["val"].shape == torch.Size((1, 2, 3, 4))
    assert permute.meta["val"].dtype == torch.int8


def test_moves_permute_before_widening_rescale() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    rescale = graph.call_function(RESCALE, args=(x, torch.int32, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    graph.output(permute)

    graph_module = _apply_pass(graph)
    permute, rescale = _call_nodes(graph_module)

    assert permute.target == PERMUTE
    assert rescale.target == RESCALE
    assert permute.meta["val"].dtype == torch.int8
    assert rescale.meta["val"].shape == torch.Size((1, 3, 4, 2))


def test_keeps_per_channel_rescale_order() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(permute, torch.int8, [1.0, 2.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output(rescale)

    graph_module = _apply_pass(graph)

    assert [node.target for node in _call_nodes(graph_module)] == [PERMUTE, RESCALE]


def test_splits_merged_permute_at_distinct_narrowing_rescales() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    rescale_1 = graph.call_function(RESCALE, args=(permute, torch.int8, [1.0], 0, 0))
    rescale_1.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    rescale_2 = graph.call_function(RESCALE, args=(permute, torch.int8, [2.0], 0, 0))
    rescale_2.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output((rescale_1, rescale_2))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        graph_module = (
            PropagateViewCopyPermuteDownPass().call(graph_module).graph_module
        )
        graph_module = PropagateViewCopyPermuteUpPass().call(graph_module).graph_module
        assert sum(node.target == PERMUTE for node in _call_nodes(graph_module)) == 1
        graph_module = (
            MoveDataMovementOpsToSmallerDtypePass().call(graph_module).graph_module
        )

    call_nodes = _call_nodes(graph_module)
    permutes = [node for node in call_nodes if node.target == PERMUTE]
    assert len(permutes) == 2
    assert all(node.meta["val"].dtype == torch.int8 for node in permutes)
    assert all(node.all_input_nodes[0].target == RESCALE for node in permutes)
