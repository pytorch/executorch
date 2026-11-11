# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from executorch.backends.arm._passes.insert_sym_size_int_for_input_dims_pass import (
    InsertSymSizeIntForInputDimsPass,
)
from executorch.backends.transforms.symbolic_shape_utils import materialize_symints
from torch.export import Dim, export


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class AddTwo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 2


def _dynamic_graph() -> tuple[torch.fx.GraphModule, torch.fx.Node]:
    exported_program = export(
        AddOne(),
        (torch.randn(48, 3),),
        dynamic_shapes={"x": {0: Dim("batch", min=2, max=6) * 16}},
    )
    graph_module = exported_program.graph_module
    placeholder = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    return graph_module, placeholder


def _shared_symbol_dynamic_graph() -> tuple[torch.fx.GraphModule, torch.fx.Node]:
    batch = Dim("batch", min=2, max=6)
    exported_program = export(
        AddTwo(),
        (torch.randn(48, 24),),
        dynamic_shapes={"x": {0: batch * 16, 1: batch * 8}},
    )
    graph_module = exported_program.graph_module
    placeholder = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    return graph_module, placeholder


def test_insert_sym_size_ignores_later_matching_node() -> None:
    graph_module, placeholder = _dynamic_graph()
    graph = graph_module.graph
    first_operator = next(node for node in graph.nodes if node.op == "call_function")
    with graph.inserting_before(graph.output_node()):
        later_sym_size = graph.call_function(
            torch.ops.aten.sym_size.int, (placeholder, 0)
        )

    InsertSymSizeIntForInputDimsPass()(graph_module)

    matching_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.sym_size.int
        and node.args == (placeholder, 0)
    ]
    node_order = list(graph.nodes)
    assert len(matching_nodes) == 2
    assert node_order.index(matching_nodes[0]) < node_order.index(first_operator)
    assert node_order.index(later_sym_size) > node_order.index(first_operator)


def test_materialize_symints_ignores_later_matching_node() -> None:
    graph_module, placeholder = _dynamic_graph()
    graph = graph_module.graph
    first_operator = next(node for node in graph.nodes if node.op == "call_function")
    dim = placeholder.meta["val"].shape[0] // 16
    with graph.inserting_before(graph.output_node()):
        later_sym_size = graph.call_function(
            torch.ops.aten.sym_size.int, (placeholder, 0)
        )

    with patch.object(
        torch.fx.Graph,
        "materialize_symints",
        side_effect=AssertionError("materialize_symints: no producer for batch"),
    ):
        with graph.inserting_before(first_operator):
            materialized = materialize_symints(graph, [dim])

    graph.lint()
    node_order = list(graph.nodes)
    assert isinstance(materialized[0], torch.fx.Node)
    assert node_order.index(materialized[0]) < node_order.index(first_operator)
    assert node_order.index(later_sym_size) > node_order.index(first_operator)


def test_materialize_symints_reads_mixed_placeholder_metadata() -> None:
    graph_module, placeholder = _dynamic_graph()
    graph = graph_module.graph
    first_operator = next(node for node in graph.nodes if node.op == "call_function")
    dim = placeholder.meta["val"].shape[0] // 16
    with graph.inserting_before(placeholder):
        other_placeholder = graph.placeholder("other")
    other_placeholder.meta["example_value"] = torch.tensor(1)

    with patch.object(
        torch.fx.Graph,
        "materialize_symints",
        side_effect=AssertionError("materialize_symints: no producer for batch"),
    ):
        with graph.inserting_before(first_operator):
            materialized = materialize_symints(graph, [dim])

    graph.lint()
    assert isinstance(materialized[0], torch.fx.Node)
    assert materialized[0].meta["val"] == dim


def test_materialize_symints_reuses_first_source_for_shared_symbol() -> None:
    graph_module, placeholder = _shared_symbol_dynamic_graph()
    graph = graph_module.graph
    first_operator = next(node for node in graph.nodes if node.op == "call_function")
    dim = placeholder.meta["val"].shape[0] // 16

    with patch.object(
        torch.fx.Graph,
        "materialize_symints",
        side_effect=AssertionError("materialize_symints: no producer for batch"),
    ):
        with graph.inserting_before(first_operator):
            materialized = materialize_symints(graph, [dim])

    graph.lint()
    sym_size_nodes = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.sym_size.int
    ]
    assert len(sym_size_nodes) == 1
    assert isinstance(materialized[0], torch.fx.Node)
