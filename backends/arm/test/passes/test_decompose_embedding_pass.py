# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.decompose_embedding_pass import (
    DecomposeEmbeddingPass,
)
from executorch.backends.arm.quantizer.arm_quantizer_utils import NodeFinder
from executorch.backends.cortex_m.quantizer.node_finders import (
    ModuleNameNodeFinder,
    ModuleTypeNodeFinder,
)
from torch.fx import Node


class Embedding(torch.nn.Module):
    """Exercise the exported aten.embedding op from a real named submodule."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def get_inputs(self, dtype: torch.dtype = torch.int32) -> tuple[torch.Tensor]:
        return (torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype),)


def _decompose_embedding(model: Embedding, inputs: tuple[torch.Tensor]):
    graph_module = torch.export.export(model, inputs).module()
    source_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.embedding.default
    ]
    assert len(source_nodes) == 1
    original_stack = source_nodes[0].meta.get("nn_module_stack")
    assert original_stack, "Export must preserve Embedding module provenance"

    result = DecomposeEmbeddingPass(tfa_pass=True)(graph_module)
    assert result is not None, "Embedding decomposition must return a PassResult"
    nodes = list(result.graph_module.graph.nodes)
    lookup_nodes = [
        node for node in nodes if node.target == torch.ops.aten.index_select.default
    ]
    assert len(lookup_nodes) == 1
    lookup = lookup_nodes[0]

    flattened_indices = lookup.args[2]
    assert isinstance(flattened_indices, Node)
    assert flattened_indices.target == torch.ops.aten.reshape.default

    restored_outputs = [
        node
        for node in nodes
        if node.target == torch.ops.aten.reshape.default and node.args[0] is lookup
    ]
    assert len(restored_outputs) == 1
    restored_output = restored_outputs[0]

    return (
        result.graph_module,
        (flattened_indices, lookup, restored_output),
        original_stack,
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_embedding_decomposition_preserves_module_metadata(index_dtype: torch.dtype):
    """Both reshapes and the lookup must inherit the original module stack."""
    model = Embedding().eval()
    inputs = model.get_inputs(dtype=index_dtype)
    graph_module, generated_nodes, original_stack = _decompose_embedding(model, inputs)

    assert all(
        node.meta.get("nn_module_stack") == original_stack for node in generated_nodes
    )
    assert generated_nodes[0].meta["val"].dtype == index_dtype
    assert generated_nodes[1].meta["val"].is_floating_point()
    torch.testing.assert_close(graph_module(*inputs), model(*inputs))


@pytest.mark.parametrize("selector", ["module_type", "module_name"])
def test_embedding_decomposition_remains_selectable_by_module(selector: str):
    """The replacement ops must match selectors used by the Arm quantizer."""
    model = Embedding().eval()
    graph_module, generated_nodes, _ = _decompose_embedding(model, model.get_inputs())

    finder: NodeFinder
    if selector == "module_type":
        finder = ModuleTypeNodeFinder(torch.nn.Embedding)
    else:
        finder = ModuleNameNodeFinder("embedding")

    selected_nodes = set(finder.find_nodes(graph_module))
    assert set(generated_nodes) <= selected_nodes
