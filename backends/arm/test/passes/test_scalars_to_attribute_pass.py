# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest
import torch
from executorch.backends.arm._passes.scalars_to_attribute_pass import (
    ScalarsToAttributePass,
)
from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.cortex_m.quantizer.node_finders import (
    ModuleNameNodeFinder,
    ModuleTypeNodeFinder,
)
from torch.fx import GraphModule, Node


class ReverseSubtract(torch.nn.Module):
    """A real submodule so torch.export records module provenance."""

    def __init__(self, scalar: float = 2.5) -> None:
        super().__init__()
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsub(x, self.scalar)


class ReverseSubtractModel(torch.nn.Module):
    def __init__(self, scalar: float = 2.5) -> None:
        super().__init__()
        self.reverse_sub = ReverseSubtract(scalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reverse_sub(x)


def _rewrite_rsub(
    scalar: float = 2.5,
) -> tuple[
    ReverseSubtractModel,
    tuple[torch.Tensor],
    GraphModule,
    Node,
    dict[str, Any],
]:
    model = ReverseSubtractModel(scalar).eval()
    inputs = (torch.randn(2, 3),)
    graph_module = torch.export.export(model, inputs).module()

    source_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.rsub.Scalar
    ]
    assert len(source_nodes) == 1, "Expected one exported rsub.Scalar"
    source = source_nodes[0]
    assert source.meta.get("nn_module_stack"), "Export must preserve module stack"

    # Test propagation beyond meta["val"]. False does not disable this TFA pass.
    source.meta[DISALLOW_TFA_META_KEY] = False
    source.meta["_rsub_test_provenance"] = "source-rsub"
    source_meta = dict(source.meta)

    result = ScalarsToAttributePass(tfa_pass=True)(graph_module)
    assert result is not None, "Scalar conversion must return a PassResult"
    assert result.modified, "rsub.Scalar must be converted"
    transformed = result.graph_module

    replacements = [
        node
        for node in transformed.graph.nodes
        if node.target == torch.ops.aten.sub.Tensor
    ]
    assert len(replacements) == 1, "Expected one replacement sub.Tensor"
    assert not any(
        node.target == torch.ops.aten.rsub.Scalar for node in transformed.graph.nodes
    )
    return model, inputs, transformed, replacements[0], source_meta


def test_rsub_scalar_decomposition_preserves_module_metadata() -> None:
    """The replacement sub.Tensor must inherit rsub's nn_module_stack."""
    _, _, _, replacement, source_meta = _rewrite_rsub()
    assert replacement.meta.get("nn_module_stack") == source_meta["nn_module_stack"]


@pytest.mark.parametrize("selector", ["module_type", "module_name"])
def test_rsub_scalar_decomposition_remains_selectable(selector: str) -> None:
    """Module-type/name filters used by Arm selective quantization still
    match.
    """
    _, _, transformed, replacement, _ = _rewrite_rsub()
    if selector == "module_type":
        selected = set(ModuleTypeNodeFinder(ReverseSubtract).find_nodes(transformed))
    else:
        selected = set(ModuleNameNodeFinder("reverse_sub").find_nodes(transformed))
    assert replacement in selected


def test_rsub_scalar_decomposition_preserves_other_metadata() -> None:
    """A metadata-preserving rewrite must not copy only the fake tensor
    value.
    """
    _, _, _, replacement, source_meta = _rewrite_rsub()
    assert replacement.meta.get("_rsub_test_provenance") == "source-rsub"
    assert DISALLOW_TFA_META_KEY in replacement.meta
    assert replacement.meta[DISALLOW_TFA_META_KEY] is False
    assert replacement.meta["val"].shape == source_meta["val"].shape
    assert replacement.meta["val"].dtype == source_meta["val"].dtype


@pytest.mark.parametrize("scalar", [2.5, -1.25, 0.0])
def test_rsub_scalar_decomposition_numerical_equivalence(scalar: float) -> None:
    """Swapping the operand order must retain normal rsub semantics."""
    model, inputs, transformed, _, _ = _rewrite_rsub(scalar)
    torch.testing.assert_close(transformed(*inputs), model(*inputs))
