# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.transforms.absorb_boundary_layout_copies import (
    AbsorbBoundaryLayoutCopies,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops

_LAYOUT_COPY = exir_ops.edge.channels_last.permute_copy.default
_ATEN_PERMUTE = exir_ops.edge.aten.permute_copy.default
_QUANTIZE = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default

_TO_NHWC = (0, 2, 3, 1)
_TO_NCHW = (0, 3, 1, 2)


class Region(torch.nn.Module):
    """A layout region: a body bracketed by a permute and its inverse."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.permute(*_TO_NHWC)).permute(*_TO_NCHW)


class Fork(torch.nn.Module):
    """A region whose entry copy feeds two consumers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(*_TO_NHWC)
        return (torch.relu(y) + torch.sigmoid(y)).permute(*_TO_NCHW)


class MixedUse(torch.nn.Module):
    """An input read both through a region and directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.permute(*_TO_NHWC)).permute(*_TO_NCHW) + x


def _count(graph_module, target) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def _build_region(module, inputs):
    """Export ``module`` and retarget its permutes to the layout dialect.

    ``ToContiguousChannelsLastPass`` emits exactly these nodes, but building
    them here keeps this suite independent of it: absorption is defined against
    the dialect operator, not against whoever produced it.
    """
    module.eval()
    with torch.no_grad():
        exported = torch.export.export(module, inputs)
        edge = to_edge(
            exported,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        )
    graph_module = edge.exported_program().graph_module
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == _ATEN_PERMUTE:
            node.target = _LAYOUT_COPY
    graph_module.recompile()
    assert _count(graph_module, _LAYOUT_COPY) > 0
    return edge


def _split_shared_copy(edge):
    """Give each consumer of the entry copy its own copy.

    Region formation inserts one copy per anchor; export would have collapsed
    identical ones, so the fan-out shape is constructed directly.
    """
    graph_module = edge.exported_program().graph_module
    graph = graph_module.graph
    copy = next(
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == _LAYOUT_COPY
        and node.args[0].op == "placeholder"
    )
    users = list(copy.users)
    assert len(users) > 1
    for user in users[1:]:
        with graph.inserting_before(user):
            clone = graph.call_function(_LAYOUT_COPY, copy.args, copy.kwargs)
        clone.meta.update(copy.meta)
        user.replace_input_with(copy, clone)
    graph_module.recompile()
    return edge


def _absorb(edge):
    layout_pass = AbsorbBoundaryLayoutCopies(edge.exported_program())
    return edge.transform([layout_pass]), layout_pass.contract


def _run(edge, contract, inputs):
    """Invoke the method through its (possibly rewritten) layout contract."""
    args = list(inputs)
    for index, dims in contract.inputs.items():
        args[index] = args[index].permute(list(dims)).contiguous()
    result = edge.exported_program().module()(*args)
    results = list(result) if isinstance(result, (tuple, list)) else [result]
    for index, dims in contract.outputs.items():
        results[index] = results[index].permute(list(dims))
    return results[0] if len(results) == 1 else results


@pytest.mark.parametrize("module", [Region(), Fork()])
def test_boundary_copies_are_absorbed_and_numerics_hold(module) -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module.eval()(*inputs)
    edge = _build_region(module, inputs)

    edge, contract = _absorb(edge)

    assert contract.inputs and contract.outputs
    assert _count(edge.exported_program().graph_module, _LAYOUT_COPY) == 0
    assert torch.allclose(_run(edge, contract, inputs), expected, atol=1e-6)


def test_fan_out_collapses_to_one_contract_entry() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    module = Fork()
    expected = module.eval()(*inputs)
    edge = _split_shared_copy(_build_region(module, inputs))
    assert _count(edge.exported_program().graph_module, _LAYOUT_COPY) == 3

    edge, contract = _absorb(edge)

    assert list(contract.inputs) == [0]
    assert _count(edge.exported_program().graph_module, _LAYOUT_COPY) == 0
    assert torch.allclose(_run(edge, contract, inputs), expected, atol=1e-6)


def test_mixed_users_are_left_alone() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    module = MixedUse()
    expected = module.eval()(*inputs)
    edge = _build_region(module, inputs)

    edge, contract = _absorb(edge)

    assert 0 not in contract.inputs
    assert torch.allclose(_run(edge, contract, inputs), expected, atol=1e-6)


def test_per_tensor_quantization_is_traversed() -> None:
    """A copy behind a per-tensor quantize is still a boundary copy.

    Quantized graphs interpose q/dq between the placeholder and the region;
    those reorder nothing, so absorption has to see through them.
    """
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _build_region(Region(), inputs)
    graph_module = edge.exported_program().graph_module
    graph = graph_module.graph
    entry = next(
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == _LAYOUT_COPY
        and node.args[0].op == "placeholder"
    )
    placeholder = entry.args[0]
    with graph.inserting_after(placeholder):
        quantize = graph.call_function(
            _QUANTIZE, (placeholder, 1.0, 0, -128, 127, torch.int8)
        )
    with graph.inserting_after(quantize):
        dequantize = graph.call_function(
            _DEQUANTIZE, (quantize, 1.0, 0, -128, 127, torch.int8)
        )
    quantize.meta.update(placeholder.meta)
    dequantize.meta.update(placeholder.meta)
    entry.replace_input_with(placeholder, dequantize)
    graph_module.recompile()

    _, contract = _absorb(edge)

    assert contract.inputs == {0: _TO_NHWC}


def test_absorbing_is_idempotent() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _build_region(Region(), inputs)
    edge, first = _absorb(edge)
    edge, second = _absorb(edge)

    assert first
    assert not second


def test_signature_stays_valid() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _build_region(Region(), inputs)

    edge, contract = _absorb(edge)

    assert contract
    edge.exported_program()._validate()
