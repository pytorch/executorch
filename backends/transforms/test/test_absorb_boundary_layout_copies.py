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
from executorch.backends.transforms.to_contiguous_channels_last_pass import (
    ToContiguousChannelsLastPass,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops

_LAYOUT_COPY = exir_ops.edge.channels_last.permute_copy.default


class Conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvPool(torch.nn.Module):
    """One input feeding two branches, so the region brackets it twice."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.pool(x)


def _count(graph_module, target) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def _lower(module, inputs):
    module.eval()
    with torch.no_grad():
        exported = torch.export.export(module, inputs)
        edge = to_edge(
            exported,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        )
        edge = edge.transform([ToContiguousChannelsLastPass(edge.exported_program())])
    return edge


def _absorb(edge):
    layout_pass = AbsorbBoundaryLayoutCopies(edge.exported_program())
    return edge.transform([layout_pass]), layout_pass.contract


def _run(edge, contract, inputs):
    args = list(inputs)
    for index, dims in contract.inputs.items():
        args[index] = args[index].permute(list(dims)).contiguous()
    result = edge.exported_program().module()(*args)
    results = list(result) if isinstance(result, (tuple, list)) else [result]
    for index, dims in contract.outputs.items():
        results[index] = results[index].permute(list(dims))
    return results[0] if len(results) == 1 else results


@pytest.mark.parametrize("module", [Conv(), ResidualConvPool()])
def test_boundary_copies_are_absorbed_and_numerics_hold(module) -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    expected = module.eval()(*inputs)
    edge = _lower(module, inputs)
    assert _count(edge.exported_program().graph_module, _LAYOUT_COPY) > 0

    edge, contract = _absorb(edge)

    assert contract.inputs and contract.outputs
    assert _count(edge.exported_program().graph_module, _LAYOUT_COPY) == 0
    assert torch.allclose(_run(edge, contract, inputs), expected, atol=1e-6)


def test_fan_out_collapses_to_one_contract_entry() -> None:
    """Both branches of a residual share the input, so one entry covers them."""
    module = ResidualConvPool()
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _lower(module, inputs)
    copies_on_input = [
        node
        for node in edge.exported_program().graph_module.graph.nodes
        if node.op == "placeholder"
        and node.name in edge.exported_program().graph_signature.user_inputs
        for _ in node.users
    ]
    assert len(copies_on_input) > 1

    _, contract = _absorb(edge)

    assert list(contract.inputs) == [0]


def test_mixed_users_are_left_alone() -> None:
    """An input consumed both by a layout region and directly is not a boundary."""

    class MixedUse(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)

        def forward(self, x):
            return self.conv(x) + x

    inputs = (torch.randn(1, 4, 8, 8),)
    module = MixedUse()
    expected = module.eval()(*inputs)
    edge = _lower(module, inputs)

    edge, contract = _absorb(edge)

    assert 0 not in contract.inputs
    assert torch.allclose(_run(edge, contract, inputs), expected, atol=1e-6)


def test_absorbing_is_idempotent() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _lower(Conv(), inputs)
    edge, first = _absorb(edge)
    edge, second = _absorb(edge)

    assert first
    assert not second


def test_signature_stays_valid() -> None:
    inputs = (torch.randn(1, 4, 8, 8),)
    edge = _lower(Conv(), inputs)
    edge, contract = _absorb(edge)

    assert contract
    edge.exported_program()._validate()


# The layout pass is measured against these 36 models in
# test_to_contiguous_channels_last_pass.py, which pins its own per-case counts
# but stops before absorption. These are the totals across that matrix.
_MATRIX_BASELINE_PERMUTES = 138
_MATRIX_LAYOUT_ONLY_PERMUTES = 156
_MATRIX_ABSORBED_PERMUTES = 137

# Absorption does not destroy 19 permutes, it moves them across the method
# boundary: 17 become obligations on the caller. Pinning both numbers keeps the
# in-graph count honest, since it could otherwise be driven to zero by handing
# the caller unlimited work. The gap is the genuine saving, and it comes from
# fan-out — one placeholder feeding several branches needs several copies but
# only one contract entry.
_MATRIX_CONTRACT_ENTRIES = 17

# Cases still above baseline after absorbing. Every one of these is an internal
# copy left by a region that a non-permutable op (batch_norm, linear) cut in
# two, which is region-merging work rather than boundary work.
_MATRIX_RESIDUAL_REGRESSIONS = {
    "conv2d_rank3",
    "model_1_conv_maxpool_residual_linear",
    "model_8_conv_batchnorm_maxpool_residual",
    "model_9_dilated_conv_batchnorm_avgpool_residual",
    "views",
}

_PERMUTE_TARGETS = {
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.channels_last.permute_copy.default,
}


def _permutes(edge) -> int:
    return sum(
        node.op == "call_function" and node.target in _PERMUTE_TARGETS
        for node in edge.exported_program().graph.nodes
    )


def test_absorption_pays_for_the_layout_pass_across_the_model_matrix() -> None:
    from executorch.backends.transforms.test.test_to_contiguous_channels_last_pass import (
        cases,
    )

    baseline = layout_only = absorbed = contract_entries = 0
    regressions = set()
    for name, case in cases.items():
        case.module.eval()
        with torch.no_grad():
            exported = torch.export.export(case.module, case.inputs)
            config = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
            case_baseline = _permutes(to_edge(exported, compile_config=config))

            edge = to_edge(exported, compile_config=config)
            edge = edge.transform(
                [ToContiguousChannelsLastPass(edge.exported_program())]
            )
            case_layout = _permutes(edge)

            absorb = AbsorbBoundaryLayoutCopies(edge.exported_program())
            edge = edge.transform([absorb])
            case_absorbed = _permutes(edge)

        baseline += case_baseline
        layout_only += case_layout
        absorbed += case_absorbed
        contract_entries += len(absorb.contract.inputs) + len(absorb.contract.outputs)
        if case_absorbed > case_baseline:
            regressions.add(name)

    assert baseline == _MATRIX_BASELINE_PERMUTES
    assert layout_only == _MATRIX_LAYOUT_ONLY_PERMUTES
    assert absorbed == _MATRIX_ABSORBED_PERMUTES
    assert contract_entries == _MATRIX_CONTRACT_ENTRIES
    assert regressions == _MATRIX_RESIDUAL_REGRESSIONS
