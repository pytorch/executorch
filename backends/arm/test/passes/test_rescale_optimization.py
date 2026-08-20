# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import FuseConsecutiveRescalesPass
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    TosaPipelineINT,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops

RESCALE_TARGET = exir_ops.backend.tosa.RESCALE.default
TOSA_INT_SPEC = TosaSpecification.create_from_string("TOSA-1.0+INT")


class AddChain(torch.nn.Module):
    """Two cascaded adds: (x + y) + z."""

    input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def forward(self, x, y, z):
        return (x + y) + z

    @staticmethod
    def get_test_inputs():
        return (
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
        )


class BranchingAdd(torch.nn.Module):
    """(x + y) feeds two downstream adds."""

    input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def forward(self, x, y, z, w):
        a = x + y
        b = a + z
        c = a + w
        return b + c

    @staticmethod
    def get_test_inputs():
        return (
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
        )


class LSTMGatePattern(torch.nn.Module):
    """Mimics the LSTM cell-state update f * c_prev + i * g."""

    input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def forward(self, forget_gate_in, cell_prev, input_gate_in, candidate):
        f = torch.sigmoid(forget_gate_in)
        i = torch.sigmoid(input_gate_in)
        g = torch.tanh(candidate)
        return f * cell_prev + i * g

    @staticmethod
    def get_test_inputs():
        return (
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
        )


def _is_rescale(node) -> bool:
    return node.op == "call_function" and node.target == RESCALE_TARGET


def _count_rescales(graph_module: torch.fx.GraphModule) -> int:
    return sum(1 for node in graph_module.graph.nodes if _is_rescale(node))


def _run_fuse_pass(graph_module: torch.fx.GraphModule):
    with TosaLoweringContext(TOSA_INT_SPEC):
        return FuseConsecutiveRescalesPass().call(graph_module)


def _make_int32_placeholder(graph: torch.fx.Graph, name: str = "x") -> torch.fx.Node:
    x = graph.placeholder(name)
    x.meta["val"] = torch.ones((1,), dtype=torch.int32)
    return x


def _make_rescale(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    output_dtype: torch.dtype,
    scale: float,
    input_zp: int = 0,
    output_zp: int = 0,
) -> torch.fx.Node:
    return graph.create_node(
        "call_function",
        RESCALE_TARGET,
        args=(input_node, output_dtype, [scale], input_zp, output_zp),
    )


@pytest.mark.parametrize(
    "r1_scale,r2_scale",
    [(0.5, 2.0), (0.25, 4.0), (0.125, 8.0)],
    ids=["0.5x2.0", "0.25x4.0", "0.125x8.0"],
)
def test_fuse_consecutive_rescales_tosa_INT_identity_pair_removed(
    r1_scale: float, r2_scale: float
) -> None:
    """Checks that an identity RESCALE pair is removed."""
    graph = torch.fx.Graph()
    x = _make_int32_placeholder(graph)
    r1 = _make_rescale(graph, x, torch.int8, r1_scale)
    r2 = _make_rescale(graph, r1, torch.int32, r2_scale)
    graph.output(r2)
    graph_module = torch.fx.GraphModule({}, graph)
    rescale_count_before = _count_rescales(graph_module)

    result = _run_fuse_pass(graph_module)

    assert rescale_count_before == 2
    assert result.modified
    assert _count_rescales(result.graph_module) == 0


def test_fuse_consecutive_rescales_tosa_INT_non_identity_pair_preserved() -> None:
    """Checks that a non-identity RESCALE pair is left unchanged."""
    graph = torch.fx.Graph()
    x = _make_int32_placeholder(graph)
    r1 = _make_rescale(graph, x, torch.int8, 0.5)
    r2 = _make_rescale(graph, r1, torch.int32, 3.0)
    graph.output(r2)
    graph_module = torch.fx.GraphModule({}, graph)
    rescale_count_before = _count_rescales(graph_module)

    result = _run_fuse_pass(graph_module)

    assert not result.modified
    assert _count_rescales(result.graph_module) == rescale_count_before


def test_fuse_consecutive_rescales_tosa_INT_zero_point_mismatch_preserved() -> None:
    """Checks that mismatched zero points prevent fusion."""
    graph = torch.fx.Graph()
    x = _make_int32_placeholder(graph)
    r1 = _make_rescale(graph, x, torch.int8, 0.5, input_zp=0, output_zp=3)
    r2 = _make_rescale(graph, r1, torch.int32, 2.0, input_zp=4, output_zp=0)
    graph.output(r2)
    graph_module = torch.fx.GraphModule({}, graph)
    rescale_count_before = _count_rescales(graph_module)

    result = _run_fuse_pass(graph_module)

    assert not result.modified
    assert _count_rescales(result.graph_module) == rescale_count_before


def test_fuse_consecutive_rescales_tosa_INT_shared_producer_all_rescale_users_removed() -> (
    None
):
    """Checks that a shared producer is removed when all users are fuseable."""
    graph = torch.fx.Graph()
    x = _make_int32_placeholder(graph)
    r1 = _make_rescale(graph, x, torch.int8, 0.5)
    r2_left = _make_rescale(graph, r1, torch.int32, 2.0)
    r2_right = _make_rescale(graph, r1, torch.int32, 2.0)
    graph.output((r2_left, r2_right))
    graph_module = torch.fx.GraphModule({}, graph)
    rescale_count_before = _count_rescales(graph_module)

    result = _run_fuse_pass(graph_module)

    assert rescale_count_before == 3
    assert result.modified
    assert _count_rescales(result.graph_module) == 0


def test_fuse_consecutive_rescales_tosa_INT_shared_producer_non_rescale_user_preserved() -> (
    None
):
    """Checks that a shared producer is kept for remaining non-RESCALE users."""
    graph = torch.fx.Graph()
    x = _make_int32_placeholder(graph)
    r1 = _make_rescale(graph, x, torch.int8, 0.5)
    r2 = _make_rescale(graph, r1, torch.int32, 2.0)
    graph.output((r2, r1))
    graph_module = torch.fx.GraphModule({}, graph)
    rescale_count_before = _count_rescales(graph_module)

    result = _run_fuse_pass(graph_module)

    assert rescale_count_before == 2
    assert result.modified
    assert _count_rescales(result.graph_module) == 1
    remaining_rescales = [
        node for node in result.graph_module.graph.nodes if _is_rescale(node)
    ]
    assert remaining_rescales == [r1]


def test_fuse_consecutive_rescales_tosa_INT_lstm_gate_pattern_pipeline() -> None:
    """Checks the LSTM-shaped regression path in the TOSA INT pipeline."""
    model = LSTMGatePattern()
    pipeline = TosaPipelineINT[LSTMGatePattern.input_t](
        model,
        model.get_test_inputs(),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_fuse_consecutive_rescales_u55_INT_lstm_gate_pattern_pipeline() -> None:
    """Checks the LSTM-shaped regression path in the U55 INT pipeline."""
    model = LSTMGatePattern()
    pipeline = EthosU55PipelineINT[LSTMGatePattern.input_t](
        model,
        model.get_test_inputs(),
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
