# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""FuseConsecutiveRescalesPass validation tests.

Tests that InsertRescaleInt32Pass creates consecutive RESCALE pairs between
chained arithmetic ops and that FuseConsecutiveRescalesPass correctly removes
identity pairs (composed scale ~1.0) while leaving non-identity pairs
unchanged.

Numerical difference sources covered:
  Source A: Fixed-point decomposition non-associativity (~1 INT8 step)
  Source B: INT8 clamping bypass on the identity removal path (0-1 steps
            observed; theoretical worst case ~120 steps near clamp boundary)

"""

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    FuseConsecutiveRescalesPass,
    InsertRescaleInt32Pass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.exir.dialects._ops import ops as exir_ops

RESCALE_OP = "executorch_exir_dialects_backend__ops_tosa_RESCALE_default"


# ============================================================================
# Toy Models
# ============================================================================


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
    """Multi-user R1: (x + y) feeds two downstream adds.

    After InsertRescaleInt32Pass, add1's output RESCALE (R1) feeds
    into both add2's and add3's input RESCALEs (R2, R3). The pass
    must fuse each R1->R2 and R1->R3 pair individually, removing R1
    only when all RESCALE users are fused away.

    """

    input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

    def forward(self, x, y, z, w):
        a = x + y  # add1: output RESCALE R1 has two RESCALE users
        b = a + z  # add2: input RESCALE R2 consumes R1
        c = a + w  # add3: input RESCALE R3 consumes R1
        return b + c

    @staticmethod
    def get_test_inputs():
        return (
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
            torch.randn(1, 3, 8, 8),
        )


# ============================================================================
# Assertion Functions (used via pass_functions, not pass_list)
# ============================================================================


def _is_rescale(node):
    """Check if a graph node is a TOSA RESCALE op."""
    return (
        node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    )


def assert_consecutive_rescales_exist(exported_program):
    """Assert at least one RESCALE->RESCALE adjacency exists."""
    graph_module = exported_program.graph_module
    rescale_info = []
    has_consecutive = False
    for node in graph_module.graph.nodes:
        if not _is_rescale(node):
            continue
        user_names = [u.name for u in node.users if _is_rescale(u)]
        rescale_info.append(f"{node.name} -> users: {user_names}")
        if user_names:
            has_consecutive = True

    assert has_consecutive, (
        "RESCALE nodes exist but no consecutive pattern found.\n"
        "RESCALE edges:\n" + "\n".join(f"  {info}" for info in rescale_info)
    )
    return exported_program


# Utility for debugging; not wired into pass_functions currently.
def assert_no_consecutive_rescales(exported_program):
    """Assert no RESCALE->RESCALE adjacency remains after fusion."""
    graph_module = exported_program.graph_module
    for node in graph_module.graph.nodes:
        if not _is_rescale(node):
            continue
        rescale_users = [u for u in node.users if _is_rescale(u)]
        assert not rescale_users, (
            f"Consecutive RESCALE pair still exists: "
            f"{node.name} -> {[u.name for u in rescale_users]}"
        )
    return exported_program


# ============================================================================
# Tests
# ============================================================================


def test_add_chain_rescale_count():
    """Two cascaded adds produce expected RESCALEs.

    Each add has 2 INT8 inputs (need INT8->INT32) and 1 INT32 output (need
    INT32->INT8), giving 3 RESCALEs per add = 6 total for two chained adds.

    """
    model = AddChain()
    pipeline = PassPipeline[AddChain.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        ops_not_before_pass=[RESCALE_OP],
        ops_after_pass={RESCALE_OP: 6},
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.run()


def test_add_chain_consecutive_rescales():
    """Consecutive RESCALE->RESCALE pattern exists between adds.

    add1's output RESCALE (INT32->INT8) feeds directly into add2's input RESCALE
    (INT8->INT32), creating a redundant round-trip.

    """
    model = AddChain()
    pipeline = PassPipeline[AddChain.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
        pass_functions=[assert_consecutive_rescales_exist],
    )
    pipeline.run()


def test_fuse_consecutive_rescales():
    """FuseConsecutiveRescalesPass removes identity pairs.

    After InsertRescaleInt32Pass, chained adds produce consecutive
    INT32->INT8->INT32 pairs.  The pass removes identity pairs (composed scale
    ~1.0) and leaves non-identity pairs unchanged. With random calibration data
    the composed scale is typically non-identity (~0.76), so most pairs are
    preserved.

    """
    model = AddChain()
    pipeline = PassPipeline[AddChain.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[
            FoldAndAnnotateQParamsPass,
            InsertRescaleInt32Pass,
            FuseConsecutiveRescalesPass,
        ],
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


def test_branching_add_precondition_consecutive_rescales():
    """Precondition check: InsertRescaleInt32Pass creates consecutive RESCALE pairs.

    BranchingAdd creates (x+y) which feeds both (a+z) and (a+w).
    After InsertRescaleInt32Pass, add1's output RESCALE R1 has two
    RESCALE users (R2 for add2, R3 for add3).  This test verifies
    that InsertRescaleInt32Pass produces these consecutive R1->R2
    and R1->R3 pairs; it does NOT run FuseConsecutiveRescalesPass.

    """
    model = BranchingAdd()
    pipeline = PassPipeline[BranchingAdd.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
        pass_functions=[assert_consecutive_rescales_exist],
    )
    pipeline.run()


def test_fuse_branching_add_only_identity_remaining():
    """Multi-user fusion: remaining consecutive pairs are identity.

    Verifies end-to-end output correctness (via qtol=1), not graph structure.

    BranchingAdd creates (x+y) which feeds two downstream adds.  After
    fusion, identity pairs are removed and non-identity pairs are left
    unchanged.  Any remaining consecutive pair that is still an R1->R2
    adjacency should have composed scale ~1.0 (identity).

    """
    model = BranchingAdd()
    pipeline = PassPipeline[BranchingAdd.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[
            FoldAndAnnotateQParamsPass,
            InsertRescaleInt32Pass,
            FuseConsecutiveRescalesPass,
        ],
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


# ---------------------------------------------------------------------------
# Deterministic graph-level tests for identity / non-identity behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "r1_scale,r2_scale",
    [
        (0.5, 2.0),
        (0.25, 4.0),
        (0.125, 8.0),
    ],
    ids=["0.5*2.0", "0.25*4.0", "0.125*8.0"],
)
def test_identity_pair_removed(r1_scale, r2_scale):
    """Identity RESCALE pairs (composed_scale == 1.0) are removed.

    Constructs a minimal graph:
        placeholder → R1(INT32→INT8, scale=r1) → R2(INT8→INT32, scale=r2) → output
    with r1*r2 == 1.0 exactly (power-of-2 scales).  The pass removes both
    RESCALEs and directly wires the placeholder to the output.

    Verifies:
    1. Both RESCALE nodes are removed
    2. The graph is marked as modified

    """
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r1 = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.RESCALE.default,
        args=(x, torch.int8, [r1_scale], 0, 0),
    )
    r2 = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.RESCALE.default,
        args=(r1, torch.int32, [r2_scale], 0, 0),
    )
    graph.output(r2)
    gm = torch.fx.GraphModule({}, graph)

    result = FuseConsecutiveRescalesPass().call(gm)

    rescale_count = sum(
        1
        for n in result.graph_module.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.backend.tosa.RESCALE.default
    )
    assert rescale_count == 0, (
        f"Identity pair (composed={r1_scale * r2_scale})"
        f" should be removed (0 RESCALEs), got {rescale_count}"
    )
    assert result.modified, "Graph should be modified for identity pair removal"


def test_identity_pair_with_nonzero_zp_removed():
    """Identity pair with matching non-zero zero points is removed.

    Same as test_identity_pair_removed but with non-zero zero points (zp=5/10)
    to verify the zero-point matching guards work correctly.

    """
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    # R1: INT32→INT8, scale=0.5, input_zp=5, output_zp=10
    r1 = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.RESCALE.default,
        args=(x, torch.int8, [0.5], 5, 10),
    )
    # R2: INT8→INT32, scale=2.0, input_zp=10 (matches r1 output_zp),
    #     output_zp=5 (matches r1 input_zp → identity guard)
    r2 = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.RESCALE.default,
        args=(r1, torch.int32, [2.0], 10, 5),
    )
    graph.output(r2)
    gm = torch.fx.GraphModule({}, graph)

    result = FuseConsecutiveRescalesPass().call(gm)

    rescale_count = sum(
        1
        for n in result.graph_module.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.backend.tosa.RESCALE.default
    )
    assert rescale_count == 0, (
        f"Identity pair with non-zero zp should be removed, "
        f"got {rescale_count} RESCALEs"
    )
    assert result.modified


def test_non_identity_pair_is_preserved():
    """Non-identity RESCALE pair (composed_scale != 1.0) is left unchanged.

    Constructs R1(scale=0.5) → R2(scale=3.0) with composed=1.5 (non- identity).
    The pass leaves this pair unchanged because Vela cannot handle INT32→INT32.

    """
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    r1 = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.RESCALE.default,
        args=(x, torch.int8, [0.5], 0, 0),
    )
    r2 = graph.create_node(
        "call_function",
        exir_ops.backend.tosa.RESCALE.default,
        args=(r1, torch.int32, [3.0], 0, 0),
    )
    graph.output(r2)
    gm = torch.fx.GraphModule({}, graph)

    result = FuseConsecutiveRescalesPass().call(gm)

    assert (
        not result.modified
    ), "Non-identity pair (composed=1.5) should NOT be fused (Vela limitation)"
    rescale_nodes = [
        n
        for n in result.graph_module.graph.nodes
        if n.op == "call_function" and n.target == exir_ops.backend.tosa.RESCALE.default
    ]
    assert (
        len(rescale_nodes) == 2
    ), f"Expected 2 RESCALEs (non-identity preserved), got {len(rescale_nodes)}"


# ---------------------------------------------------------------------------
# End-to-end correctness tests (via PassPipeline)
# ---------------------------------------------------------------------------


def test_fuse_consecutive_rescales_output_correctness():
    """End-to-end correctness: optimized graph matches original within
    qtol=1.

    Identity pairs are removed (bypassing the INT8 clamp); non-identity
    pairs are left unchanged.  Output differences arise from bypassing
    the intermediate INT8 clamp (Source B), typically 0-1 INT8 steps.

    """
    model = AddChain()
    pipeline = PassPipeline[AddChain.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[
            FoldAndAnnotateQParamsPass,
            InsertRescaleInt32Pass,
            FuseConsecutiveRescalesPass,
        ],
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


def test_fuse_branching_add_output_correctness():
    """Multi-user end-to-end correctness: optimized branching graph
    matches original within qtol=1.

    Identity pairs are removed; non-identity pairs are left unchanged.
    Output differences from bypassing the INT8 clamp (Source B) are
    typically 0-1 INT8 steps.

    """
    model = BranchingAdd()
    pipeline = PassPipeline[BranchingAdd.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[
            FoldAndAnnotateQParamsPass,
            InsertRescaleInt32Pass,
            FuseConsecutiveRescalesPass,
        ],
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


# ============================================================================
# LSTM-like pattern tests (sigmoid→mul, tanh→mul → add chains)
# ============================================================================


class LSTMGatePattern(torch.nn.Module):
    """Mimics LSTM cell-state update: f*c_prev + i*g.

    sigmoid and tanh produce gate values; mul applies them; add
    combines.  After InsertRescaleInt32Pass, the mul->add boundaries
    create consecutive RESCALE pairs.  FuseConsecutiveRescalesPass
    removes identity pairs and leaves non-identity pairs unchanged.
    This tests the pattern that originally caused all-zero EthosU55
    outputs (fixed by identity-only removal instead of INT32->INT32
    fusion).

    """

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


def test_lstm_gate_pattern_consecutive_rescales():
    """LSTM gate pattern creates consecutive RESCALE pairs at mul→add
    boundaries.
    """
    model = LSTMGatePattern()
    pipeline = PassPipeline[LSTMGatePattern.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
        pass_functions=[assert_consecutive_rescales_exist],
    )
    pipeline.run()


def test_lstm_gate_pattern_fuse_correctness():
    """LSTM gate pattern end-to-end correctness after fusion.

    Verifies that FuseConsecutiveRescalesPass produces correct output for the
    sigmoid→mul + tanh→mul → add chain that LSTM uses.  This isolates the TOSA
    graph correctness (no Vela/FVP needed).

    """
    model = LSTMGatePattern()
    pipeline = PassPipeline[LSTMGatePattern.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[
            FoldAndAnnotateQParamsPass,
            InsertRescaleInt32Pass,
            FuseConsecutiveRescalesPass,
        ],
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


def test_lstm_gate_pattern_full_tosa_pipeline():
    """LSTM gate pattern through the full TOSA INT pipeline.

    Runs through the complete TOSA lowering pipeline (including
    InsertTableOpsPass, DecomposeQuantNodesPass, and all other downstream passes
    that call super().call()). This tests whether the optimized graph (with
    identity pairs removed) survives the full pipeline correctly.

    """
    from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT

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
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=model.get_test_inputs(),
        atol=3e-1,
        qtol=1,
    )
    pipeline.run()


def test_actual_lstm_tosa_INT():
    """Run the ACTUAL quantizable LSTM model through TosaPipelineINT.

    This is the same model used by test_lstm_u55_INT but run against the TOSA
    reference model (no FVP needed). If this fails, the issue is in the TOSA
    graph generation with FuseConsecutiveRescalesPass, not Vela/FVP.

    """
    from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
    from torch.nn.quantizable.modules import rnn

    lstm = rnn.LSTM(10, 20, 2)
    lstm = lstm.eval()

    input_t = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

    def get_test_inputs():
        return (
            torch.randn(5, 3, 10),
            (torch.randn(2, 3, 20), torch.randn(2, 3, 20)),
        )

    model_example_inputs = get_test_inputs()

    pipeline = TosaPipelineINT[input_t](
        lstm,
        model_example_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_test_inputs(),
        atol=3e-1,
        qtol=1,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_actual_lstm_u55_INT():
    """Run the ACTUAL quantizable LSTM model through EthosU55PipelineINT.

    This is the exact same test as test_lstm_u55_INT from test_lstm_arm.py.
    Verifies the LSTM produces correct (non-zero) output on Ethos-U55 FVP after
    FuseConsecutiveRescalesPass removes identity RESCALE pairs.

    """
    from executorch.backends.arm.test.tester.test_pipeline import EthosU55PipelineINT
    from torch.nn.quantizable.modules import rnn

    lstm = rnn.LSTM(10, 20, 2)
    lstm = lstm.eval()

    input_t = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

    def get_test_inputs():
        return (
            torch.randn(5, 3, 10),
            (torch.randn(2, 3, 20), torch.randn(2, 3, 20)),
        )

    model_example_inputs = get_test_inputs()

    pipeline = EthosU55PipelineINT[input_t](
        lstm,
        model_example_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=get_test_inputs(),
        atol=3e-1,
        qtol=1,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_lstm_gate_pattern_u55_INT():
    """Run the LSTM gate pattern model through EthosU55PipelineINT.

    Smaller than the full LSTM, this tests if the specific sigmoid->mul->add
    pattern produces correct output on the Ethos-U55 NPU after
    FuseConsecutiveRescalesPass removes identity RESCALE pairs.

    """
    from executorch.backends.arm.test.tester.test_pipeline import EthosU55PipelineINT

    model = LSTMGatePattern()

    pipeline = EthosU55PipelineINT[LSTMGatePattern.input_t](
        model,
        model.get_test_inputs(),
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs",
        inputs=model.get_test_inputs(),
        atol=3e-1,
        qtol=1,
    )
    pipeline.run()
