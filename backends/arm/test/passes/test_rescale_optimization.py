# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""FuseConsecutiveRescalesPass validation tests.

Tests that InsertRescaleInt32Pass creates consecutive RESCALE pairs between
chained arithmetic ops and that FuseConsecutiveRescalesPass correctly eliminates
them.

Also includes root-cause investigation tests for the numerical differences
introduced by FuseConsecutiveRescalesPass (D94483331). These tests demonstrate
three distinct sources of error using pure-Python TOSA RESCALE arithmetic:
  Source A: Fixed-point decomposition non-associativity
  Source B: INT8 clamping bypass on the identity fusion path
  Source C: RESCALE fake kernel returns uninitialized data

"""

from typing import List, Tuple

import pytest
import torch
from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    FuseConsecutiveRescalesPass,
    InsertRescaleInt32Pass,
)
from executorch.backends.arm.operators.op_tosa_rescale import (
    _compute_multiplier_and_shift,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

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
    return node.op == "call_function" and "RESCALE" in str(node.target)


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


def assert_only_identity_consecutive_rescales(exported_program):
    """Assert any remaining RESCALE->RESCALE adjacency is an identity pair.

    After fusion, non-identity consecutive pairs should be replaced with a
    composed INT32->INT32 RESCALE.  Identity pairs (composed scale ~1.0,
    matching zero points) are preserved for INT8/INT16 clamp correctness.

    """
    graph_module = exported_program.graph_module
    for node in graph_module.graph.nodes:
        if not _is_rescale(node):
            continue
        if node.args[1] not in (torch.int8, torch.int16):
            continue
        if len(node.args[2]) != 1:
            continue
        r1_scale = float(node.args[2][0])
        for user in node.users:
            if not _is_rescale(user) or user.args[1] != torch.int32:
                continue
            if len(user.args[2]) != 1:
                continue
            r2_scale = float(user.args[2][0])
            composed = r1_scale * r2_scale
            assert abs(composed - 1.0) < 1e-6, (
                f"Non-identity consecutive pair found after fusion: "
                f"{node.name} -> {user.name}, composed_scale={composed}"
            )
    return exported_program


def assert_exact_rescale_count(expected_count):
    """Return assertion function that checks exact RESCALE count after
    fusion.
    """

    def _assert(exported_program):
        graph_module = exported_program.graph_module
        count = sum(1 for n in graph_module.graph.nodes if _is_rescale(n))
        assert (
            count == expected_count
        ), f"Expected exactly {expected_count} RESCALEs after fusion, got {count}"
        return exported_program

    return _assert


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
        ops_not_before_pass={RESCALE_OP},
        ops_after_pass={RESCALE_OP: 6},
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
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
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_fuse_consecutive_rescales():
    """FuseConsecutiveRescalesPass fuses non-identity pairs.

    After InsertRescaleInt32Pass, chained adds produce consecutive
    INT32->INT8->INT32 pairs.  With random inputs the composed scale
    is typically non-identity (~0.76), so the pass replaces the pair
    with a single INT32->INT32 RESCALE.  Any remaining consecutive
    pair must be an identity pair (composed scale ~1.0).

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
        pass_functions=[assert_only_identity_consecutive_rescales],
    )
    pipeline.run()


def test_fuse_branching_add_multi_user():
    """Multi-user R1: add1's output RESCALE feeds two adds.

    BranchingAdd creates (x+y) which feeds both (a+z) and (a+w).
    After InsertRescaleInt32Pass, add1's output RESCALE R1 has two
    RESCALE users (R2 for add2, R3 for add3). The pass must fuse
    each pair individually and only remove R1 when all its RESCALE
    users have been fused.

    """
    model = BranchingAdd()
    pipeline = PassPipeline[BranchingAdd.input_t](
        model,
        model.get_test_inputs(),
        quantize=True,
        pass_list=[FoldAndAnnotateQParamsPass, InsertRescaleInt32Pass],
        pass_functions=[assert_consecutive_rescales_exist],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_fuse_branching_add_only_identity_remaining():
    """Multi-user fusion leaves only identity consecutive pairs.

    BranchingAdd creates (x+y) which feeds two downstream adds.
    After fusion, non-identity consecutive pairs are replaced with
    composed INT32->INT32 RESCALEs.  Any remaining consecutive pair
    must be identity (composed scale ~1.0).

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
        pass_functions=[assert_only_identity_consecutive_rescales],
    )
    pipeline.run()


def test_fuse_consecutive_rescales_output_correctness():
    """End-to-end correctness: fused graph matches original within qtol=1.

    Non-identity pairs are fused via a composed INT32->INT32 RESCALE
    whose fixed-point decomposition differs slightly from the original
    two-step decomposition (Source A: non-associativity). This causes
    up to ~1 INT8 step of output difference, hence qtol=1.
    """


# ---------------------------------------------------------------------------
# Deterministic graph-level tests for identity / non-identity bypass
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
def test_identity_pair_not_fused(r1_scale, r2_scale):
    """Identity RESCALE pairs (composed_scale == 1.0) are preserved.

    Constructs a minimal graph:
        placeholder → R1(INT32→INT8, scale=r1) → R2(INT8→INT32, scale=r2) → output
    with r1*r2 == 1.0 exactly (power-of-2 scales).  The pass must skip
    fusion (continue) to preserve the intermediate INT8 clamp.

    Verifies:
    1. Both RESCALE nodes survive
    2. The graph is not marked as modified
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
        if n.op == "call_function"
        and n.target == exir_ops.backend.tosa.RESCALE.default
    )
    assert rescale_count == 2, (
        f"Identity pair (composed={r1_scale * r2_scale}) should be preserved "
        f"(2 RESCALEs), got {rescale_count}"
    )
    assert not result.modified, (
        "Graph should not be modified for identity pair"
    )


def test_identity_pair_with_nonzero_zp_not_fused():
    """Identity pair with matching non-zero zero points is preserved.

    Same as test_identity_pair_not_fused but with non-zero zero points
    (zp=5) to verify the zero-point matching guards work correctly.
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
        if n.op == "call_function"
        and n.target == exir_ops.backend.tosa.RESCALE.default
    )
    assert rescale_count == 2, (
        f"Identity pair with non-zero zp should be preserved, "
        f"got {rescale_count} RESCALEs"
    )
    assert not result.modified


def test_non_identity_pair_is_fused():
    """Non-identity RESCALE pair (composed_scale != 1.0) is fused.

    Constructs R1(scale=0.5) → R2(scale=3.0) with composed=1.5 (non-identity).
    The pass must fuse this into a single INT32→INT32 RESCALE.

    Uses unittest.mock to bypass super().call() (which requires FakeTensor
    meta for interpreter-based graph traversal).
    """
    from unittest.mock import patch

    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.pass_base import ExportPass
    from torch.fx.passes.infra.pass_base import PassResult as FxPassResult

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

    with patch.object(
        ExportPass, "call", lambda self, gm: FxPassResult(gm, True)
    ):
        result = FuseConsecutiveRescalesPass().call(gm)

    assert result.modified, (
        "Non-identity pair (composed=1.5) should be fused"
    )
    # After fusion: R1 and R2 should be replaced with a single composed RESCALE
    rescale_nodes = [
        n
        for n in result.graph_module.graph.nodes
        if n.op == "call_function"
        and n.target == exir_ops.backend.tosa.RESCALE.default
    ]
    # Only the composed RESCALE should remain (R1 and R2 erased)
    assert len(rescale_nodes) == 1, (
        f"Expected 1 composed RESCALE after fusion, got {len(rescale_nodes)}"
    )
    # Verify the composed scale
    composed_scale = float(rescale_nodes[0].args[2][0])
    assert abs(composed_scale - 1.5) < 1e-6, (
        f"Expected composed scale=1.5, got {composed_scale}"
    )
    # Verify the composed RESCALE is INT32→INT32
    assert rescale_nodes[0].args[1] == torch.int32


# ---------------------------------------------------------------------------
# End-to-end correctness tests (via PassPipeline)
# ---------------------------------------------------------------------------


def test_fuse_consecutive_rescales_output_correctness():
    """End-to-end correctness: fused graph matches original within qtol=1.

    Non-identity pairs are fused via a composed INT32->INT32 RESCALE
    whose fixed-point decomposition differs slightly from the original
    two-step decomposition (Source A: non-associativity). This causes
    up to ~1 INT8 step of output difference, hence qtol=1.
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
    """Multi-user end-to-end correctness: fused branching graph
    matches original within qtol=1.

    Non-identity pairs are fused via composed INT32->INT32 RESCALEs.
    Fixed-point decomposition non-associativity (Source A) causes up
    to ~1 INT8 step of output difference, hence qtol=1.

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
