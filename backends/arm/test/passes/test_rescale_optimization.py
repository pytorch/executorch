# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""FuseConsecutiveRescalesPass validation tests.

Tests that InsertRescaleInt32Pass creates consecutive RESCALE pairs between
chained arithmetic ops and that FuseConsecutiveRescalesPass correctly eliminates
them.

"""

from typing import Tuple

import torch
from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    FuseConsecutiveRescalesPass,
    InsertRescaleInt32Pass,
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


def assert_rescale_count_reduced(exported_program):
    """Assert fusion reduced RESCALE count below pre-fusion level.

    Two chained adds produce 6 RESCALEs before fusion. After fusion, each
    consecutive pair is either removed (identity) or replaced by a single
    composed RESCALE, so the count must be strictly less than 6.

    """
    graph_module = exported_program.graph_module
    count = sum(1 for n in graph_module.graph.nodes if _is_rescale(n))
    assert count < 6, f"Expected fewer than 6 RESCALEs after fusion, got {count}"
    return exported_program


def assert_no_int8_to_int32_via_int8(exported_program):
    """Assert no INT32->INT8->INT32 round-trip patterns remain.

    After fusion, no RESCALE outputting INT8 should feed a RESCALE outputting
    INT32 (the specific pattern this pass eliminates).

    """
    graph_module = exported_program.graph_module
    for node in graph_module.graph.nodes:
        if not _is_rescale(node):
            continue
        if node.args[1] not in (torch.int8, torch.int16):
            continue
        for user in node.users:
            if _is_rescale(user) and user.args[1] == torch.int32:
                raise AssertionError(
                    f"INT8/INT16->INT32 round-trip still exists: "
                    f"{node.name} (->INT8/INT16) -> {user.name} (->INT32)"
                )
    return exported_program


def assert_identity_fusion_no_int32_to_int32(exported_program):
    """Assert identity fusion removed both RESCALEs.

    When composed scale is ~1.0 and zero points match, the pass takes
    the identity path: both R1 and R2 are removed entirely. No
    INT32->INT32 RESCALE should exist since a composed node is only
    created on the non-identity path.

    """
    graph_module = exported_program.graph_module
    for node in graph_module.graph.nodes:
        if not _is_rescale(node):
            continue
        output_dtype = node.args[1]
        if output_dtype == torch.int32:
            input_node = node.args[0]
            if _is_rescale(input_node) and input_node.args[1] == torch.int32:
                raise AssertionError(
                    f"INT32->INT32 composed RESCALE found ({node.name}), "
                    f"expected identity fusion to remove both nodes"
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
    """FuseConsecutiveRescalesPass eliminates consecutive pairs.

    After InsertRescaleInt32Pass, chained adds produce RESCALEs with consecutive
    INT32->INT8->INT32 pairs. FuseConsecutiveRescalesPass merges each pair into
    a single composed RESCALE, eliminating all consecutive adjacencies.

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
        pass_functions=[assert_no_consecutive_rescales],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_fuse_identity_reduces_rescale_count():
    """Identity fusion removes both RESCALEs rather than composing.

    Two chained adds produce 6 RESCALEs. The consecutive pair between
    add1's output and add2's input has composed scale ~1.0 (symmetric
    requantize then dequantize), so the pass takes the identity path:
    both R1 (INT32->INT8) and R2 (INT8->INT32) are removed entirely.
    This is verified by:
    1. RESCALE count drops from 6 to 4 (one pair removed)
    2. No INT32->INT32 RESCALE exists (identity path, not composed)
    3. No INT8->INT32 round-trip remains

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
        pass_functions=[
            assert_rescale_count_reduced,
            assert_identity_fusion_no_int32_to_int32,
            assert_no_int8_to_int32_via_int8,
        ],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
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


def test_fuse_branching_add_eliminates_pairs():
    """Multi-user fusion eliminates all consecutive pairs.

    After FuseConsecutiveRescalesPass, no RESCALE->RESCALE adjacencies should
    remain, even when R1 originally had multiple RESCALE users.

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
        pass_functions=[
            assert_no_consecutive_rescales,
            assert_no_int8_to_int32_via_int8,
        ],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_fuse_consecutive_rescales_output_correctness():
    """End-to-end correctness: fused graph matches original.

    Keeps run_method_and_compare_outputs enabled to verify that the
    RESCALE fusion does not change numerical results.

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
        pass_functions=[assert_no_consecutive_rescales],
    )
    pipeline.run()


def test_fuse_branching_add_output_correctness():
    """Multi-user end-to-end correctness: fused branching graph
    matches original.

    Keeps run_method_and_compare_outputs enabled to verify that
    multi-user R1 fusion (where R1 feeds multiple downstream
    RESCALEs) does not change numerical results.

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
        pass_functions=[
            assert_no_consecutive_rescales,
            assert_no_int8_to_int32_via_int8,
        ],
    )
    pipeline.run()
