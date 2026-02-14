# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for the ArmPassManager pass insertion APIs:
- insert_passes_before
- insert_passes_after
- _apply_pass_insertions
"""

import unittest

from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class DummyPass(ExportPass):
    """A dummy pass for testing insertion APIs."""

    def __init__(self, name: str = "dummy") -> None:
        super().__init__()
        self.name = name

    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)


class TargetPassA(ExportPass):
    """Target pass A for insertion tests."""

    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)


class TargetPassB(ExportPass):
    """Target pass B for insertion tests."""

    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)


class InsertedPassX(ExportPass):
    """Pass to be inserted before/after target passes."""

    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)


class InsertedPassY(ExportPass):
    """Another pass to be inserted before/after target passes."""

    def call(self, graph_module: GraphModule) -> PassResult:
        return PassResult(graph_module, False)


def create_test_pass_manager() -> ArmPassManager:
    """Create a test ArmPassManager with a minimal compile spec."""
    compile_spec = ArmCompileSpec(
        TosaSpecification.create_from_string("TOSA-0.80+BI")
    )
    return ArmPassManager(compile_spec)


class TestInsertPassesBefore(unittest.TestCase):
    """Tests for insert_passes_before API."""

    def test_insert_single_pass_before_target(self) -> None:
        """Test inserting a single pass before a target pass."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_before(TargetPassA, [InsertedPassX()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types, [InsertedPassX, TargetPassA, TargetPassB]
        )

    def test_insert_multiple_passes_before_target(self) -> None:
        """Test inserting multiple passes before a target pass."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_before(TargetPassB, [InsertedPassX(), InsertedPassY()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types, [TargetPassA, InsertedPassX, InsertedPassY, TargetPassB]
        )

    def test_insert_before_nonexistent_target_no_effect(self) -> None:
        """Test that inserting before a nonexistent target has no effect."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())

        pm.insert_passes_before(TargetPassB, [InsertedPassX()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(pass_types, [TargetPassA])


class TestInsertPassesAfter(unittest.TestCase):
    """Tests for insert_passes_after API."""

    def test_insert_single_pass_after_target(self) -> None:
        """Test inserting a single pass after a target pass."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_after(TargetPassA, [InsertedPassX()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types, [TargetPassA, InsertedPassX, TargetPassB]
        )

    def test_insert_multiple_passes_after_target(self) -> None:
        """Test inserting multiple passes after a target pass."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_after(TargetPassA, [InsertedPassX(), InsertedPassY()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types, [TargetPassA, InsertedPassX, InsertedPassY, TargetPassB]
        )

    def test_insert_after_last_pass(self) -> None:
        """Test inserting a pass after the last pass in the list."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_after(TargetPassB, [InsertedPassX()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types, [TargetPassA, TargetPassB, InsertedPassX]
        )


class TestApplyPassInsertions(unittest.TestCase):
    """Tests for _apply_pass_insertions behavior."""

    def test_apply_insertions_only_once(self) -> None:
        """Test that _apply_pass_insertions is idempotent."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.insert_passes_before(TargetPassA, [InsertedPassX()])

        pm._apply_pass_insertions()
        first_count = len(pm.passes)

        pm._apply_pass_insertions()
        second_count = len(pm.passes)

        self.assertEqual(first_count, second_count)
        self.assertEqual(first_count, 2)

    def test_no_insertions_no_effect(self) -> None:
        """Test that _apply_pass_insertions with no insertions has no effect."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        original_passes = list(pm.passes)
        pm._apply_pass_insertions()

        self.assertEqual(pm.passes, original_passes)

    def test_combined_before_and_after_insertions(self) -> None:
        """Test combining before and after insertions on the same target."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_before(TargetPassA, [InsertedPassX()])
        pm.insert_passes_after(TargetPassA, [InsertedPassY()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types,
            [InsertedPassX, TargetPassA, InsertedPassY, TargetPassB],
        )

    def test_insertions_on_different_targets(self) -> None:
        """Test insertions on multiple different target passes."""
        pm = create_test_pass_manager()

        pm.add_pass(TargetPassA())
        pm.add_pass(TargetPassB())

        pm.insert_passes_before(TargetPassA, [InsertedPassX()])
        pm.insert_passes_before(TargetPassB, [InsertedPassY()])
        pm._apply_pass_insertions()

        pass_types = [type(p) for p in pm.passes]
        self.assertEqual(
            pass_types,
            [InsertedPassX, TargetPassA, InsertedPassY, TargetPassB],
        )


if __name__ == "__main__":
    unittest.main()
