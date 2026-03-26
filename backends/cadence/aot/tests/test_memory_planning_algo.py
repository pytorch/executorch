# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torch.fx
from executorch.backends.cadence.aot.memory_constraints import MemConstraints
from executorch.backends.cadence.aot.memory_planning import (
    PositionBasedGreedyWithHierarchy,
)
from executorch.backends.cadence.aot.memory_planning_algo import (
    InvalidPinnedMemId,
    MemoryPlanningState,
)
from executorch.backends.cadence.aot.utils import MemoryConfig
from executorch.exir.tensor import TensorSpec


def _make_spec(shape: list[int], *, mem_id: int | None = None) -> TensorSpec:
    """Create a TensorSpec for a uint8 tensor of given shape, optionally pre-pinning mem_id."""
    spec = TensorSpec(dtype=torch.uint8, shape=torch.Size(shape))
    # The planner's overlap checker requires valid lifetimes on every spec.
    spec.lifetime = [0, 1]
    if mem_id is not None:
        spec.mem_id = mem_id
    return spec


def _make_algo_and_state(
    mem_sizes: list[int],
) -> tuple[PositionBasedGreedyWithHierarchy, MemoryPlanningState, MemConstraints]:
    """Build a 2-memory config planner (mem_id 1 = fast, 2 = slow) for tests."""
    config = MemoryConfig(mem_sizes)
    algo = PositionBasedGreedyWithHierarchy(config)
    state = MemoryPlanningState(config)
    constraints = MemConstraints()
    return algo, state, constraints


class TestPinnedMemIdPromotion(unittest.TestCase):
    """Tests for plan_with_constraints pre-set mem_id → AbsolutePlacementConstraint promotion."""

    def _run(
        self,
        specs: list[TensorSpec],
        mem_sizes: list[int],
    ) -> None:
        algo, state, constraints = _make_algo_and_state(mem_sizes)
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        algo.plan_with_constraints(
            specs, gm, None, state, constraints  # pyre-ignore[6]
        )

    def test_spec_without_preset_mem_id_planned_freely(self) -> None:
        """A spec with no pre-set mem_id is placed by the greedy algo in mem_id=1."""
        spec = _make_spec([512])
        self._run([spec], mem_sizes=[1024, 1024])
        self.assertIsNotNone(spec.mem_id)
        self.assertEqual(spec.mem_id, 1)
        self.assertIsNotNone(spec.mem_offset)

    def test_spec_with_preset_mem_id_stays_in_that_memory(self) -> None:
        """A spec with pre-set mem_id=2 stays in memory 2 even though memory 1 is faster."""
        spec = _make_spec([256])
        spec.mem_id = 2
        self._run([spec], mem_sizes=[4096, 4096])
        # mem_id must be preserved as 2
        self.assertEqual(spec.mem_id, 2)
        # Must have a valid offset assigned
        assert spec.mem_offset is not None
        assert spec.mem_offset >= 0

    def test_preset_mem_id_offset_computed_by_planner(self) -> None:
        """Two specs pinned to mem_id=2 get distinct non-overlapping offsets."""
        spec_a = _make_spec([100])
        spec_b = _make_spec([200])
        spec_a.mem_id = 2
        spec_b.mem_id = 2
        self._run([spec_a, spec_b], mem_sizes=[4096, 4096])
        self.assertEqual(spec_a.mem_id, 2)
        self.assertEqual(spec_b.mem_id, 2)
        # Offsets must not overlap: [a_start, a_end) ∩ [b_start, b_end) == ∅
        a_end = spec_a.mem_offset + spec_a.allocated_memory
        b_end = spec_b.mem_offset + spec_b.allocated_memory
        no_overlap = spec_a.mem_offset >= b_end or spec_b.mem_offset >= a_end
        self.assertTrue(no_overlap, f"Specs overlap: {spec_a} and {spec_b}")

    def test_unpinned_spec_unaffected_by_pinned_peers(self) -> None:
        """Specs without pre-set mem_id are not forced into the pinned tier."""
        pinned = _make_spec([128])
        pinned.mem_id = 2
        free = _make_spec([64])  # No preset; greedy should pick mem_id=1
        self._run([pinned, free], mem_sizes=[4096, 4096])
        self.assertEqual(pinned.mem_id, 2)
        # Greedy algo prefers mem_id=1 (faster) for unconstrained specs
        self.assertEqual(free.mem_id, 1)

    def test_already_constrained_spec_not_overridden(self) -> None:
        """A spec that already has an AbsolutePlacementConstraint is not double-promoted."""
        from executorch.backends.cadence.aot.memory_constraints import (
            AbsolutePlacementConstraint,
        )

        spec = _make_spec([256])
        spec.mem_id = 1  # will be set but constraint added externally to mem_id=2

        algo, state, constraints = _make_algo_and_state([4096, 4096])
        # Add an explicit constraint to mem_id=2 (overrides the spec.mem_id=1 preset)
        constraints.set_absolute_placement_constraint(
            spec, AbsolutePlacementConstraint(pinned_memory_id=2)
        )
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        algo.plan_with_constraints(
            [spec], gm, None, state, constraints  # pyre-ignore[6]
        )
        # The existing constraint (mem_id=2) takes precedence over spec.mem_id=1
        self.assertEqual(spec.mem_id, 2)

    def test_mem_id_zero_raises(self) -> None:
        """mem_id=0 is reserved by ExecuTorch and should raise InvalidPinnedMemId."""
        spec = _make_spec([512], mem_id=0)
        with self.assertRaises(InvalidPinnedMemId):
            self._run([spec], mem_sizes=[1024, 1024])

    def test_mem_id_out_of_range_raises(self) -> None:
        """A spec with mem_id >= num_memories should raise InvalidPinnedMemId."""
        # With 2 memory tiers, valid mem_ids are 1 and 2; mem_id=3 is out of range.
        spec = _make_spec([256], mem_id=3)
        with self.assertRaises(InvalidPinnedMemId):
            self._run([spec], mem_sizes=[4096, 4096])

    def test_mem_id_negative_raises(self) -> None:
        """A spec with negative mem_id should raise InvalidPinnedMemId."""
        spec = _make_spec([256])
        spec.mem_id = -1
        with self.assertRaises(InvalidPinnedMemId):
            self._run([spec], mem_sizes=[1024, 1024])

