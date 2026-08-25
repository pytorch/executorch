# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.banked_memory_planning import (
    Bank,
    banked_memory_planning_pass,
    BankedGreedy,
    BankPlacementError,
    format_placement_report,
    TargetMemoryMap,
)
from executorch.exir.memory_planning import (
    collect_specs_from_nodes,
    greedy,
    materialize_buffer,
    MemoryPlanningAlgorithmSuite,
    pick_shared_obj,
    update_all_tensors_lifetime,
    Verifier,
)
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import MemoryPlanningPass, SpecPropPass, ToOutVarPass
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from functorch.experimental.control_flow import map as torch_map
from torch.export import export
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule

try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


KiB = 1024
MiB = 1024 * 1024


def make_spec(nbytes: int, lifetime: Tuple[int, int]) -> TensorSpec:
    """A uint8 spec of exactly nbytes, with an explicit lifetime."""
    spec = TensorSpec(dtype=torch.uint8, shape=[nbytes])
    spec.lifetime = [lifetime[0], lifetime[1]]
    return spec


def empty_graph_module() -> GraphModule:
    return GraphModule(torch.nn.Module(), torch.fx.Graph())


class ToyModelForBankPlanning(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        o = a
        for _ in range(10):
            o = o * a
            o = o + b
        return o

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(10), torch.randn(10))


def prepare_toy_model() -> Tuple[GraphModule, ExportGraphSignature]:
    model = ToyModelForBankPlanning()
    edge = to_edge(export(model, model.get_random_inputs(), strict=True))
    gm = edge.exported_program().graph_module
    gs = edge.exported_program().graph_signature
    gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
    # Give the alloc nodes their specs, as MemoryPlanningPass does before
    # planning; without this the graph exposes only its boundary tensors.
    MemoryPlanningPass()._set_alloc_node_spec(gm)
    update_all_tensors_lifetime(gm, gs)
    return gm, gs


def flat_map(size: int = 1 << 20) -> TargetMemoryMap:
    return TargetMemoryMap([Bank(name="sram", size=size, mem_id=1)])


def two_banks(fast: int, slow: int) -> TargetMemoryMap:
    return TargetMemoryMap(
        [Bank(name="fast", size=fast, mem_id=1), Bank(name="slow", size=slow, mem_id=2)]
    )


def plan(
    planner: BankedGreedy,
    specs: List[TensorSpec],
    graph_module: GraphModule = None,
    extra_padding: int = 0,
) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    result = planner(
        16, specs, graph_module or empty_graph_module(), None, extra_padding
    )
    return result.bufsizes, {
        id(spec): (alloc.mem_id, alloc.mem_offset)
        for spec, alloc in result.spec_dict.items()
    }


def packed_size(specs: Sequence[TensorSpec]) -> int:
    """Pack specs the way the planner does and return the bytes used."""
    objects = []
    ordered = sorted(specs, key=lambda s: s.allocated_memory, reverse=True)
    for spec in ordered:
        pick_shared_obj(objects, spec, True)
    return materialize_buffer(objects)


def assert_no_free_readmission(
    test: unittest.TestCase,
    target: TargetMemoryMap,
    specs: Sequence[TensorSpec],
    alloc: Mapping[int, Tuple[int, int]],
) -> None:
    """No spec may sit in a slow bank when a faster one would have taken it free.

    This is the invariant that a rank-only eviction rule violates: evicting a
    buffer that shares a shared object frees no bytes, so it gets stranded in
    slow memory for nothing.
    """
    by_bank: Dict[int, List[TensorSpec]] = {m: [] for m in target.mem_ids()}
    for spec in specs:
        by_bank[alloc[id(spec)][0]].append(spec)

    for mem_id in target.mem_ids():
        for spec in by_bank[mem_id]:
            for faster in range(1, mem_id):
                residents = by_bank[faster]
                before = packed_size(residents)
                after = packed_size(list(residents) + [spec])
                if after == before and after <= target.bank(faster).size:
                    test.fail(
                        f"spec of {spec.allocated_memory} B sits in "
                        f"{target.bank(mem_id).name} but would ride free in "
                        f"{target.bank(faster).name} "
                        f"(packed size unchanged at {before})"
                    )


class TestTargetMemoryMap(unittest.TestCase):
    def test_banks_are_addressed_by_their_declared_mem_id(self) -> None:
        target = two_banks(1 * KiB, 8 * KiB)
        self.assertEqual(target.mem_id("fast"), 1)
        self.assertEqual(target.mem_id("slow"), 2)
        self.assertEqual(target.bank(1).name, "fast")
        self.assertEqual(target.mem_ids(), [1, 2])

    def test_declaration_order_is_the_preference_order(self) -> None:
        target = two_banks(64 * KiB, 64 * KiB)
        specs = [make_spec(256, (0, 4))]
        _, alloc = plan(BankedGreedy(target), specs)
        self.assertEqual(alloc[id(specs[0])][0], 1)

    def test_duplicate_bank_names_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetMemoryMap(
                [
                    Bank(name="sram", size=16, mem_id=1),
                    Bank(name="sram", size=16, mem_id=2),
                ]
            )

    def test_duplicate_mem_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetMemoryMap(
                [
                    Bank(name="tcm", size=16, mem_id=1),
                    Bank(name="sram", size=16, mem_id=1),
                ]
            )

    def test_declaration_order_beats_mem_id_order(self) -> None:
        """List order is preference, Bank.mem_id is identity; sorting would fuse them."""
        target = TargetMemoryMap(
            [
                Bank(name="slow", size=4 * KiB, mem_id=2),
                Bank(name="fast", size=4 * KiB, mem_id=1),
            ]
        )
        self.assertEqual(target.mem_ids(), [2, 1])

        spec = make_spec(256, (0, 4))
        bufsizes, alloc = plan(BankedGreedy(target), [spec])
        self.assertEqual(alloc[id(spec)][0], 2)  # first declared, not lowest id
        self.assertEqual(bufsizes[1], 0)
        self.assertEqual(bufsizes[2], 256)

    def test_empty_map_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetMemoryMap([])

    def test_non_positive_bank_size_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetMemoryMap([Bank(name="sram", size=0, mem_id=1)])


class TestSingleBankEquivalence(unittest.TestCase):
    """A flat map must plan bit-identically to exir's greedy."""

    def _fresh(self) -> List[TensorSpec]:
        return [
            make_spec(1024, (0, 4)),
            make_spec(512, (1, 6)),
            make_spec(2048, (5, 9)),
            make_spec(256, (7, 12)),
            make_spec(512, (11, 15)),
        ]

    def test_matches_greedy_on_synthetic_specs(self) -> None:
        banked_specs = self._fresh()
        banked_sizes, banked_alloc = plan(BankedGreedy(flat_map()), banked_specs)

        greedy_specs = self._fresh()
        greedy_result = greedy(16, greedy_specs, empty_graph_module(), None, 0)
        greedy_alloc = {
            i: (
                greedy_result.spec_dict[spec].mem_id,
                greedy_result.spec_dict[spec].mem_offset,
            )
            for i, spec in enumerate(greedy_specs)
        }
        banked_by_index = {
            i: banked_alloc[id(spec)] for i, spec in enumerate(banked_specs)
        }

        self.assertEqual(banked_sizes, greedy_result.bufsizes)
        self.assertEqual(banked_by_index, greedy_alloc)

    def test_matches_greedy_per_spec_on_a_real_graph(self) -> None:
        """Compare every placement, not just the totals."""
        gm, gs = prepare_toy_model()
        specs = list(collect_specs_from_nodes(gm.graph.nodes, gs, do_assertion=False))
        self.assertGreater(len(specs), 10)

        banked = BankedGreedy(flat_map())(16, specs, gm, gs, 0)
        expected = greedy(16, specs, gm, gs, 0)

        self.assertEqual(banked.bufsizes, expected.bufsizes)
        for spec in specs:
            self.assertEqual(
                (
                    banked.spec_dict[spec].mem_id,
                    banked.spec_dict[spec].mem_offset,
                    banked.spec_dict[spec].mem_obj_id,
                ),
                (
                    expected.spec_dict[spec].mem_id,
                    expected.spec_dict[spec].mem_offset,
                    expected.spec_dict[spec].mem_obj_id,
                ),
            )

    def test_matches_greedy_at_alignment_32(self) -> None:
        banked_specs = self._fresh()
        greedy_specs = self._fresh()
        banked = BankedGreedy(flat_map())(
            32, banked_specs, empty_graph_module(), None, 0
        )
        expected = greedy(32, greedy_specs, empty_graph_module(), None, 0)
        self.assertEqual(banked.bufsizes, expected.bufsizes)
        for got, want in zip(banked_specs, greedy_specs):
            self.assertEqual(
                banked.spec_dict[got].mem_offset, expected.spec_dict[want].mem_offset
            )

    def test_matches_greedy_at_a_coarse_alignment(self) -> None:
        """Sort must key on the incoming size, as greedy does, not the realigned one.

        Realigning before sorting manufactures ties, whose insort order then differs
        from greedy's. These sizes make that divergence visible; many do not.
        """

        def fresh() -> List[TensorSpec]:
            sizes = [899, 534, 508, 696, 940, 910, 331]
            return [make_spec(n, (i, i + 3)) for i, n in enumerate(sizes)]

        banked_specs, greedy_specs = fresh(), fresh()
        banked = BankedGreedy(flat_map())(
            256, banked_specs, empty_graph_module(), None, 0
        )
        expected = greedy(256, greedy_specs, empty_graph_module(), None, 0)
        self.assertEqual(banked.bufsizes, expected.bufsizes)
        for got, want in zip(banked_specs, greedy_specs):
            self.assertEqual(
                (banked.spec_dict[got].mem_offset, banked.spec_dict[got].mem_obj_id),
                (
                    expected.spec_dict[want].mem_offset,
                    expected.spec_dict[want].mem_obj_id,
                ),
            )

    def test_is_suite_compatible(self) -> None:
        specs = [make_spec(1024, (0, 4)), make_spec(512, (1, 6))]
        suite = MemoryPlanningAlgorithmSuite(algo_list=[BankedGreedy(flat_map())])
        bufsizes = suite(16, specs, empty_graph_module(), None, 0)
        self.assertEqual(bufsizes[0], 0)
        self.assertGreater(bufsizes[1], 0)
        for spec in specs:
            self.assertEqual(spec.mem_id, 1)


class TestPlacement(unittest.TestCase):
    def test_reserves_buffer_index_zero_for_constants(self) -> None:
        bufsizes, _ = plan(
            BankedGreedy(two_banks(1 * KiB, 8 * KiB)), [make_spec(256, (0, 4))]
        )
        self.assertEqual(len(bufsizes), 3)
        self.assertEqual(bufsizes[0], 0)

    def test_places_everything_in_the_fastest_bank_when_it_fits(self) -> None:
        specs = [make_spec(256, (0, 4)), make_spec(256, (1, 6))]
        bufsizes, alloc = plan(BankedGreedy(two_banks(4 * KiB, 64 * KiB)), specs)
        for spec in specs:
            self.assertEqual(alloc[id(spec)][0], 1)
        self.assertEqual(bufsizes[2], 0)

    def test_spills_to_the_next_bank_when_the_fast_bank_is_full(self) -> None:
        specs = [make_spec(1024, (0, 10)) for _ in range(3)]
        bufsizes, alloc = plan(BankedGreedy(two_banks(2 * KiB, 64 * KiB)), specs)
        self.assertEqual(sorted(alloc[id(s)][0] for s in specs), [1, 1, 2])
        self.assertEqual(bufsizes[1], 2 * KiB)
        self.assertEqual(bufsizes[2], 1 * KiB)

    def test_cascades_across_three_banks(self) -> None:
        target = TargetMemoryMap(
            [
                Bank(name="tcm", size=2 * KiB, mem_id=1),
                Bank(name="sram", size=2 * KiB, mem_id=2),
                Bank(name="dram", size=64 * KiB, mem_id=3),
            ]
        )
        specs = [make_spec(1024, (0, 10)) for _ in range(6)]
        bufsizes, alloc = plan(BankedGreedy(target), specs)
        self.assertEqual(sorted(alloc[id(s)][0] for s in specs), [1, 1, 2, 2, 3, 3])
        self.assertEqual(bufsizes[1], 2 * KiB)
        self.assertEqual(bufsizes[2], 2 * KiB)

    def test_never_exceeds_any_bank_capacity(self) -> None:
        target = two_banks(2 * KiB, 8 * KiB)
        specs = [make_spec(512, (i, i + 20)) for i in range(16)]
        bufsizes, _ = plan(BankedGreedy(target), specs)
        for mem_id in target.mem_ids():
            self.assertLessEqual(bufsizes[mem_id], target.bank(mem_id).size)

    def test_preserves_lifetime_reuse_within_a_bank(self) -> None:
        specs = [make_spec(1024, (0, 4)), make_spec(1024, (5, 9))]
        bufsizes, alloc = plan(BankedGreedy(two_banks(2 * KiB, 8 * KiB)), specs)
        self.assertEqual(alloc[id(specs[0])], alloc[id(specs[1])])
        self.assertEqual(bufsizes[1], 1024)
        self.assertEqual(bufsizes[2], 0)

    def test_fails_when_no_bank_can_hold_a_spec(self) -> None:
        with self.assertRaises(BankPlacementError) as caught:
            plan(
                BankedGreedy(two_banks(1 * KiB, 2 * KiB)), [make_spec(16 * KiB, (0, 4))]
            )
        message = str(caught.exception)
        self.assertIn("16384", message)
        self.assertIn("fast", message)
        self.assertIn("slow", message)

    def test_plan_is_deterministic(self) -> None:
        def run() -> Tuple[List[int], List[int]]:
            specs = [make_spec(256 * (i % 5 + 1), (i, i + 8)) for i in range(40)]
            bufsizes, alloc = plan(BankedGreedy(two_banks(4 * KiB, 64 * KiB)), specs)
            return bufsizes, [alloc[id(s)][0] for s in specs]

        self.assertEqual(run(), run())


class TestFastBankIsFilled(unittest.TestCase):
    """Buffers that would ride free in a faster bank must not land in a slow one."""

    def test_does_not_strand_a_spec_that_would_ride_free_in_a_faster_bank(self) -> None:
        target = two_banks(16 * KiB, 64 * MiB)
        specs = [make_spec(4 * KiB + 16 * i, (i, i + 3)) for i in range(120)]
        _, alloc = plan(BankedGreedy(target), specs)
        assert_no_free_readmission(self, target, specs, alloc)

    def test_short_lifetimes_let_many_specs_share_the_fast_bank(self) -> None:
        target = two_banks(8 * KiB, 64 * MiB)
        specs = [make_spec(4 * KiB, (i, i + 1)) for i in range(200)]
        _, alloc = plan(BankedGreedy(target), specs)
        in_fast = sum(1 for s in specs if alloc[id(s)][0] == 1)
        # Two 4 KiB objects in an 8 KiB bank, and lifetimes are disjoint in pairs,
        # so most of the 200 specs should ride inside them.
        self.assertGreater(in_fast, 100)

    def test_reports_the_bank_when_a_spec_fits_nowhere(self) -> None:
        target = TargetMemoryMap([Bank(name="fast", size=2 * KiB, mem_id=1)])
        specs = [make_spec(1024, (0, 10)) for _ in range(3)]
        with self.assertRaises(BankPlacementError) as caught:
            plan(BankedGreedy(target), specs)
        self.assertIn("fast", str(caught.exception))

    def test_extra_padding_is_charged_only_to_banks_that_hold_something(self) -> None:
        """greedy pads only the arenas it touched; an idle bank must stay at 0."""
        target = two_banks(4 * KiB, 64 * KiB)
        bufsizes, _ = plan(
            BankedGreedy(target), [make_spec(256, (0, 4))], extra_padding=64
        )
        self.assertEqual(bufsizes[1], 256 + 64)
        self.assertEqual(bufsizes[2], 0)

    def test_padding_counts_against_bank_capacity(self) -> None:
        """A buffer that fits a bank on its own can still be pushed out by padding."""
        target = two_banks(32, 64 * KiB)
        bufsizes, _ = plan(
            BankedGreedy(target), [make_spec(16, (0, 4))], extra_padding=64
        )
        # 16 B fits the 32 B bank; 16 B plus 64 B of padding does not.
        self.assertEqual(bufsizes[1], 0)
        self.assertEqual(bufsizes[2], 16 + 64)

    def test_respects_submodule_reserved_bytes(self) -> None:
        target = two_banks(4 * KiB, 64 * KiB)
        gm = empty_graph_module()
        # Index 0 is the constants slot; bank 1 already has 3 KiB reserved by a
        # control-flow submodule, leaving room for only one 1 KiB spec.
        gm.input_mem_buffer_sizes = [0, 3 * KiB, 0]
        specs = [make_spec(1024, (0, 10)) for _ in range(3)]
        bufsizes, alloc = plan(BankedGreedy(target), specs, graph_module=gm)
        self.assertLessEqual(bufsizes[1], 4 * KiB)
        self.assertEqual(sorted(alloc[id(s)][0] for s in specs), [1, 2, 2])


class TestPreAssignedMemIds(unittest.TestCase):
    """A custom pool pass may run ahead of the planner and pin specs."""

    def test_pinned_spec_stays_in_its_bank(self) -> None:
        target = two_banks(4 * KiB, 64 * KiB)
        pinned = make_spec(256, (0, 10))
        pinned.mem_id = 2  # a custom pool pass put this in the slow bank
        free = make_spec(256, (0, 10))

        _, alloc = plan(BankedGreedy(target), [pinned, free])
        self.assertEqual(alloc[id(pinned)][0], 2)
        self.assertEqual(alloc[id(free)][0], 1)

    def test_free_specs_cascade_around_a_pin(self) -> None:
        target = two_banks(2 * KiB, 64 * KiB)
        pinned = make_spec(1024, (0, 10))
        pinned.mem_id = 1
        free = [make_spec(1024, (0, 10)) for _ in range(3)]

        _, alloc = plan(BankedGreedy(target), [pinned] + free)
        self.assertEqual(alloc[id(pinned)][0], 1)
        self.assertEqual(sorted(alloc[id(s)][0] for s in free), [1, 2, 2])

    def test_capacity_is_reserved_so_a_pin_is_never_squeezed_out(self) -> None:
        """The pin is smallest, so a first-come rule would fill the bank first."""
        target = two_banks(2 * KiB, 64 * KiB)
        pinned = make_spec(256, (0, 10))
        pinned.mem_id = 1
        free = [make_spec(1024, (0, 10)) for _ in range(4)]

        _, alloc = plan(BankedGreedy(target), free + [pinned])
        self.assertEqual(alloc[id(pinned)][0], 1)
        # 2 KiB less the 256 B reservation leaves room for exactly one 1 KiB spec.
        self.assertEqual(sorted(alloc[id(s)][0] for s in free), [1, 2, 2, 2])

    def test_pinned_and_unpinned_buffers_share_storage(self) -> None:
        """A pin must not wall off its bytes from disjoint unpinned buffers."""
        target = two_banks(4 * KiB, 64 * KiB)
        pinned = make_spec(1024, (0, 4))
        pinned.mem_id = 1
        free = make_spec(1024, (5, 9))  # disjoint lifetime: can share

        bufsizes, alloc = plan(BankedGreedy(target), [pinned, free])
        self.assertEqual(alloc[id(free)], alloc[id(pinned)])
        self.assertEqual(bufsizes[1], 1024)
        self.assertEqual(bufsizes[2], 0)

    def test_unpinned_buffer_larger_than_a_pin_takes_its_own_object(self) -> None:
        """Placing pins first costs bytes when a pin is the smaller buffer.

        A shared object's size is fixed by its first occupant, so the 64 B pin's
        object cannot host the 512 B buffer. greedy, which sorts purely by size,
        would place the 512 B buffer first and let the pin ride inside it for 512
        total. This is the price of giving pins first claim on capacity, and it is
        an ordering effect -- not a restriction on sharing.
        """
        target = two_banks(4 * KiB, 64 * KiB)
        pinned = make_spec(64, (0, 4))
        pinned.mem_id = 1
        free = make_spec(512, (5, 9))

        bufsizes, alloc = plan(BankedGreedy(target), [pinned, free])
        self.assertEqual(alloc[id(free)][0], 1)
        self.assertEqual(bufsizes[1], 576)  # 64 + 512, vs greedy's 512

    def test_pinned_graph_matches_greedy_when_one_bank_is_declared(self) -> None:
        """Sharing across the pin boundary is what makes this hold."""

        def fresh():
            specs = [
                make_spec(1024, (0, 4)),
                make_spec(512, (5, 9)),
                make_spec(256, (10, 14)),
                make_spec(2048, (0, 14)),
            ]
            for spec in specs[:2]:
                spec.mem_id = 1
            return specs

        banked = BankedGreedy(flat_map())(16, fresh(), empty_graph_module(), None, 0)
        expected = greedy(16, fresh(), empty_graph_module(), None, 0)
        self.assertEqual(banked.bufsizes, expected.bufsizes)

    def test_a_pin_that_rides_free_is_not_charged_capacity(self) -> None:
        """Two pins with disjoint lifetimes share one object, so one bank's worth fits."""
        target = two_banks(1024, 64 * KiB)
        first, second = make_spec(1024, (0, 4)), make_spec(1024, (5, 9))
        first.mem_id = second.mem_id = 1

        bufsizes, alloc = plan(BankedGreedy(target), [first, second])
        self.assertEqual(alloc[id(second)], alloc[id(first)])
        self.assertEqual(bufsizes[1], 1024)

    def test_pin_larger_than_its_bank_is_an_error(self) -> None:
        target = two_banks(1 * KiB, 64 * KiB)
        pinned = make_spec(4 * KiB, (0, 10))
        pinned.mem_id = 1
        with self.assertRaises(BankPlacementError) as caught:
            plan(BankedGreedy(target), [pinned])
        message = str(caught.exception)
        self.assertIn("4096", message)
        self.assertIn("fast", message)

    def test_pins_that_collectively_overflow_are_an_error(self) -> None:
        target = two_banks(2 * KiB, 64 * KiB)
        pins = []
        for _ in range(3):
            spec = make_spec(1024, (0, 10))
            spec.mem_id = 1
            pins.append(spec)
        with self.assertRaises(BankPlacementError) as caught:
            plan(BankedGreedy(target), pins)
        message = str(caught.exception)
        self.assertIn("pinned to bank fast", message)
        self.assertIn("no fallback bank", message)

    def test_pin_outside_the_map_is_laid_out_but_not_budgeted(self) -> None:
        """An undeclared pool belongs to the pass that wrote it."""
        target = two_banks(4 * KiB, 64 * KiB)
        pinned = make_spec(256, (0, 10))
        pinned.mem_id = 7
        free = make_spec(256, (0, 10))

        bufsizes, alloc = plan(BankedGreedy(target), [pinned, free])
        self.assertEqual(alloc[id(pinned)][0], 7)
        self.assertEqual(alloc[id(free)][0], 1)
        self.assertEqual(bufsizes[7], 256)

    def test_undeclared_pool_never_receives_unpinned_buffers(self) -> None:
        """The reason to leave a semantically special region out of the map."""
        target = TargetMemoryMap([Bank(name="fast", size=2 * KiB, mem_id=1)])
        dma = make_spec(256, (0, 10))
        dma.mem_id = 4  # a DMA-visible pool the planner must not fill
        free = [make_spec(1024, (0, 10)) for _ in range(2)]

        with self.assertRaises(BankPlacementError):
            # The third buffer has nowhere to go: bank 1 is full and pool 4 is
            # not a spill target. It must fail rather than land in the DMA pool.
            plan(BankedGreedy(target), [dma] + free + [make_spec(1024, (0, 10))])

        _, alloc = plan(BankedGreedy(target), [dma] + free)
        self.assertEqual(alloc[id(dma)][0], 4)
        for spec in free:
            self.assertEqual(alloc[id(spec)][0], 1)

    def test_custom_pool_pass_composes_end_to_end(self) -> None:
        """The documented tagging pattern, planned by the banked algorithm."""

        class TagAddsToSlowBank(MemoryPlanningPass):
            def run(self, graph_module, graph_signature=None):
                for subgm in graph_module.modules():
                    if not isinstance(subgm, GraphModule):
                        continue
                    for node in subgm.graph.nodes:
                        if node.op == "call_function" and "add" in str(node.target):
                            spec = node.meta.get("spec")
                            if isinstance(spec, TensorSpec):
                                spec.mem_id = 2
                return super().run(graph_module, graph_signature)

        class AddNet(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + x) + 1.0

        target = two_banks(64 * KiB, 1024 * KiB)
        memory_pass = TagAddsToSlowBank(
            memory_planning_algo=MemoryPlanningAlgorithmSuite(
                algo_list=[BankedGreedy(target)]
            )
        )
        edge = to_edge(export(AddNet(), (torch.randn(1, 64),), strict=True))
        program = edge.to_executorch(
            ExecutorchBackendConfig(memory_planning_pass=memory_pass)
        )
        bufsizes = list(
            program.executorch_program.execution_plan[0].non_const_buffer_sizes
        )
        self.assertEqual(len(bufsizes), 3)
        # Tagged adds landed in bank 2; everything else in bank 1.
        self.assertGreater(bufsizes[2], 0)
        self.assertGreater(bufsizes[1], 0)


class TestInPlaceSpecs(unittest.TestCase):
    def test_inplace_spec_shares_its_base_placement(self) -> None:
        base = make_spec(1024, (0, 4))
        aliased = make_spec(1024, (4, 8))
        aliased.inplace_base = base

        _, alloc = plan(BankedGreedy(two_banks(4 * KiB, 64 * KiB)), [base, aliased])
        self.assertEqual(alloc[id(aliased)], alloc[id(base)])

    def test_inplace_spec_follows_its_base_into_the_slow_bank(self) -> None:
        target = two_banks(2 * KiB, 64 * KiB)
        big = [make_spec(1024, (0, 10)) for _ in range(2)]
        base = make_spec(1024, (0, 10))
        aliased = make_spec(1024, (0, 10))
        aliased.inplace_base = base

        _, alloc = plan(BankedGreedy(target), big + [base, aliased])
        self.assertEqual(alloc[id(aliased)], alloc[id(base)])

    def test_inplace_spec_follows_a_base_that_spilled(self) -> None:
        """The base is forced to the slow bank; the alias must go with it."""
        target = two_banks(2 * KiB, 64 * KiB)
        fillers = [make_spec(1024, (0, 20)) for _ in range(2)]
        base = make_spec(512, (0, 20))  # smaller, so it is placed last and spills
        aliased = make_spec(512, (0, 20))
        aliased.inplace_base = base

        _, alloc = plan(BankedGreedy(target), fillers + [base, aliased])
        self.assertEqual(alloc[id(base)][0], 2)
        self.assertEqual(alloc[id(aliased)], alloc[id(base)])

    def test_inplace_spec_does_not_double_count_against_capacity(self) -> None:
        base = make_spec(1024, (0, 10))
        aliased = make_spec(1024, (0, 10))
        aliased.inplace_base = base
        bufsizes, _ = plan(BankedGreedy(two_banks(1 * KiB, 64 * KiB)), [base, aliased])
        self.assertEqual(bufsizes[1], 1024)
        self.assertEqual(bufsizes[2], 0)


class TestPlanningPassIntegration(unittest.TestCase):
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(64, 64)
            self.fc2 = torch.nn.Linear(64, 64)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    def _plan(self, target: TargetMemoryMap):
        edge = to_edge(export(self.Net(), (torch.randn(1, 64),), strict=True))
        program = edge.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=banked_memory_planning_pass(target)
            )
        )
        sizes = program.executorch_program.execution_plan[0].non_const_buffer_sizes
        return program, list(sizes)

    def test_export_produces_one_arena_per_bank(self) -> None:
        _, bufsizes = self._plan(two_banks(64 * KiB, 1024 * KiB))
        self.assertEqual(len(bufsizes), 3)
        self.assertEqual(bufsizes[0], 0)

    def test_export_respects_capacity_and_has_no_aliasing(self) -> None:
        target = two_banks(8 * KiB, 1024 * KiB)
        program, bufsizes = self._plan(target)
        for mem_id in target.mem_ids():
            self.assertLessEqual(bufsizes[mem_id], target.bank(mem_id).size)
        gm = program.exported_program().graph_module
        Verifier(gm, True, True, True, None).verify_storage_reuse()


class TestBankedProgramExecutes(unittest.TestCase):
    """A banked plan must not just serialize -- it must run."""

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.ModuleList(
                [torch.nn.Conv2d(8, 8, 3, padding=1) for _ in range(4)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for conv in self.conv:
                x = torch.relu(conv(x))
            return x

    @unittest.skipUnless(_HAS_RUNTIME, "portable_lib not built")
    def test_spilled_plan_runs_and_matches_eager(self) -> None:
        model, inputs = self.Net().eval(), (torch.randn(1, 8, 16, 16),)
        # 8 KiB holds one 8192 B activation, so the plan must use both arenas.
        target = two_banks(8 * KiB, 8 * MiB)
        program = to_edge(export(model, inputs, strict=True)).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=banked_memory_planning_pass(target)
            )
        )
        arenas = list(
            program.executorch_program.execution_plan[0].non_const_buffer_sizes
        )
        self.assertEqual(arenas, [0, 8 * KiB, 8 * KiB])

        runtime = _load_for_executorch_from_buffer(program.buffer)
        got = runtime.forward(list(inputs))[0]
        torch.testing.assert_close(got, model(*inputs))


class TestControlFlow(unittest.TestCase):
    class MapNet(torch.nn.Module):
        def forward(self, xs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            def body(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            return torch_map(body, xs, y)

    def test_submodule_arenas_are_reserved_and_match_greedy(self) -> None:
        """apply_algo re-enters the planner per submodule; the parent must reserve."""
        inputs = (torch.randn(4, 256), torch.randn(256))
        target = two_banks(64 * KiB, 1024 * KiB)

        def sizes(memory_pass):
            edge = to_edge(export(self.MapNet(), inputs, strict=True))
            program = edge.to_executorch(
                ExecutorchBackendConfig(memory_planning_pass=memory_pass)
            )
            return list(
                program.executorch_program.execution_plan[0].non_const_buffer_sizes
            )

        flat = sizes(
            MemoryPlanningPass(
                memory_planning_algo=MemoryPlanningAlgorithmSuite(algo_list=[greedy])
            )
        )
        banked = sizes(banked_memory_planning_pass(target))
        self.assertEqual(banked[1], flat[1])
        self.assertLessEqual(banked[1], 64 * KiB)


class TestPerBankAlignment(unittest.TestCase):
    def test_bank_alignment_overrides_the_global_alignment(self) -> None:
        strict = TargetMemoryMap(
            [Bank(name="tcm", size=4 * KiB, mem_id=1, alignment=32)]
        )
        bufsizes, _ = plan(BankedGreedy(strict), [make_spec(40, (0, 4))])
        self.assertEqual(bufsizes[1], 64)  # 40 -> 64 at alignment 32

    def test_bank_without_alignment_inherits_the_global_one(self) -> None:
        bufsizes, _ = plan(BankedGreedy(flat_map()), [make_spec(40, (0, 4))])
        self.assertEqual(bufsizes[1], 48)  # 40 -> 48 at alignment 16

    def test_each_bank_applies_its_own_alignment(self) -> None:
        target = TargetMemoryMap(
            [
                Bank(name="tcm", size=64, mem_id=1, alignment=32),
                Bank(name="sram", size=64 * KiB, mem_id=2, alignment=16),
            ]
        )
        first, second = make_spec(40, (0, 10)), make_spec(40, (0, 10))
        bufsizes, alloc = plan(BankedGreedy(target), [first, second])
        # tcm rounds 40 up to 64 and is then full; sram takes the other at 48.
        self.assertEqual(sorted(alloc[id(s)][0] for s in (first, second)), [1, 2])
        self.assertEqual(bufsizes[1], 64)
        self.assertEqual(bufsizes[2], 48)

    def test_bank_alignment_must_be_a_multiple_of_the_graph_alignment(self) -> None:
        """Otherwise ties invert and pick_shared_obj trips a bare internal assert."""
        for bad in (4, 24):  # looser, and stricter but not a multiple
            target = TargetMemoryMap(
                [Bank(name="tcm", size=4 * KiB, mem_id=1, alignment=bad)]
            )
            with self.assertRaises(BankPlacementError) as caught:
                plan(BankedGreedy(target), [make_spec(32, (0, 4))])
            self.assertIn("multiple", str(caught.exception))

    def test_pins_respect_per_bank_alignment(self) -> None:
        target = TargetMemoryMap(
            [Bank(name="tcm", size=4 * KiB, mem_id=1, alignment=32)]
        )
        pinned = make_spec(40, (0, 4))
        pinned.mem_id = 1
        bufsizes, _ = plan(BankedGreedy(target), [pinned])
        self.assertEqual(bufsizes[1], 64)

    def test_non_positive_bank_alignment_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetMemoryMap([Bank(name="x", size=64, mem_id=1, alignment=0)])


class TestOverlappingReuse(unittest.TestCase):
    """_reusable_object re-implements pick_shared_obj's second reuse rule."""

    def test_offset_reuse_matches_greedy_per_spec(self) -> None:
        """The third spec is admitted above an allocation whose lifetime overlaps."""

        def fresh() -> List[TensorSpec]:
            return [
                make_spec(256, (6, 7)),
                make_spec(128, (0, 4)),
                make_spec(128, (1, 1)),
            ]

        banked_specs, greedy_specs = fresh(), fresh()
        banked = BankedGreedy(flat_map())(
            16, banked_specs, empty_graph_module(), None, 0
        )
        expected = greedy(16, greedy_specs, empty_graph_module(), None, 0)

        self.assertEqual(banked.bufsizes, expected.bufsizes)
        for got, want in zip(banked_specs, greedy_specs):
            self.assertEqual(
                (banked.spec_dict[got].mem_id, banked.spec_dict[got].mem_offset),
                (expected.spec_dict[want].mem_id, expected.spec_dict[want].mem_offset),
            )
        # Reuse must land at a non-zero offset, or this only exercises path one.
        self.assertGreater(max(a.mem_offset for a in banked.spec_dict.values()), 0)


class TestUnsupportedConfigurations(unittest.TestCase):
    def test_non_cpu_specs_are_rejected(self) -> None:
        """A memory map describes one address space, not one per device."""
        spec = make_spec(256, (0, 4))
        spec.device = DeviceType.CUDA
        with self.assertRaises(BankPlacementError) as caught:
            plan(BankedGreedy(two_banks(4 * KiB, 64 * KiB)), [spec])
        self.assertIn("CUDA", str(caught.exception))

    def test_non_cpu_in_place_specs_are_rejected(self) -> None:
        """In-place specs are excluded from placement but still carry a device."""
        base = make_spec(256, (0, 4))
        aliased = make_spec(256, (0, 4))
        aliased.inplace_base = base
        aliased.device = DeviceType.CUDA
        with self.assertRaises(BankPlacementError):
            plan(BankedGreedy(two_banks(4 * KiB, 64 * KiB)), [base, aliased])

    def test_allow_overlapping_allocations_reaches_the_planner(self) -> None:
        """Vulkan-style configs disable overlapping; the factory must forward it."""
        memory_pass = banked_memory_planning_pass(
            flat_map(), allow_overlapping_allocations=False
        )
        planner = memory_pass.memory_planning_algo.algo_list[0]
        self.assertFalse(planner.allow_overlapping_allocations)

    def test_share_mutable_buffers_is_rejected_for_now(self) -> None:
        """Supporting it needs a core change to where shared state is placed."""
        with self.assertRaises(ValueError) as caught:
            banked_memory_planning_pass(flat_map(), share_mutable_buffers=True)
        self.assertIn("share_mutable_buffers", str(caught.exception))


class TestPlacementReport(unittest.TestCase):
    def test_report_lists_bytes_and_occupancy_per_bank(self) -> None:
        target = two_banks(2 * KiB, 64 * KiB)
        specs = [make_spec(1024, (0, 10)) for _ in range(3)]
        planner = BankedGreedy(target)
        result = planner(16, specs, empty_graph_module(), None, 0)
        report = format_placement_report(
            target,
            result.bufsizes,
            [alloc.mem_id for alloc in result.spec_dict.values()],
        )

        self.assertIn("fast", report)
        self.assertIn("slow", report)
        self.assertIn("2048", report)
        self.assertIn("100.0%", report)


if __name__ == "__main__":
    unittest.main()
