# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

r"""Capacity-aware memory planning across a target's memory hierarchy.

ExecuTorch plans tensors into numbered arenas (``mem_id``) and the runtime supports
several, but nothing tells the planner how large an arena is: ``greedy`` honors
whatever ``mem_id`` a custom pool pass assigned, puts everything else in arena 1,
and over-subscription surfaces as a link error or a runtime abort.

:class:`TargetMemoryMap` supplies the missing sizes. Buffers fill the fastest bank
that can hold them and spill to the next; one that fits nowhere fails the export
rather than the device. With a single bank the plan is bit-identical to ``greedy``.

Custom pool passes still work ahead of this planner, as described in
``docs/source/compiler-memory-planning.md``; the ``mem_id`` they assign is honored
as a pin. Per-buffer constraints richer than capacity -- DMA reachability,
cacheability -- are deliberately absent until a consumer can enforce them.
"""

import bisect
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from executorch.exir.memory_planning import (
    _compute_total_sizes,
    _does_not_overlap,
    _find_max_overlapping_allocations_offset,
    _resolve_inplace_specs,
    AllocationSpec,
    get_node_tensor_specs,
    MemoryAlgoResult,
    MemoryPlanningAlgorithmSuite,
    pick_shared_obj,
    SharedObject,
    SpecAllocResult,
)
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import ALIGNMENT, TensorSpec
from torch.export.exported_program import ExportGraphSignature


@dataclass(frozen=True)
class Bank:
    """A memory region the planner may fill.

    ``size`` is the budget for *this model*, usually less than the physical region:
    the stack, the method allocator and anything else resident there come out of it
    first. ``mem_id`` is the runtime arena index, declared rather than inferred from
    list position because the map is sparse. ``alignment`` tightens the graph's
    alignment for this bank and must be a multiple of it; ``None`` inherits it.
    """

    name: str
    size: int
    mem_id: int
    alignment: Optional[int] = None


class TargetMemoryMap:
    """The pools the planner may fill, in preference order (fastest first).

    Declaring a bank hands that arena to the planner: it spills unpinned buffers
    into it and enforces its capacity. An undeclared ``mem_id`` behaves as it does
    under ``greedy`` -- a pass may pin buffers to it and they are laid out normally
    -- but the planner never places anything there on its own. That is how a
    DMA-visible pool or accelerator-private scratch is protected: leave it out.

    List order is preference; ``Bank.mem_id`` is identity. They are separate
    because the map is sparse.
    """

    def __init__(self, banks: Sequence[Bank]) -> None:
        if not banks:
            raise ValueError("A target memory map needs at least one bank")
        names = [bank.name for bank in banks]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate bank names in target memory map: {duplicates}")
        ids = [bank.mem_id for bank in banks]
        clashes = {i for i in ids if ids.count(i) > 1}
        if clashes:
            raise ValueError(f"Duplicate mem_ids in target memory map: {clashes}")
        for bank in banks:
            if bank.size <= 0:
                raise ValueError(f"Bank {bank.name} has non-positive size {bank.size}")
            if bank.alignment is not None and bank.alignment < 1:
                raise ValueError(
                    f"Bank {bank.name} has non-positive alignment {bank.alignment}"
                )
            if bank.mem_id < 1:
                raise ValueError(
                    f"Bank {bank.name} has mem_id {bank.mem_id}; arena 0 is reserved "
                    "for constants, so planned banks start at 1"
                )
        self.banks: List[Bank] = list(banks)
        self._by_id: Dict[int, Bank] = {bank.mem_id: bank for bank in banks}

    def mem_ids(self) -> List[int]:
        """Declared arena ids, in preference order."""
        return [bank.mem_id for bank in self.banks]

    def manages(self, mem_id: int) -> bool:
        return mem_id in self._by_id

    def bank(self, mem_id: int) -> Bank:
        if mem_id not in self._by_id:
            raise ValueError(f"mem_id {mem_id} is not in the target memory map")
        return self._by_id[mem_id]

    def mem_id(self, name: str) -> int:
        for bank in self.banks:
            if bank.name == name:
                return bank.mem_id
        raise ValueError(f"No bank named {name!r} in the target memory map")

    def describe(self) -> str:
        return ", ".join(
            f"{bank.name}(mem_id={bank.mem_id}, size={bank.size})"
            for bank in self.banks
        )


class BankPlacementError(Exception):
    """The target memory map cannot hold the graph's planned buffers."""


class BankedGreedy:
    """Plan specs across a :class:`TargetMemoryMap`, honoring bank capacity.

    One pass, largest buffer first, offering each to the fastest bank that can take
    it: a bank accepts if the buffer fits an existing shared object for free, or if
    it has headroom for a new one. Running out of banks fails the export. Every bank
    therefore receives a size-descending subsequence, which is the ordering
    ``pick_shared_obj`` requires, so a single-bank map reproduces ``greedy`` exactly.

    Pins -- specs whose ``mem_id`` a custom pool pass already set -- go down first,
    so nothing can take the space they need, and a pin that does not fit is an error
    rather than a silent relocation. A pin to an undeclared ``mem_id`` is laid out
    but not budgeted.

    Unpinned buffers then share storage with pins freely: every pin is already
    placed, so admitting one cannot block a pin still to come. That phase uses
    :func:`_reusable_object` rather than ``pick_shared_obj``, because pins and
    unpinned buffers are each size-descending but their concatenation is not, and
    that function asserts on the ordering it would then see.

    Buffers are sorted by their incoming size, as ``greedy`` sorts, and realigned to
    whichever bank they are offered to. A bank's alignment must be a multiple of the
    graph's, which is what keeps that realignment order-preserving.
    """

    def __init__(
        self,
        target: TargetMemoryMap,
        allow_overlapping_allocations: bool = True,
    ) -> None:
        self.target = target
        self.allow_overlapping_allocations = allow_overlapping_allocations
        # Named so MemoryPlanningAlgorithmSuite can report which algo it picked.
        self.__name__ = "banked_greedy"

    def __call__(
        self,
        alignment: int,
        specs: Iterable[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
        extra_padding: int = 0,
    ) -> MemoryAlgoResult:
        result = MemoryAlgoResult({}, [])
        self._check_bank_alignments(alignment)
        all_specs = list(specs)

        # In-place outputs share their base's storage, so they inherit its bank
        # rather than being placed or counted against capacity.
        planned = [spec for spec in all_specs if spec.inplace_base is None]
        deferred_inplace = [spec for spec in all_specs if spec.inplace_base is not None]
        for spec in deferred_inplace:
            spec.realign(alignment)

        self._reject_non_cpu(all_specs)

        objects: Dict[int, List[SharedObject]] = {}
        used: Dict[int, int] = {}
        for mem_id in self.target.mem_ids():
            objects[mem_id] = []
            used[mem_id] = _submodule_reserved(graph_module, mem_id)
        spec2obj: Dict[TensorSpec, SharedObject] = {}

        # Pins first: nothing else may take the space they need, and they have no
        # fallback bank. extra_padding is charged only to banks that hold something,
        # as greedy pads only the arenas it touched.
        pinned = [spec for spec in planned if spec.mem_id is not None]
        for spec in _packing_order(pinned):
            mem_id = spec.mem_id
            bank_objects = objects.setdefault(mem_id, [])
            spec.realign(self._bank_alignment(mem_id, alignment))
            before = len(bank_objects)
            sobj = pick_shared_obj(
                bank_objects, spec, self.allow_overlapping_allocations
            )
            spec2obj[spec] = sobj
            result.spec_dict[spec] = SpecAllocResult(mem_id, 0, 0)
            if len(bank_objects) == before:
                continue  # rode along inside an object already there
            if not self.target.manages(mem_id):
                # Not our pool: lay it out, but claim no budget over it.
                continue
            bank = self.target.bank(mem_id)
            cost = spec.allocated_memory + (
                extra_padding if len(bank_objects) == 1 else 0
            )
            if used[mem_id] + cost > bank.size:
                raise BankPlacementError(
                    f"A {spec.allocated_memory}-byte buffer (shape "
                    f"{list(spec.shape)}) is pinned to bank {bank.name} "
                    f"(mem_id={mem_id}), which has {bank.size - used[mem_id]} of "
                    f"{bank.size} bytes left. A pinned buffer has no fallback bank."
                )
            used[mem_id] += cost

        for spec in _packing_order([s for s in planned if s.mem_id is None]):
            mem_id, sobj = self._place_free(
                spec, objects, used, alignment, extra_padding
            )
            result.spec_dict[spec] = SpecAllocResult(mem_id, 0, 0)
            spec2obj[spec] = sobj

        for spec in deferred_inplace:
            result.spec_dict[spec] = SpecAllocResult(0, 0, 0)
        _resolve_inplace_specs(deferred_inplace, spec2obj, result)

        result.bufsizes = _compute_total_sizes(
            objects, graph_module, 0, result, len(spec2obj)
        )
        for mem_id, bank_objects in objects.items():
            if bank_objects and mem_id < len(result.bufsizes):
                result.bufsizes[mem_id] += extra_padding
        self._check_capacity(result.bufsizes)
        logging.debug(f"banked greedy returns bufsizes: {result.bufsizes}")
        return result

    def _check_bank_alignments(self, alignment: int) -> None:
        for bank in self.target.banks:
            if bank.alignment is not None and bank.alignment % alignment:
                raise BankPlacementError(
                    f"Bank {bank.name} declares alignment {bank.alignment}, which is "
                    f"not a multiple of the graph's alignment {alignment}. A bank may "
                    f"tighten alignment, but only to a multiple: otherwise buffers "
                    f"land at offsets the export did not promise, and the packing "
                    f"order stops being size-descending."
                )

    def _bank_alignment(self, mem_id: int, default: int) -> int:
        if not self.target.manages(mem_id):
            return default
        return self.target.bank(mem_id).alignment or default

    def _reject_non_cpu(self, specs: Sequence[TensorSpec]) -> None:
        """One map describes one address space, so plan only the CPU's.

        Two ways a graph can violate that. With ``enable_non_cpu_memory_planning``
        on, ``apply_algo`` calls the algorithm once per device partition, and the
        same bank sizes would then be granted to each -- a 1 MiB bank handed out
        twice for one physical region. With it off, ``_partition_specs_by_device``
        puts every spec in a single bucket regardless of device, so a mixed batch
        arrives here and non-CPU tensors would be planned into CPU budgets.

        Checking that the batch is *homogeneous* would catch neither: per-device
        partitioning already makes every batch homogeneous.
        """
        offenders = {spec.device for spec in specs if spec.device != DeviceType.CPU}
        if offenders:
            names = ", ".join(sorted(d.name for d in offenders))
            raise BankPlacementError(
                f"Banked memory planning supports CPU specs only, but this graph "
                f"has specs on {names}. A target memory map describes one address "
                f"space and its bank sizes are host memory budgets, so it cannot "
                f"stand in for another device's arenas."
            )

    def _place_free(
        self,
        spec: TensorSpec,
        objects: Dict[int, List[SharedObject]],
        used: Dict[int, int],
        alignment: int,
        extra_padding: int,
    ) -> Tuple[int, SharedObject]:
        """Put an unpinned spec in the fastest declared bank that can take it."""
        for mem_id in self.target.mem_ids():
            spec.realign(self._bank_alignment(mem_id, alignment))
            reuse = _reusable_object(
                objects[mem_id], spec, self.allow_overlapping_allocations
            )
            if reuse is not None:
                # Fits inside an existing object: the bank does not grow.
                sobj, offset = reuse
                sobj.first_used_index = min(sobj.first_used_index, spec.lifetime[0])
                sobj.last_used_index = max(sobj.last_used_index, spec.lifetime[1])
                sobj.allocations.append(AllocationSpec(offset, spec))
                return mem_id, sobj
            cost = spec.allocated_memory + (extra_padding if not objects[mem_id] else 0)
            if used[mem_id] + cost <= self.target.bank(mem_id).size:
                used[mem_id] += cost
                sobj = SharedObject(
                    len(objects[mem_id]),
                    -1,
                    spec.allocated_memory,
                    spec.lifetime[0],
                    spec.lifetime[1],
                )
                sobj.allocations.append(AllocationSpec(0, spec))
                objects[mem_id].append(sobj)
                return mem_id, sobj

        remaining = ", ".join(
            f"{self.target.bank(m).name} {self.target.bank(m).size - used[m]} of "
            f"{self.target.bank(m).size} free"
            for m in self.target.mem_ids()
        )
        raise BankPlacementError(
            f"No declared bank can hold a {spec.allocated_memory}-byte buffer "
            f"(shape {list(spec.shape)}): {remaining}. Give a bank more room, or "
            f"declare another one. Undeclared pools are never used as spill "
            f"targets. Target memory map: {self.target.describe()}"
        )

    def _check_capacity(self, bufsizes: Sequence[int]) -> None:
        """Backstop. Unreachable when the incremental accounting is right."""
        for mem_id in self.target.mem_ids():
            bank = self.target.bank(mem_id)
            size = bufsizes[mem_id] if mem_id < len(bufsizes) else 0
            if size > bank.size:
                raise BankPlacementError(
                    f"Bank {bank.name} (mem_id={mem_id}) needs {size} bytes but is "
                    f"only {bank.size} bytes. Target memory map: "
                    f"{self.target.describe()}"
                )


def _packing_order(specs: Sequence[TensorSpec]) -> List[TensorSpec]:
    """Size-descending, tie-broken exactly as ``exir.memory_planning.greedy``."""
    sorted_specs: List[TensorSpec] = []
    for spec in specs:
        bisect.insort(sorted_specs, spec, key=lambda x: x.allocated_memory)
    sorted_specs.reverse()
    return sorted_specs


def _reusable_object(
    objects: Sequence[SharedObject],
    spec: TensorSpec,
    allow_overlapping_allocations: bool,
) -> Optional[Tuple[SharedObject, int]]:
    """The object and offset that hold spec without growing the bank, if any.

    Mirrors ``pick_shared_obj``'s two reuse paths, read-only, and *filters* on
    ``sobj.size >= spec.allocated_memory`` where that function asserts on it.
    Unpinned buffers are offered after the pins, so the combined sequence a bank
    sees is not size-descending and an undersized object must be skipped rather
    than tripping the assert.
    """
    for sobj in objects:
        if sobj.size >= spec.allocated_memory and _does_not_overlap(sobj, spec):
            return sobj, 0
    if allow_overlapping_allocations:
        for sobj in objects:
            max_offset = _find_max_overlapping_allocations_offset(sobj, spec)
            if max_offset > 0 and max_offset + spec.allocated_memory <= sobj.size:
                return sobj, max_offset
    return None


def _submodule_reserved(graph_module: torch.fx.GraphModule, mem_id: int) -> int:
    """Bytes a control-flow submodule already reserved in this bank."""
    bufsizes = getattr(graph_module, "input_mem_buffer_sizes", None)
    if not bufsizes or len(bufsizes) <= mem_id:
        return 0
    return bufsizes[mem_id]


def banked_memory_planning_pass(
    target: TargetMemoryMap,
    alignment: int = ALIGNMENT,
    allow_overlapping_allocations: bool = True,
    **kwargs: object,
) -> MemoryPlanningPass:
    """A ``MemoryPlanningPass`` that plans across ``target``'s banks.

    The planner is wrapped in ``MemoryPlanningAlgorithmSuite`` because that is
    what writes the winning placement back onto each spec. Remaining keyword
    arguments go to ``MemoryPlanningPass`` (``alloc_graph_input`` and friends).

    Pass it as the *only* algorithm, as this helper does. The suite picks whichever
    algorithm reports the smallest ``sum(bufsizes)``, which values a byte of slow
    memory exactly like a byte of fast memory and applies no capacity check of its
    own; putting ``greedy`` alongside this planner would let an unvalidated
    single-arena plan win and silently over-subscribe a bank.

    To combine this with a custom pool pass, subclass that pass as
    ``docs/source/compiler-memory-planning.md`` describes and hand it this
    algorithm; the tags it writes are honored as pins.

    Note that a ``mem_id`` of 1 is a no-op tag under ``greedy`` -- it names the
    arena everything already lands in -- but here it is a pin to the first declared
    bank, which gets first claim on that bank's capacity. A pass written against
    ``greedy`` that tags buffers with 1 will constrain them to the fastest bank; drop
    those tags, or declare a bank 1 that can hold them.
    """
    if kwargs.get("share_mutable_buffers"):
        # run_multimethod hardcodes shared state to arena 2 and
        # _check_default_mem_ids requires every other buffer on arena 1, neither of
        # which a multi-bank plan can satisfy. Supporting it means placing shared
        # state one arena past the last planned one, which is a change to core
        # memory planning and belongs in its own review.
        raise ValueError(
            "share_mutable_buffers is not yet supported with banked memory "
            "planning: it reserves arena 2 for shared state and requires every "
            "other buffer on arena 1."
        )

    planner = BankedGreedy(
        target, allow_overlapping_allocations=allow_overlapping_allocations
    )
    return MemoryPlanningPass(
        memory_planning_algo=MemoryPlanningAlgorithmSuite(algo_list=[planner]),
        alignment=alignment,
        **kwargs,  # pyre-ignore[6]
    )


def planned_mem_ids(graph_module: torch.fx.GraphModule) -> List[int]:
    """The bank each planned buffer ended up in, read back off the graph."""
    seen: Set[int] = set()
    mem_ids: List[int] = []
    for node in graph_module.graph.nodes:
        for spec in get_node_tensor_specs(node):
            if id(spec) in seen or spec.const or spec.mem_id is None:
                continue
            seen.add(id(spec))
            mem_ids.append(spec.mem_id)
    return mem_ids


def format_placement_report(
    target: TargetMemoryMap,
    bufsizes: Sequence[int],
    mem_ids: Iterable[int] = (),
) -> str:
    """A per-bank summary of where a plan put things.

    Takes the arena sizes and the planned buffers' ``mem_id``s rather than reading
    state off a planner, so it is correct for any one method of a multi-method
    program. ``mem_ids`` comes from :func:`planned_mem_ids` when all you have is
    the exported program.
    """
    counts: Dict[int, int] = {}
    for mem_id in mem_ids:
        counts[mem_id] = counts.get(mem_id, 0) + 1

    lines = [
        "Memory bank placement:",
        f"  {'bank':<12}{'mem_id':>7}{'used':>12}{'size':>12}{'util':>8}{'buffers':>9}",
    ]
    for mem_id in target.mem_ids():
        bank = target.bank(mem_id)
        used = bufsizes[mem_id] if mem_id < len(bufsizes) else 0
        util = 100.0 * used / bank.size if bank.size else 0.0
        lines.append(
            f"  {bank.name:<12}{mem_id:>7}{used:>12}{bank.size:>12}"
            f"{util:>7.1f}%{counts.get(mem_id, 0):>9}"
        )
    total_used = sum(
        bufsizes[mem_id] if mem_id < len(bufsizes) else 0 for mem_id in target.mem_ids()
    )
    lines.append(f"  total planned: {total_used} bytes over {len(target.banks)} banks")
    return "\n".join(lines)
