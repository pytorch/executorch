# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections import defaultdict
from collections.abc import Collection, Iterator
from functools import partial
from math import isfinite
from typing import Callable, cast, Optional, TYPE_CHECKING

import torch
from executorch.backends.arm.arm_vela import VelaIO, VelaIOLayout
from executorch.exir import memory
from executorch.exir.delegate import executorch_call_delegate, is_lowered_module
from executorch.exir.error import internal_assert, InternalError
from executorch.exir.memory_planning import (
    AllocationSpec,
    MemoryAlgoResult,
    MemoryPlanningAlgorithmSuite,
    SpecAllocResult,
    Verifier,
)
from executorch.exir.pass_base import PassResult
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature

if TYPE_CHECKING:
    from executorch.exir.lowered_backend_module import LoweredBackendModule

logger = logging.getLogger(__name__)


def _can_alias_to_scratch(
    spec: TensorSpec,
    scratch_spec: TensorSpec,
    io: VelaIO,
    alignment: int,
) -> bool:
    if spec.const or not spec.is_static_shape_tensor:
        return False
    allocated_size = spec.realign(alignment)
    return (
        spec.nbytes() == io.size
        and spec.mem_id in (None, scratch_spec.mem_id)
        and spec.device == scratch_spec.device
        and spec.device_index == scratch_spec.device_index
        and io.offset % alignment == 0
        and io.offset + allocated_size <= scratch_spec.nbytes()
        and spec.storage_base in (None, scratch_spec)
        and (spec.storage_base is None or spec.storage_base_offset == io.offset)
    )


def _get_scratch_io_layout(
    graph_module: torch.fx.GraphModule,
    delegate: torch.fx.Node,
) -> Optional[tuple[VelaIOLayout, TensorSpec]]:
    if not delegate.args or not isinstance(delegate.args[0], torch.fx.Node):
        return None
    lowered_node = delegate.args[0]
    if lowered_node.op != "get_attr" or not isinstance(lowered_node.target, str):
        return None
    lowered_module = graph_module.get_submodule(lowered_node.target)
    if not is_lowered_module(lowered_module):
        return None
    lowered_module = cast("LoweredBackendModule", lowered_module)
    if lowered_module.backend_id != "EthosUBackend":
        return None

    scratch = memory.delegate_scratch(delegate)
    if not scratch:
        logger.warning(
            "Ethos-U delegate %s has no planned scratch. Set "
            "plan_delegate_scratch=True in EthosUCompileSpec to enable "
            "scratch-based I/O placement. Using ordinary planning for this delegate.",
            delegate.name,
        )
        return None

    metadata = (lowered_module.meta or {}).get("_delegate_info_meta")
    if not isinstance(metadata, dict):
        return None
    layout = metadata.get("vela_io")
    if not isinstance(layout, VelaIOLayout) or any(
        io.region != 1 for io in layout.inputs + layout.outputs
    ):
        return None

    if (
        len(scratch) != 1
        or scratch[0].label != "ethosu_scratch"
        or scratch[0].spec.nbytes() != layout.scratch_size
        or len(delegate.args) != 1 + len(layout.inputs)
    ):
        return None
    return layout, scratch[0].spec


def _delegate_io_specs(
    delegate: torch.fx.Node,
    layout: VelaIOLayout,
) -> Iterator[tuple[TensorSpec, VelaIO]]:
    for input_node, io in zip(delegate.args[1:], layout.inputs):
        if not isinstance(input_node, torch.fx.Node):
            continue
        if any(user is not delegate for user in input_node.users):
            continue
        input_spec = input_node.meta.get("spec")
        if isinstance(input_spec, TensorSpec):
            yield input_spec, io

    output_specs = delegate.meta.get("spec")
    if isinstance(output_specs, TensorSpec):
        output_specs = (output_specs,)
    if not isinstance(output_specs, (list, tuple)) or len(output_specs) != len(
        layout.outputs
    ):
        return

    for output_spec, io in zip(output_specs, layout.outputs):
        if not isinstance(output_spec, TensorSpec):
            continue
        yield output_spec, io


def _collect_scratch_io_candidates(
    graph_module: torch.fx.GraphModule,
    alignment: int,
    specs: Collection[TensorSpec],
) -> dict[TensorSpec, Optional[tuple[TensorSpec, int]]]:
    candidates: dict[TensorSpec, Optional[tuple[TensorSpec, int]]] = {}
    for delegate in graph_module.graph.nodes:
        if (
            delegate.op != "call_function"
            or delegate.target != executorch_call_delegate
        ):
            continue
        delegate_layout = _get_scratch_io_layout(graph_module, delegate)
        if delegate_layout is None:
            continue
        layout, scratch_spec = delegate_layout
        if scratch_spec not in specs:
            continue
        for spec, io in _delegate_io_specs(delegate, layout):
            if spec not in specs:
                continue
            if candidates.get(spec) is None:
                # Keep rejected attempts for the report and the first valid owner.
                candidates[spec] = (
                    (scratch_spec, io.offset)
                    if _can_alias_to_scratch(spec, scratch_spec, io, alignment)
                    else None
                )
    return candidates


def _get_storage_groups(
    alignment: int, specs: Collection[TensorSpec]
) -> dict[TensorSpec, list[AllocationSpec]]:
    groups: dict[TensorSpec, list[AllocationSpec]] = {}
    for spec in specs:
        spec.realign(alignment)
        root = spec
        offset = 0
        while root.storage_base is not None:
            base = root.storage_base
            internal_assert(base in specs, "Storage base must be memory planned.")
            base.realign(alignment)
            internal_assert(
                root.storage_base_offset >= 0
                and root.storage_base_offset + root.allocated_memory
                <= base.allocated_memory,
                "Storage-backed TensorSpec must fit within storage_base.",
            )
            internal_assert(
                root.mem_id in (None, base.mem_id)
                and root.device == base.device
                and root.device_index == base.device_index,
                "Storage-backed TensorSpec must use its base's memory pool.",
            )
            offset += root.storage_base_offset
            root = base
        internal_assert(
            offset % alignment == 0,
            "Storage-backed TensorSpec must respect alignment.",
        )
        groups.setdefault(root, []).append(AllocationSpec(offset, spec))

    for members in groups.values():
        for index, lhs in enumerate(members):
            for rhs in members[index + 1 :]:
                if (
                    lhs.offset + lhs.spec.allocated_memory <= rhs.offset
                    or rhs.offset + rhs.spec.allocated_memory <= lhs.offset
                    or not Verifier.lifetime_overlap(lhs.spec, rhs.spec)
                ):
                    continue
                internal_assert(
                    lhs.spec in Verifier._storage_root_chain(rhs.spec)
                    or rhs.spec in Verifier._storage_root_chain(lhs.spec),
                    "Storage group has overlapping live sibling ranges.",
                )

    return groups


def _greedy_by_storage_group(
    alignment: int,
    specs: Collection[TensorSpec],
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    extra_padding: int,
    *,
    scratch_specs: set[TensorSpec],
    prioritize_scratch: bool,
) -> MemoryAlgoResult:
    """Greedily place storage groups at the lowest compatible addresses.

    A group contains an owning tensor and tensors sharing its storage at fixed
    relative offsets. Ordinary storage bases retain their extended lifetimes.
    Ethos-U scratch reserves its full range during the delegate call; aliases
    independently protect the slices live outside that interval.

    1. Sort groups by descending owner size. When prioritize_scratch is set,
       place delegate scratch groups before other groups.
    2. For each group, select its memory pool and start after any reserved
       prefix, rounded up to alignment.
    3. Build candidate group addresses from that starting address and positions
       that put a member after a previously placed tensor with an overlapping
       lifetime. Subtract the member's relative offset from the previous
       tensor's end address, then round up to alignment.
    4. Try candidates in ascending address order. Accept the first where no
       member overlaps a previously placed tensor in both bytes and lifetime.
       Tensors with disjoint lifetimes may reuse bytes.
    5. Record each member's address and update the pool's extent. Repeat for
       the remaining groups without moving earlier placements.
    6. Add extra padding to each used pool and return the allocations and
       arena sizes.

    There is no backtracking, so the result need not be optimal.

    """
    groups = _get_storage_groups(alignment, specs)
    ordered_groups = sorted(
        groups,
        key=lambda root: (
            prioritize_scratch and root in scratch_specs,
            root.allocated_memory,
        ),
        reverse=True,
    )
    result = MemoryAlgoResult({}, [0, 0])
    occupied: dict[int, list[AllocationSpec]] = defaultdict(list)
    input_sizes = getattr(graph_module, "input_mem_buffer_sizes", None) or []
    for root in ordered_groups:
        mem_id = root.mem_id if root.mem_id is not None else 1
        members = groups[root]
        pool_start = input_sizes[mem_id] if mem_id < len(input_sizes) else 0
        offsets = {((pool_start + alignment - 1) // alignment) * alignment}
        for member in members:
            for previous in occupied[mem_id]:
                if not Verifier.lifetime_overlap(member.spec, previous.spec):
                    continue
                offset = (
                    previous.offset + previous.spec.allocated_memory - member.offset
                )
                if offset >= pool_start:
                    offsets.add(((offset + alignment - 1) // alignment) * alignment)
        for offset in sorted(offsets):
            if all(
                not Verifier.lifetime_overlap(member.spec, previous.spec)
                or offset + member.offset + member.spec.allocated_memory
                <= previous.offset
                or previous.offset + previous.spec.allocated_memory
                <= offset + member.offset
                for member in members
                for previous in occupied[mem_id]
            ):
                break
        else:
            raise InternalError("Could not place storage group.")
        if mem_id >= len(result.bufsizes):
            result.bufsizes.extend([0] * (mem_id + 1 - len(result.bufsizes)))
        result.bufsizes[mem_id] = max(
            result.bufsizes[mem_id], offset + root.allocated_memory
        )
        for member in members:
            member_offset = offset + member.offset
            result.spec_dict[member.spec] = SpecAllocResult(mem_id, 0, member_offset)
            occupied[mem_id].append(AllocationSpec(member_offset, member.spec))
    for mem_id in occupied:
        result.bufsizes[mem_id] += extra_padding
    return result


class EthosUPreferredAlgorithmSuite:
    """Prefer eligible Ethos-U direct I/O plans over an ordinary baseline.

    Run the baseline once, then place compatible I/O inside declared scratch
    using each preferred algorithm. Discard a preferred result if any arena
    exceeds either growth limit relative to the baseline. Select the eligible
    preferred result with the smallest total arena size, or restore the ordinary
    plan if none qualifies. Invalid preferred placements are skipped; errors
    from the baseline propagate.

    Like MemoryPlanningAlgorithmSuite, apply the selected allocation addresses
    to the tensor specs and return the arena sizes.

    """

    def __init__(
        self,
        preferred_algos: dict[str, Callable[..., MemoryAlgoResult]],
        baseline: Callable[..., list[int]],
        *,
        max_arena_growth_percent: float,
        max_arena_growth_bytes: int,
    ) -> None:
        """Configure the preferred algorithms, baseline, and per-arena limits.

        Args:
            preferred_algos (dict[str, Callable]): Named allocation algorithms
                preferred over the baseline when they satisfy both growth
                limits. Names identify candidates in the INFO-level report.
            baseline (Callable): Ordinary allocator for comparison and fallback.
                Applies allocation addresses and returns arena sizes.
            max_arena_growth_percent (float): Maximum growth per arena as a
                finite, nonnegative percentage of the baseline arena size.
            max_arena_growth_bytes (int): Nonnegative maximum growth per arena
                in bytes. Both limits apply.

        """
        self.preferred_algos = preferred_algos
        self.baseline = baseline
        self.max_arena_growth_percent = max_arena_growth_percent
        self.max_arena_growth_bytes = max_arena_growth_bytes

    def __call__(
        self,
        alignment: int,
        specs: Collection[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        extra_padding: int,
    ) -> list[int]:
        candidates = _collect_scratch_io_candidates(graph_module, alignment, specs)
        aliases = {
            spec: placement
            for spec, placement in candidates.items()
            if placement is not None
        }
        fallback_storage = {
            spec: (spec.storage_base, spec.storage_base_offset, list(spec.lifetime))
            for spec in aliases
        }
        original_mem_ids = {spec: spec.mem_id for spec in specs}
        sizes = self.baseline(
            alignment, specs, graph_module, graph_signature, extra_padding
        )
        baseline = MemoryAlgoResult(
            {
                spec: SpecAllocResult(
                    cast(int, spec.mem_id),
                    cast(int, spec.mem_obj_id),
                    cast(int, spec.mem_offset),
                )
                for spec in specs
            },
            list(sizes),
        )
        for spec, mem_id in original_mem_ids.items():
            spec.mem_id = mem_id
        for spec, (base, offset) in aliases.items():
            spec.storage_base, spec.storage_base_offset = base, offset
            # Scratch protects the delegate call; consumed inputs only need
            # their own reservation before the call starts.
            lifetime = cast(list[int], spec.lifetime)
            if lifetime[0] < lifetime[1] == base.lifetime[0]:
                lifetime[1] -= 1

        result = (
            self._select(
                alignment,
                specs,
                graph_module,
                graph_signature,
                extra_padding,
                baseline=baseline,
                scratch_specs={base for base, _ in aliases.values()},
            )
            if aliases
            else baseline
        )
        if result is baseline:
            for spec, (
                original_base,
                original_offset,
                original_lifetime,
            ) in fallback_storage.items():
                spec.storage_base = original_base
                spec.storage_base_offset = original_offset
                spec.lifetime = original_lifetime
        for spec, allocation in result.spec_dict.items():
            spec.mem_id = allocation.mem_id  # type: ignore[assignment]
            spec.mem_obj_id = allocation.mem_obj_id  # type: ignore[assignment]
            spec.mem_offset = allocation.mem_offset  # type: ignore[assignment]
        if logger.isEnabledFor(logging.INFO):
            if not aliases:
                logger.info(
                    "Ordinary planning selected: no eligible direct I/O placements; "
                    "arena sizes=%s bytes",
                    result.bufsizes,
                )
            placed = sum(
                (spec.storage_base, spec.storage_base_offset) == placement
                for spec, placement in aliases.items()
            )
            logger.info(
                "Scratch placement: %d/%d attempted tensors placed; "
                "declared delegate scratch remains planned",
                placed,
                len(candidates),
            )
        return result.bufsizes

    def _select(
        self,
        alignment: int,
        specs: Collection[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        extra_padding: int,
        *,
        baseline: MemoryAlgoResult,
        scratch_specs: set[TensorSpec],
    ) -> MemoryAlgoResult:
        logger.info(
            "Memory planning for %s: ordinary arena sizes=%s bytes",
            graph_module._get_name(),
            baseline.bufsizes,
        )
        best: Optional[MemoryAlgoResult] = None
        best_name = "ordinary"
        for name, algo in self.preferred_algos.items():
            try:
                candidate = algo(
                    alignment,
                    specs,
                    graph_module,
                    graph_signature,
                    extra_padding,
                    scratch_specs=scratch_specs,
                )
            except InternalError as error:
                logger.info("%s: rejected (invalid placement)", name)
                logger.debug("%s: %s", name, error)
                continue
            logger.info("%s: arena sizes=%s bytes", name, candidate.bufsizes)
            for index, size in enumerate(candidate.bufsizes):
                if index >= len(baseline.bufsizes):
                    logger.info("%s: rejected (new arena %d)", name, index)
                    break
                allowance = min(
                    int(baseline.bufsizes[index] * self.max_arena_growth_percent / 100),
                    self.max_arena_growth_bytes,
                )
                growth = size - baseline.bufsizes[index]
                if growth > allowance:
                    logger.info(
                        "%s: rejected (arena %d grows by %d bytes; allowance=%d bytes)",
                        name,
                        index,
                        growth,
                        allowance,
                    )
                    break
            else:
                if best is None or sum(candidate.bufsizes) < sum(best.bufsizes):
                    best, best_name = candidate, name
        result = baseline if best is None else best
        logger.info("Selected %s: arena sizes=%s bytes", best_name, result.bufsizes)
        return result


class EthosUMemoryPlanning(MemoryPlanningPass):
    """Places compatible I/O in the declared Ethos-U scratch buffer.

    Use as ExecutorchBackendConfig.memory_planning_pass. Scratch remains
    delegate metadata and is delivered through the execution context. Each
    aliased slice protects its own live range, leaving dead scratch ranges
    available to other tensors. Compared with ordinary planning using the
    configured allocator, each arena may grow by at most the configured
    percentage and byte limits. Defaults are 1% and 1 KiB per arena. The runtime
    requires the scratch arena base to be aligned to 16 bytes. Incompatible
    tensors retain separate allocations and are copied at execution. Falling
    back removes I/O aliases but keeps declared scratch memory planned.

    Enable this module's INFO logger for candidate arena sizes, selection and
    rejection reasons, and placed/attempted tensor counts.

    """

    def __init__(
        self,
        *,
        baseline_algo: Optional[Callable[..., list[int]]] = None,
        max_arena_growth_percent: float = 1.0,
        max_arena_growth_bytes: int = 1024,
        **kwargs,
    ) -> None:
        """Configure direct I/O growth limits and ordinary planning options.

        Args:
            baseline_algo (Callable | None): Ordinary allocator used for arena
                comparison and fallback. Defaults to MemoryPlanningAlgorithmSuite.
            max_arena_growth_percent (float): Maximum arena growth relative to
                ordinary planning, as a percentage. Must be finite and
                nonnegative. Defaults to 1.0.
            max_arena_growth_bytes (int): Maximum arena growth in bytes.
                Must be nonnegative. Defaults to 1024.
            **kwargs: Keyword arguments for MemoryPlanningPass. Alignment must
                be a positive multiple of 16 bytes.

        """
        if not isfinite(max_arena_growth_percent) or max_arena_growth_percent < 0:
            raise ValueError("max_arena_growth_percent must be finite and nonnegative")
        if max_arena_growth_bytes < 0:
            raise ValueError("max_arena_growth_bytes must be nonnegative")
        if baseline_algo is None:
            baseline_algo = MemoryPlanningAlgorithmSuite()
        super().__init__(
            memory_planning_algo=EthosUPreferredAlgorithmSuite(
                preferred_algos={
                    "direct I/O (largest-first)": partial(
                        _greedy_by_storage_group, prioritize_scratch=False
                    ),
                    "direct I/O (scratch-first)": partial(
                        _greedy_by_storage_group, prioritize_scratch=True
                    ),
                },
                baseline=baseline_algo,
                max_arena_growth_percent=max_arena_growth_percent,
                max_arena_growth_bytes=max_arena_growth_bytes,
            ),
            **kwargs,
        )
        if self.alignment <= 0 or self.alignment % 16 != 0:
            raise ValueError(
                "Ethos-U planning alignment must be a positive multiple of 16"
            )

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        # The ordinary pass only verifies storage reuse when DEBUG is enabled.
        result = super().run(graph_module, graph_signature)
        verifier = Verifier(
            graph_module,
            self.alloc_graph_input,
            self.alloc_graph_output,
            self.alloc_mutable_buffers and not self.share_mutable_buffers,
            graph_signature,
        )
        verifier.verify_storage_reuse(self.allow_lifetime_and_storage_overlap)
        return result
