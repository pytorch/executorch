# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import logging
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Sequence

import torch
from executorch.backends.cadence.aot.memory_constraints import (
    AbsolutePlacementConstraint,
    ConstraintsGenPass,
    GenerateMemConstraints,
    MemConstraints,
)
from executorch.backends.cadence.aot.utils import MemoryConfig
from executorch.exir.memory_planning import Verifier
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature


def get_aligned_offset(pre_aligned_offset: int, alignment: int) -> int:
    return int(math.ceil(pre_aligned_offset / alignment) * alignment)


class MemoryPlanningState:
    def __init__(self, memory_config: MemoryConfig) -> None:
        self.num_memories: int = len(memory_config.memory_sizes) + 1
        alignment = memory_config.memory_alignments
        assert alignment is not None
        assert len(alignment) == self.num_memories - 1
        self.alignment: list[int] = [1] + alignment
        # TODO: Maybe keep this sorted with heapq?
        self.allocated_buffers: list[list[TensorSpec]] = [
            [] for _ in range(self.num_memories)
        ]
        self.bufsizes: list[int] = [0] * self.num_memories

    def place_spec(self, spec: TensorSpec) -> None:
        """Place the spec at the given memory and offset."""
        logging.debug(f"Placing spec {spec}: {spec.mem_id=}, {spec.mem_offset=}")
        assert self.get_overlapping_spec(spec) is None
        self.allocated_buffers[spec.mem_id].append(spec)
        self.bufsizes[spec.mem_id] = max(
            self.bufsizes[spec.mem_id],
            get_aligned_offset(
                spec.mem_offset + spec.allocated_memory, self.alignment[spec.mem_id]
            ),
        )

    def get_overlapping_spec(self, spec: TensorSpec) -> Optional[TensorSpec]:
        """Get the overlapping spec for the given spec."""
        for allocated_spec in self.allocated_buffers[spec.mem_id]:
            if Verifier.lifetime_overlap(
                spec, allocated_spec
            ) and Verifier.storage_overlap(spec, allocated_spec):
                return allocated_spec
        return None

    def is_placed(self, spec: TensorSpec) -> bool:
        """Check if the spec is placed."""
        return spec.mem_id is not None and spec in self.allocated_buffers[spec.mem_id]

    def __str__(self) -> str:
        allocated_buffers_str = ""
        for i, specs in enumerate(self.allocated_buffers):
            allocated_buffers_str += (
                f"Memory {i}: "
                + ", ".join(
                    [
                        f"<{s.shape=} {s.mem_id=} {s.mem_offset=} {s.allocated_memory=}>"
                        for s in specs
                    ]
                )
                + "\n"
            )
        return f"MemoryPlanningState(bufsizes={self.bufsizes}, allocated_buffers={allocated_buffers_str})"


class MemoryPlanningAlgo(ABC):
    """Callable memory planning algorithm interface."""

    def __init__(
        self,
        memory_config: MemoryConfig,
        opt_level: int = 1,
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
        additional_constraint_gen_passes: Optional[Sequence[ConstraintsGenPass]] = None,
    ) -> None:
        self.memory_config: MemoryConfig = memory_config
        self.additional_constraint_gen_passes: Optional[
            Sequence[ConstraintsGenPass]
        ] = additional_constraint_gen_passes
        self.opt_level: int = opt_level
        self.alloc_graph_input: bool = alloc_graph_input
        self.alloc_graph_output: bool = alloc_graph_output
        self.memory_id_is_valid: list[bool] = [True] * self.get_num_memories()

    def get_num_memories(self) -> int:
        """Get num memories indexed from 1..N, compatible with EXIR's spec.mem_id."""
        return len(self.memory_config.memory_sizes) + 1

    def get_size(self, exir_id: int) -> int:
        # memory_space module provides num_memories indexed 0..num_memories-1.
        return self.memory_config.memory_sizes[exir_id - 1]

    def get_alignment(self, exir_id: int) -> int:
        # EXIR's spec.mem_id is indexed from 1..N.
        assert self.memory_config.memory_alignments is not None
        return self.memory_config.memory_alignments[exir_id - 1]

    def populate_constraints(
        self, graph_module: torch.fx.GraphModule
    ) -> tuple[MemoryPlanningState, MemConstraints]:
        """Populate the constraints for the memory planning algorithm."""
        state = MemoryPlanningState(self.memory_config)
        placement_constraints = MemConstraints(
            self.opt_level, self.alloc_graph_input, self.alloc_graph_output
        )
        GenerateMemConstraints(
            mem_constraints=placement_constraints,
            additional_constraint_gen_passes=self.additional_constraint_gen_passes,
        )(graph_module)
        return state, placement_constraints

    def is_valid_placement(
        self, spec: TensorSpec, placement_constraints: MemConstraints
    ) -> bool:
        """Returns true if the spec can be placed at the given memory id."""
        end_of_allocation = get_aligned_offset(
            spec.mem_offset + spec.allocated_memory,
            self.get_alignment(spec.mem_id),
        )
        return (
            self.memory_id_is_valid[spec.mem_id]
            and end_of_allocation <= self.get_size(spec.mem_id)
            and not placement_constraints.is_mem_id_in_blocklist(spec, spec.mem_id)
        )

    @contextmanager
    def block_memories_except(self, memory_id: int) -> Iterator[None]:
        """Block all memories except the given memory_id."""
        try:
            prev_valid = self.memory_id_is_valid.copy()
            self.memory_id_is_valid = [False] * self.get_num_memories()
            self.memory_id_is_valid[memory_id] = prev_valid[memory_id]
            yield
        finally:
            self.memory_id_is_valid = prev_valid

    @abstractmethod
    def plan(
        self,
        specs: Iterable[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
        extra_padding: int = 0,
    ) -> None:
        """Plan memory allocation for the given tensor specs."""
        pass

    def _place_pinned_specs(
        self,
        spec_with_abs_constraint: dict[
            TensorSpec, Optional[AbsolutePlacementConstraint]
        ],
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
    ) -> None:
        """Place pinned specs with fixed mem_id AND offset."""
        # All specs that have absolute constraints that pin spec to mem id and offset.
        pinned_specs = {
            spec: c
            for spec, c in spec_with_abs_constraint.items()
            if c is not None and c.offset is not None
        }
        for spec, constraint in pinned_specs.items():
            spec.mem_id = constraint.pinned_memory_id
            spec.mem_offset = constraint.offset
            state.place_spec(spec)
            placement_constraints.resolve_relative_loc_constraints(spec)

    def _place_memory_id_pinned_specs(
        self,
        spec_with_abs_constraint: dict[
            TensorSpec, Optional[AbsolutePlacementConstraint]
        ],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
        extra_padding: int = 0,
    ) -> None:
        """Callable interface for ET memory planning."""

        for mem_id in range(1, self.get_num_memories()):
            mem_id_pinned_specs: dict[TensorSpec, AbsolutePlacementConstraint] = {
                spec: c
                for spec, c in spec_with_abs_constraint.items()
                if c is not None and c.pinned_memory_id == mem_id and c.offset is None
            }
            logging.debug(f"Placing specs {mem_id_pinned_specs} for {mem_id=}")

            with self.block_memories_except(mem_id):
                self.plan(
                    mem_id_pinned_specs,
                    graph_module,
                    graph_signature,
                    state,
                    placement_constraints,
                    extra_padding,
                )

        for spec, constraint in spec_with_abs_constraint.items():
            if constraint is None:
                continue

            logging.debug(f"Placing spec {spec} with {constraint}")

            if not state.is_placed(spec):
                raise MemoryError(
                    f"Cannot fit {spec} in memory {constraint.pinned_memory_id}"
                )
            if (
                # Memory id is pinned, so we can't change it.
                spec.mem_id != constraint.pinned_memory_id
                or (
                    # Memory offset is pinned, so we can't change it.
                    constraint.offset is not None
                    and spec.mem_offset != constraint.offset
                )
            ):
                raise MemoryError(
                    f"Incorrect memory planning for {spec} with {spec.mem_id=} and {spec.mem_offset=} for constraint {constraint}"
                )
            # Resolve the relative constraints for the spec.
            placement_constraints.resolve_relative_loc_constraints(spec)

    def _place_specs_with_no_absolute_constraints(
        self,
        spec_with_abs_constraint: dict[
            TensorSpec, Optional[AbsolutePlacementConstraint]
        ],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
        extra_padding: int = 0,
    ) -> None:
        # Plan the memory allocation for specs without absolute or relative constraints.
        specs_without_relative_constraints = {
            spec: c
            for spec, c in spec_with_abs_constraint.items()
            if c is None and not placement_constraints.skipped_spec(spec)
        }
        self.plan(
            specs_without_relative_constraints,
            graph_module,
            graph_signature,
            state,
            placement_constraints,
            extra_padding,
        )

        for spec in specs_without_relative_constraints:
            # And now honor the various memory location constraints (i.e., infer the memory
            # location of tensors in skip_specs from the constraints) for this spec.
            placement_constraints.resolve_relative_loc_constraints(spec)

    def plan_with_constraints(
        self,
        specs: Iterable[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
        extra_padding: int = 0,
    ) -> None:
        """Callable interface for ET memory planning."""

        spec_and_abs_constraints = {
            spec: placement_constraints.get_absolute_placement_constraint(spec)
            for spec in specs
        }

        # Place specs that have both mem_id and offset constraints.
        self._place_pinned_specs(spec_and_abs_constraints, state, placement_constraints)

        # Place specs that have both mem_id constraints.
        self._place_memory_id_pinned_specs(
            spec_and_abs_constraints,
            graph_module,
            graph_signature,
            state,
            placement_constraints,
            extra_padding,
        )

        # Place specs that have no constraints.
        self._place_specs_with_no_absolute_constraints(
            spec_and_abs_constraints,
            graph_module,
            graph_signature,
            state,
            placement_constraints,
            extra_padding,
        )

    def __call__(
        self,
        alignment: int,
        specs: Iterable[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        extra_padding: int = 0,
    ) -> list[int]:
        """Callable interface for ET memory planning."""

        # Initialize state and constraints.
        state, placement_constraints = self.populate_constraints(graph_module)

        self.plan_with_constraints(
            specs,
            graph_module,
            graph_signature,
            state,
            placement_constraints,
            extra_padding,
        )

        # At the end, all the keys in relative_loc_constraints should have been visited
        # and emptied.
        assert not placement_constraints.relative_loc_constraints_exist()

        logging.debug(f"Memory planning algo found bufsizes: {state.bufsizes}")
        return state.bufsizes
