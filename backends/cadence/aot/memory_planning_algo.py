# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from executorch.backends.cadence.aot.memory_constraints import (
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
        return spec in self.allocated_buffers[spec.mem_id]


class MemoryPlanningAlgo(ABC):
    """Callable memory planning algorithm interface."""

    def __init__(
        self,
        memory_config: MemoryConfig,
        placement_constraints: MemConstraints,
        additional_constraint_gen_passes: Optional[Sequence[ConstraintsGenPass]] = None,
    ) -> None:
        self.memory_config: MemoryConfig = memory_config
        self.placement_constraints: MemConstraints = placement_constraints
        self.additional_constraint_gen_passes: Optional[
            Sequence[ConstraintsGenPass]
        ] = additional_constraint_gen_passes

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

    def populate_constraints(self, graph_module: torch.fx.GraphModule) -> None:
        """Populate the constraints for the memory planning algorithm."""
        GenerateMemConstraints(
            mem_constraints=self.placement_constraints,
            additional_constraint_gen_passes=self.additional_constraint_gen_passes,
        )(graph_module)

    def is_valid_placement(self, spec: TensorSpec) -> bool:
        """Returns true if the spec can be placed at the given memory id."""
        end_of_allocation = get_aligned_offset(
            spec.mem_offset + spec.allocated_memory,
            self.get_alignment(spec.mem_id),
        )
        return end_of_allocation <= self.get_size(
            spec.mem_id
        ) and not self.placement_constraints.is_mem_id_in_blocklist(spec, spec.mem_id)

    @abstractmethod
    def plan(
        self,
        specs: set[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        extra_padding: int = 0,
        prev_state: Optional[MemoryPlanningState] = None,
    ) -> MemoryPlanningState:
        """Plan memory allocation for the given tensor specs."""
        pass

    def __call__(
        self,
        alignment: int,
        specs: set[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        extra_padding: int = 0,
    ) -> list[int]:
        """Callable interface for ET memory planning."""
        self.populate_constraints(graph_module)

        # First plan the memory allocation for specs without relative constraints.
        specs_without_relative_constraints = set(
            filter(
                lambda spec: not self.placement_constraints.skipped_spec(spec),
                specs,
            )
        )

        # Call memory planning to get bufsizes.
        state = self.plan(
            specs_without_relative_constraints,
            graph_module,
            graph_signature,
            extra_padding,
        )

        for spec in specs_without_relative_constraints:
            # And now honor the various memory location constraints (i.e., infer the memory
            # location of tensors in skip_specs from the constraints) for this spec.
            self.placement_constraints.resolve_relative_loc_constraints(spec)

        # At the end, all the keys in relative_loc_constraints should have been visited
        # and emptied.
        assert not self.placement_constraints.relative_loc_constraints_exist()

        logging.debug(f"Memory planning algo found bufsizes: {state.bufsizes}")
        return state.bufsizes
