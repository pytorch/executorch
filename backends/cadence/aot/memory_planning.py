# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import collections
import itertools
import logging
from typing import Callable, Iterable, Optional, Sequence, TypeAlias

import torch
from executorch.backends.cadence.aot.memory_constraints import MemConstraints
from executorch.backends.cadence.aot.memory_planning_algo import (
    ConstraintsGenPass,
    get_aligned_offset,
    MemoryPlanningAlgo,
    MemoryPlanningState,
)
from executorch.backends.cadence.aot.utils import (
    MemoryConfig,
    MemoryPlanningAlgoFailure,
)

from executorch.exir import ExecutorchProgramManager
from executorch.exir.memory_planning import collect_specs_from_nodes, Verifier
from executorch.exir.pass_base import PassBase
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.tensor import TensorSpec
from tabulate import tabulate
from torch.export.exported_program import ExportGraphSignature
from torch.fx.passes.infra.pass_base import PassResult


def collect_specs_from_graph_module(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
) -> Iterable[TensorSpec]:
    """
    Return the specs for all the nodes in the graph module in
    topological order.
    """
    # Collect the specs from all the nodes in the graph module, and return it
    return collect_specs_from_nodes(
        graph_module.graph.nodes,
        graph_signature,
        ignore_graph_input=not alloc_graph_input,
        ignore_graph_output=not alloc_graph_output,
    )


class PositionBasedGreedyWithHierarchy(MemoryPlanningAlgo):
    """Greedily place tensor in the fastest memory available."""

    def plan_spec(
        self,
        spec: TensorSpec,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
    ) -> None:
        """
        Greedily place the spec in the first memory that can fit it.
        """
        for spec.mem_id in range(1, self.get_num_memories()):
            spec.mem_offset = 0
            while self.is_valid_placement(spec, placement_constraints) and (
                overlapped := state.get_overlapping_spec(spec)
            ):
                # Found an overlapping spec, so we need to adjust the offset = end of the overlapping spec + alignment.
                spec.mem_offset = get_aligned_offset(
                    overlapped.mem_offset + overlapped.allocated_memory,
                    self.get_alignment(spec.mem_id),
                )

            if self.is_valid_placement(spec, placement_constraints):
                # Found a valid `spec.mem_offset` which is both valid and has no overlap.
                state.place_spec(spec)
                break

    def plan(
        self,
        specs: Iterable[TensorSpec],
        graph_module: torch.fx.GraphModule,
        graph_signature: ExportGraphSignature,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
        extra_padding: int = 0,
    ) -> None:

        # Iterate over all the specs in sorted order
        for spec in sorted(
            specs,
            key=lambda spec: spec.allocated_memory,
            reverse=True,
        ):
            self.plan_spec(spec, state, placement_constraints)
            if not state.is_placed(spec):
                raise MemoryPlanningAlgoFailure(
                    f"Cannot fit {spec} {spec.allocated_memory=} in any memory hierarchy for {self.memory_config}"
                )


class GreedyWithHeuristic(MemoryPlanningAlgo):
    """Greedy tensor placement with the heuristics from arxiv.org/pdf/2001.03288.pdf."""

    def plan_spec(
        self,
        spec: TensorSpec,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
    ) -> None:
        """
        Greedily place the spec in the first memory that can fit it.
        """
        for spec.mem_id in range(1, self.get_num_memories()):
            if placement_constraints.is_mem_id_in_blocklist(spec, spec.mem_id):
                # Skip placement for blocked memory id.
                continue
            prev_offset, smallest_gap = 0, float("inf")
            for allocated_spec in state.allocated_buffers[spec.mem_id]:
                if not Verifier.lifetime_overlap(spec, allocated_spec):
                    continue

                if (
                    gap := allocated_spec.mem_offset - prev_offset
                ) >= spec.allocated_memory and gap < smallest_gap:
                    smallest_gap = gap
                    spec.mem_offset = prev_offset
                # Note that different from the paper, which updates prev_offset for all
                # allocated tensors, we only update tensors with overlapping lifetime.
                # Updating prev_offset outside the if statement will include tensors without
                # overlapping lifetime, causing unnecessary waste of memory and make the
                # calculation of gap incorrect. Moving it out will make the algorithm degenerate
                # to the naive one, reusing 0 tensor. The paper may have a typo here.
                prev_offset = max(
                    get_aligned_offset(
                        allocated_spec.mem_offset + allocated_spec.allocated_memory,
                        self.get_alignment(spec.mem_id),
                    ),
                    prev_offset,
                )
            if spec.mem_offset is None:
                spec.mem_offset = prev_offset

            if not self.is_valid_placement(spec, placement_constraints):
                # Skip placement for invalid memory id.
                spec.mem_offset = None
                continue

            state.place_spec(spec)
            # A data structure used for maintaining the tensor order
            # by offset, named ordered_allocated_ids in the paper
            state.allocated_buffers[spec.mem_id].sort(key=lambda spec: spec.mem_offset)
            break

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
        # We do not use the `alignment` parameter and instead use the per-memory alignment
        # constraints from `memory_config`.

        # Iterate over all the specs in sorted order
        for spec in sorted(
            specs,
            key=lambda spec: spec.allocated_memory,
            reverse=True,
        ):
            self.plan_spec(spec, state, placement_constraints)
            if not state.is_placed(spec):
                raise MemoryPlanningAlgoFailure(
                    f"Cannot fit {spec} in any memory hierarchy for {self.memory_config}"
                )

        logging.debug(
            f"greedy by size for offset calculation with hierarchy returns bufsizes: {state.bufsizes}"
        )


def find_peak_memory_usages_per_memory(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
    mem_constraints: Optional[MemConstraints] = None,
) -> list[int]:
    """
    Given a GraphModule with a memory plan, find the peak memory usages for each memory
    in the memory hierarchy.
    """
    # Create a defaultdict to keep track of memory usages: {mem_id: mem_usage}
    # Use a defaultdict here because we don't know how many unique memory_id in
    # the memory hierarchy used in memory planning.
    usages = collections.defaultdict(int)

    # go through all nodes in the graph, collect memory usage per spec.mem_id
    for spec in collect_specs_from_graph_module(
        graph_module, graph_signature, alloc_graph_input, alloc_graph_output
    ):
        if mem_constraints is not None and mem_constraints.skipped_spec(spec):
            continue
        usages[spec.mem_id] = max(
            usages[spec.mem_id], spec.mem_offset + spec.allocated_memory
        )

    # Convert usages dictionary into list of len of max memory id
    # Ex: {1: 20, 3:30} -> [0, 20, 0, 30].
    #                       ^   ^  ^   ^
    #                       |   |  |   |_  mem_id 3
    #                       |   |  |_ mem_id 2
    #                       |   |_ mem_id 1
    #                       |_ mem_id 0
    max_mem_id = max(usages.keys(), default=0)
    usages = [usages[i] for i in range(1, max_mem_id + 1)]

    return usages


def find_peak_memory_usage(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
    mem_constraints: Optional[MemConstraints] = None,
) -> tuple[int, int]:
    """
    Given a GraphModule with a memory plan, find the peak usage over time across all
    memories in the memory hierarchy. The resulting peak memory usage should be:
    1. >= min(find_peak_memory_usages_per_memory(graph_module))
    2. <= sum(find_peak_memory_usages_per_memory(graph_module))
    """
    # memory allocations over time (measured in nodex index)
    byte_allocated = [0] * (len(graph_module.graph.nodes) + 1)

    # Iterate over all the node specs
    for spec in collect_specs_from_graph_module(
        graph_module, graph_signature, alloc_graph_input, alloc_graph_output
    ):
        if spec.lifetime[0] is None or (
            mem_constraints is not None and mem_constraints.skipped_spec(spec)
        ):
            continue

        # lifetime is [start, end], both ends inclusive
        start, end = spec.lifetime
        byte_allocated[start] += spec.allocated_memory
        byte_allocated[end + 1] -= spec.allocated_memory

    # accumulate the bytes allocated/deallocated to get memory usages
    memory_usages = list(itertools.accumulate(byte_allocated))

    # find the peak memory usage and the index
    peak_memory_usage = max(memory_usages, default=0)
    peak_memory_usage_node_idx = (
        memory_usages.index(peak_memory_usage) if memory_usages else 0
    )

    return peak_memory_usage, peak_memory_usage_node_idx


# Print two tables with relevant memory planning information
#
# Per Memory Space Usage Table:
# +--------------------------------------+----------------+-----------------------+-----------------------------+
# | Memory Space                         |   Base Address |   Memory Size (Bytes) |   Peak Memory Usage (Bytes) |
# +======================================+================+=======================+=============================+
# | MEMORY SPACE A                       |     0x57be0000 |                 65213 |                       64544 |
# | MEMORY SPACE B                       |     0x57bf0000 |                 65521 |                       36864 |
# | MEMORY SPACE ...                     |            ... |                   ... |                         ... |
# +--------------------------------------+----------------+-----------------------+-----------------------------+
#
# Total Memory Space Usage Table:
# +-------------------------------------+---------------+---------+
# | Peak memory usage across all spaces | 2380032 bytes | Node 86 |
# +-------------------------------------+---------------+---------+
def print_memory_planning_info(
    executorch_prog: ExecutorchProgramManager,
    memory_config: MemoryConfig,
    opt_level: int,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
) -> None:
    # Get the peak memory usages per memory space
    mem_constraints = MemConstraints(
        opt_level=opt_level,
        alloc_graph_input=alloc_graph_input,
        alloc_graph_output=alloc_graph_output,
    )
    # Get the peak memory usages per memory space
    peak_memory_usages_per_memory = find_peak_memory_usages_per_memory(
        executorch_prog.exported_program().graph_module,
        executorch_prog.exported_program().graph_signature,
        alloc_graph_input,
        alloc_graph_output,
        mem_constraints,
    )

    # Create a table of memory spaces and their base addresses, total memory sizes, and peak memory usage
    memory_names, base_addrs = memory_config.memory_names, memory_config.base_addrs
    memory_usage_table = [
        [
            f"{(i + 1) if memory_names is None else memory_names[i]}",
            None if base_addrs is None else hex(base_addrs[i]),
            memory_config.memory_sizes[i],
            peak_memory_usages_per_memory[i],
        ]
        for i in range(len(peak_memory_usages_per_memory))
    ]

    # Print the memory usage per memory space as a table
    logging.info(
        "\n"
        + tabulate(
            memory_usage_table,
            headers=[
                "Memory Space",
                "Base Address",
                "Memory Size (Bytes)",
                "Peak Memory Usage (Bytes)",
            ],
            tablefmt="outline",
        )
    )

    # Get the total peak memory usage across all memory spaces
    total_peak_memory_usage = find_peak_memory_usage(
        executorch_prog.exported_program().graph_module,
        executorch_prog.exported_program().graph_signature,
        alloc_graph_input,
        alloc_graph_output,
        mem_constraints,
    )

    # Create a table with total peak memory usage and node at which this occurs
    total_memory_usage_table = [
        [
            "Peak memory usage across all spaces",
            f"{total_peak_memory_usage[0]} bytes",
            f"Node {total_peak_memory_usage[1]}",
        ]
    ]

    # Print the total memory usage as a table
    logging.info(
        "\n"
        + tabulate(
            total_memory_usage_table,
            tablefmt="outline",
        )
    )


class SimplifyIdmaOpsPass(PassBase):
    """Replace idma_load and idma_store with idma_copy."""

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:
        modified = False
        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.cadence.idma_load.out
        ):
            modified = True
            node.target = torch.ops.cadence.idma_copy.out
            node.args = (node.args[0], *node.args[2:])

        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.cadence.idma_store.out
        ):
            modified = True
            node.target = torch.ops.cadence.idma_copy.out

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, modified)


ConstraintGenPassType: TypeAlias = Callable[
    [MemConstraints],
    Callable[[torch.fx.GraphModule], Optional[PassResult]],
]


class CadenceMemoryPlanning:
    def __init__(
        self,
        memory_config: MemoryConfig,
        opt_level: int,
        mem_algo: int,
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
        additional_constraint_gen_passes: Optional[Sequence[ConstraintsGenPass]] = None,
    ) -> None:
        self.memory_config = memory_config
        self.opt_level = opt_level
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output

        self.algo: MemoryPlanningAlgo = self.get_mem_algos(
            memory_config,
            opt_level,
            alloc_graph_input,
            alloc_graph_output,
            additional_constraint_gen_passes,
        )[mem_algo]

    @staticmethod
    def get_mem_algos(
        memory_config: MemoryConfig,
        opt_level: int,
        alloc_graph_input: bool,
        alloc_graph_output: bool,
        additional_constraint_gen_passes: Optional[Sequence[ConstraintsGenPass]],
    ) -> list[MemoryPlanningAlgo]:
        return [
            PositionBasedGreedyWithHierarchy(
                memory_config=memory_config,
                opt_level=opt_level,
                alloc_graph_input=alloc_graph_input,
                alloc_graph_output=alloc_graph_output,
                additional_constraint_gen_passes=additional_constraint_gen_passes,
            ),
            GreedyWithHeuristic(
                memory_config=memory_config,
                opt_level=opt_level,
                alloc_graph_input=alloc_graph_input,
                alloc_graph_output=alloc_graph_output,
                additional_constraint_gen_passes=additional_constraint_gen_passes,
            ),
        ]

    def __call__(
        self,
        graph_module: torch.fx.GraphModule,
    ) -> PassResult:
        return self.run(graph_module)

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        # Create the memory planning pass. We allocate memory for input
        # (output) tensors if alloc_graph_input (alloc_graph_output) is
        # True.
        mem_planning = MemoryPlanningPass(
            self.algo,
            # Always allow lifetime and storage overlap.
            # At opt level 0, we need overlap for idma wait.
            allow_lifetime_and_storage_overlap=True,
            alloc_graph_input=self.alloc_graph_input,
            alloc_graph_output=self.alloc_graph_output,
        )
        mem_planning.run(graph_module, graph_signature)

        graph_module = PassManager(passes=[SimplifyIdmaOpsPass()])(
            graph_module
        ).graph_module

        return PassResult(graph_module, True)
