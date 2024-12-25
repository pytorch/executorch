# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import collections
import itertools
import logging
import typing
from functools import partial
from typing import Iterable, List, Optional, Tuple

import torch
from executorch.backends.cadence.aot.memory_constraints import (
    GenerateMemConstraints,
    MemConstraints,
)
from executorch.backends.cadence.aot.utils import MemoryConfig

from executorch.exir import ExecutorchProgramManager
from executorch.exir.memory_planning import collect_specs_from_nodes, Verifier
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.tensor import TensorSpec
from tabulate import tabulate
from torch.export.exported_program import ExportGraphSignature
from torch.fx.passes.infra.pass_base import PassResult


# get num memories indexed from 1..N, compatible with EXIR's spec.mem_id
def get_num_memories(memory_config: MemoryConfig) -> int:
    return len(memory_config.memory_sizes) + 1


# memory_space module provides num_memories indexed 0..num_memories-1.
def get_size(memory_config: MemoryConfig, exir_id: int) -> int:
    return memory_config.memory_sizes[exir_id - 1]


def collect_specs_from_graph_module(
    graph_module: torch.fx.GraphModule,
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
        ignore_graph_input=not alloc_graph_input,
        ignore_graph_output=not alloc_graph_output,
    )


# baseline tensor placement algorithm, that greedily tries to place the tensor in
# the fastest memory available
# flake8: noqa 'position_based_greedy_with_hierarchy' is too complex (13)
def position_based_greedy_with_hierarchy(
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: ExportGraphSignature,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
    *,
    memory_config: MemoryConfig,
    mem_constraints: MemConstraints,
    additional_constraint_gen_passes: Optional[
        List[
            typing.Callable[
                [MemConstraints],
                typing.Callable[[torch.fx.GraphModule], Optional[PassResult]],
            ]
        ]
    ] = None,
) -> List[int]:
    num_memories = get_num_memories(memory_config)
    bufsizes = [0] * num_memories
    allocated_buffers: List[List[TensorSpec]] = [[] for _ in range(num_memories)]

    # Generate the memory constraints
    GenerateMemConstraints(mem_constraints, additional_constraint_gen_passes)(
        graph_module
    )

    def overlap(spec: TensorSpec) -> Optional[TensorSpec]:
        for allocated_spec in allocated_buffers[spec.mem_id]:
            if Verifier.lifetime_overlap(
                spec, allocated_spec
            ) and Verifier.storage_overlap(spec, allocated_spec):
                return allocated_spec
        return None

    def memory_available(spec: TensorSpec) -> bool:
        return spec.mem_offset + spec.allocated_memory <= get_size(
            memory_config, spec.mem_id
        )

    # Iterate over all the specs in sorted order
    for spec in sorted(
        collect_specs_from_graph_module(
            graph_module, alloc_graph_input, alloc_graph_output
        ),
        key=lambda spec: spec.allocated_memory,
        reverse=True,
    ):
        # Skip allocation memory to any tensor whose spec id is in skip list.
        if mem_constraints.skipped_spec(spec):
            continue

        for spec.mem_id in range(1, num_memories):
            if mem_constraints.is_mem_id_in_blocklist(spec, spec.mem_id):
                continue
            spec.mem_offset = 0
            while memory_available(spec) and (overlapped := overlap(spec)):
                spec.mem_offset = overlapped.mem_offset + overlapped.allocated_memory
            if memory_available(spec):
                allocated_buffers[spec.mem_id].append(spec)
                bufsizes[spec.mem_id] = max(
                    spec.mem_offset + spec.allocated_memory, bufsizes[spec.mem_id]
                )
                break
        if (
            not allocated_buffers[spec.mem_id]
            or allocated_buffers[spec.mem_id][-1] is not spec
        ):
            raise MemoryError(f"Cannot fit {spec} in any memory hierarchy")

        # And now honor the various memory location constraints (i.e., infer the memory
        # location of tensors in skip_specs from the constraints) for this spec.
        if mem_constraints.relative_loc_constraints_exist():
            mem_constraints.resolve_relative_loc_constraints(spec)

    # At the end, all the keys in relative_loc_constraints should have been visited
    # and emptied.
    assert not mem_constraints.relative_loc_constraints_exist()

    logging.debug(
        f"position based greedy algorithm with hierarchy returns bufsizes: {bufsizes}"
    )
    return bufsizes


# Greedy tensor placement with the heuristics from arxiv.org/pdf/2001.03288.pdf
def greedy_by_size_for_offset_calculation_with_hierarchy(
    graph_module: torch.fx.GraphModule,
    alignment: int,
    graph_signature: ExportGraphSignature,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
    *,
    memory_config: MemoryConfig,
    mem_constraints: MemConstraints,
    additional_constraint_gen_passes: Optional[
        List[
            typing.Callable[
                [MemConstraints],
                typing.Callable[[torch.fx.GraphModule], Optional[PassResult]],
            ]
        ]
    ] = None,
) -> List[int]:
    num_memories = get_num_memories(memory_config)
    bufsizes = [0] * num_memories
    allocated_buffers = [[] for _ in range(num_memories)]

    # Generate the memory constraints
    GenerateMemConstraints(mem_constraints, additional_constraint_gen_passes)(
        graph_module
    )

    # Iterate over all the specs in sorted order
    for spec in sorted(
        collect_specs_from_graph_module(
            graph_module, alloc_graph_input, alloc_graph_output
        ),
        key=lambda spec: spec.allocated_memory,
        reverse=True,
    ):
        # Skip allocation memory to any tensor whose spec id is in skip list.
        if mem_constraints.skipped_spec(spec):
            continue

        for spec.mem_id in range(1, num_memories):
            if mem_constraints.is_mem_id_in_blocklist(spec, spec.mem_id):
                continue
            prev_offset, smallest_gap = 0, float("inf")
            for allocated_spec in allocated_buffers[spec.mem_id]:
                if Verifier.lifetime_overlap(spec, allocated_spec):
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
                        allocated_spec.mem_offset + allocated_spec.allocated_memory,
                        prev_offset,
                    )
            if spec.mem_offset is None:
                if prev_offset + spec.allocated_memory > get_size(
                    memory_config, spec.mem_id
                ):
                    continue
                else:
                    spec.mem_offset = prev_offset
            bufsizes[spec.mem_id] = max(
                spec.mem_offset + spec.allocated_memory, bufsizes[spec.mem_id]
            )
            allocated_buffers[spec.mem_id].append(spec)
            allocated_buffers[spec.mem_id].sort(key=lambda spec: spec.mem_offset)
            # A data structure used for maintaining the tensor order
            # by offset, named ordered_allocated_ids in the paper
            break
        if spec not in allocated_buffers[spec.mem_id]:
            raise MemoryError(f"Cannot fit {spec} in any memory hierarchy")

        # And now honor the various memory location constraints (i.e., infer the memory
        # location of tensors in skip_specs from the constraints) for this spec.
        if mem_constraints.relative_loc_constraints_exist():
            mem_constraints.resolve_relative_loc_constraints(spec)

    # At the end, all the keys in relative_loc_constraints should have been visited
    # and emptied.
    assert not mem_constraints.relative_loc_constraints_exist()

    logging.debug(
        f"greedy by size for offset calculation with hierarchy returns bufsizes: {bufsizes}"
    )
    return bufsizes


def find_peak_memory_usages_per_memory(
    graph_module: torch.fx.GraphModule,
    alloc_graph_input: bool,
    alloc_graph_output: bool,
    mem_constraints: Optional[MemConstraints] = None,
) -> List[int]:
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
        graph_module, alloc_graph_input, alloc_graph_output
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
    alloc_graph_input: bool,
    alloc_graph_output: bool,
    mem_constraints: Optional[MemConstraints] = None,
) -> Tuple[int, int]:
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
        graph_module, alloc_graph_input, alloc_graph_output
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
        tabulate(
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
        tabulate(
            total_memory_usage_table,
            tablefmt="outline",
        )
    )


class CadenceMemoryPlanning:
    def __init__(
        self,
        memory_config: MemoryConfig,
        opt_level: int,
        mem_algo: int,
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
        additional_constraint_gen_passes: Optional[
            List[
                typing.Callable[
                    [MemConstraints],
                    typing.Callable[[torch.fx.GraphModule], Optional[PassResult]],
                ]
            ]
        ] = None,
    ) -> None:
        self._init_mem_algos()

        self.memory_config = memory_config
        self.opt_level = opt_level
        self.mem_algo = mem_algo
        self.alloc_graph_input = alloc_graph_input
        self.alloc_graph_output = alloc_graph_output
        self.additional_constraint_gen_passes = additional_constraint_gen_passes

    def _init_mem_algos(self) -> None:
        self.available_mem_algos = [
            position_based_greedy_with_hierarchy,
            greedy_by_size_for_offset_calculation_with_hierarchy,
        ]

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        mem_constraints = MemConstraints(
            opt_level=self.opt_level,
            alloc_graph_input=self.alloc_graph_input,
            alloc_graph_output=self.alloc_graph_output,
        )
        algo = partial(
            self.available_mem_algos[self.mem_algo],
            memory_config=self.memory_config,
            mem_constraints=mem_constraints,
            additional_constraint_gen_passes=self.additional_constraint_gen_passes,
        )
        # Create the memory planning pass. We allocate memory for input
        # (output) tensors if alloc_graph_input (alloc_graph_output) is
        # True.
        mem_planning = MemoryPlanningPass(
            algo,
            allow_lifetime_and_storage_overlap=(self.opt_level >= 2),
            alloc_graph_input=self.alloc_graph_input,
            alloc_graph_output=self.alloc_graph_output,
        )
        mem_planning(graph_module)

        return PassResult(graph_module, True)
