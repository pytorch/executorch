# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import typing
from dataclasses import dataclass, field
from typing import List

import executorch.exir.memory as memory
import torch
from executorch.exir import ExecutorchProgramManager
from executorch.exir.memory_planning import get_node_tensor_specs
from executorch.exir.tensor import num_bytes_from_shape_and_dtype
from torch.export import ExportedProgram


@dataclass
class Allocation:
    name: str
    op_name: str
    memory_id: int
    memory_offset: int
    size_bytes: int
    fqn: str
    file_and_line_num: str


@dataclass
class MemoryTimeline:
    allocations: List[Allocation] = field(default_factory=list)


def _get_module_hierarchy(node: torch.fx.Node) -> str:
    """
    Get the module hierarchy of the given node.
    """
    module_stack = node.meta.get("nn_module_stack")
    if module_stack is not None:
        module_values_list = list(module_stack.values())
        return module_values_list[-1][0]
    return ""


def create_tensor_allocation_info(graph: torch.fx.Graph) -> List[MemoryTimeline]:
    """
    Creates a memory timlines, where each step in the timeline is a list of active
    allocations at that timestep.
    """
    nodes = graph.nodes
    memory_timeline = [None] * len(nodes)
    for _, node in enumerate(nodes):
        if node.op == "output":
            continue
        if node.target == memory.alloc:
            continue
        tensor_specs = get_node_tensor_specs(node)
        if tensor_specs is None:
            continue
        for tensor_spec in tensor_specs:
            # TODO: Make use of mem_id in the allocation info
            if tensor_spec is None or tensor_spec.mem_id is None or tensor_spec.const:
                continue
            start, end = tensor_spec.lifetime
            size = num_bytes_from_shape_and_dtype(
                typing.cast(torch.Size, tensor_spec.shape), tensor_spec.dtype
            )
            stack_trace = node.meta.get("stack_trace")
            fqn = _get_module_hierarchy(node)
            for j in range(start, end + 1):
                if memory_timeline[j] is None:
                    # pyre-ignore
                    memory_timeline[j] = MemoryTimeline()
                # pyre-ignore
                memory_timeline[j].allocations.append(
                    Allocation(
                        node.name,
                        node.target,
                        tensor_spec.mem_id,
                        tensor_spec.mem_offset,
                        size,
                        fqn,
                        stack_trace,
                    )
                )
    # pyre-ignore
    return memory_timeline


def _validate_memory_planning_is_done(exported_program: ExportedProgram):
    """
    Validate whether the memory planning has been done on the given program.
    """
    for node in exported_program.graph.nodes:
        # If there is at least one memory allocation node, then we know the memory planning has been done.
        if node.target == memory.alloc:
            return True
    return False


def generate_memory_trace(
    executorch_program_manager: ExecutorchProgramManager,
    chrome_trace_filename: str,
    enable_memory_offsets: bool = False,
    method_name: str = "forward",
):
    """
    Generate the memory timeline from the given ExecuTorch program.
    Args:
        executorch_program The ExecuTorch program to be analyzed.
    Returns:
        Chrome trace in JSON format:
        Format:
        Each thread represents a unit of time. Thus to navigate timeline scroll up and down.
        For each thread, the x axis represents live tensor objects that are normalized according the allocation size.
    """
    if not isinstance(executorch_program_manager, ExecutorchProgramManager):
        raise ValueError(
            f"generate_memory_trace expects ExecutorchProgramManager instance but got {type(executorch_program_manager)}"
        )

    exported_program = executorch_program_manager.exported_program(method_name)
    if not _validate_memory_planning_is_done(exported_program):
        raise ValueError("Executorch program does not have memory planning.")

    memory_timeline = create_tensor_allocation_info(exported_program.graph)
    root = {}
    trace_events = []
    root["traceEvents"] = trace_events

    tid = 0
    for memory_timeline_event in memory_timeline:
        start_time = 0
        if memory_timeline_event is None:
            continue
        for allocation in memory_timeline_event.allocations:
            e = {}
            e["name"] = allocation.name
            e["cat"] = "memory_allocation"
            e["ph"] = "X"
            e["ts"] = (
                int(allocation.memory_offset)
                if enable_memory_offsets
                else int(start_time)
            )
            allocation_size_kb = allocation.size_bytes
            e["dur"] = int(allocation_size_kb)
            e["pid"] = int(allocation.memory_id)
            e["tid"] = tid
            e["args"] = {}
            e["args"]["op_name"] = f"{allocation.op_name}"
            # ID refers to memory space, typically from 1 to N.
            # For CPU, everything is allocated on one "space", other backends may have multiple.
            e["args"]["Memory ID"] = allocation.memory_id
            e["args"]["fqn"] = f"{allocation.fqn}"
            e["args"]["source"] = f"{allocation.file_and_line_num}"
            e["args"]["bytes"] = allocation.size_bytes
            start_time += allocation_size_kb
            trace_events.append(e)
        tid += 1

    json_content: str = json.dumps(root, indent=2)

    with open(chrome_trace_filename, "wb") as json_file:
        json_file.write(json_content.encode("ascii"))
