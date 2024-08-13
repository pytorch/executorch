# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import struct
from collections import OrderedDict
from enum import Enum

from typing import Dict, List, Tuple

from prettytable import PrettyTable

# This version number should match the one defined in profiler.h
ET_PROF_VER = 0x00000001

# This string defines the layout of the prof_result_t struct
# defined in executorch/profiler/profiler.h. This is used to
# unpack the binary data to derive the profiling results.
# To align the end of a structure to the alignment requirement
# of a particular type, end the format with the code for that
# type with a repeat count of zero. Adding 0Q at the ending ensures
# that the struct is aligned by 8 bytes which is the alignment we
# impose in the runtime.
PROF_HEADER_STRUCT_FMT = "32s7I0Q"
PROF_RESULT_STRUCT_FMT = "32siIQQ0Q"
ALLOCATOR_STRUCT_FMT = "32sQ0Q"
ALLOCATION_STRUCT_FMT = "2I0Q"
CHAIN_IDX_NO_CHAIN = -1


class TimeScale(Enum):
    TIME_IN_NS = 0
    TIME_IN_US = 1
    TIME_IN_MS = 2
    CPU_CYCLES = 3


# These data classes represent the structures used on device to
# log various forms of profiling data.
@dataclasses.dataclass
class ProfileData:
    name: str
    chain_idx: int
    instruction_idx: int
    start_time: int
    end_time: int


@dataclasses.dataclass
class ProfilerHeader:
    name: str
    prof_ver: int
    max_prof_entries: int
    prof_entries: int
    max_allocator_entries: int
    allocator_entries: int
    max_mem_prof_entries: int
    mem_prof_entries: int


@dataclasses.dataclass
class Allocator:
    name: str
    allocator_id: int


@dataclasses.dataclass
class MemAllocation:
    allocator_id: int
    allocation_size: int


"""
These data classes are derived from the post-processing of the
profiling data retrieved from the runtime. If there are multiple
blocks of profiling data resulting from iterations of the same code
segment then corresponding entries will be consolidated with each
entry in the consolidated list representing one iteration.
"""


@dataclasses.dataclass
class ProfileEvent:
    name: str
    ts: List[float]
    duration: List[float]
    chain_idx: int = -1
    instruction_idx: int = -1
    # pyre-ignore[8]: Incompatible attribute type
    stacktrace: str = None


@dataclasses.dataclass
class ProfileEventFrameworkTax:
    exec_time: List[int]
    kernel_and_delegate_time: List[int]
    framework_tax: List[float]


@dataclasses.dataclass
class MemEvent:
    allocator_name: str
    total_allocations_done: int


def adjust_time_scale(event: ProfileData, time_scale: TimeScale):
    time_div_factor = {
        TimeScale.CPU_CYCLES: 1,
        TimeScale.TIME_IN_MS: 1,
        TimeScale.TIME_IN_US: 1000,
        TimeScale.TIME_IN_NS: 1000000,
    }
    div_factor = time_div_factor[time_scale]
    if div_factor != 1:
        duration = round((event.end_time - event.start_time) / div_factor, 4)
        start_time = round((event.start_time) / div_factor, 4)
    else:
        duration = event.end_time - event.start_time
        start_time = event.start_time
    return start_time, duration


def parse_prof_blocks(
    prof_blocks: Dict[str, List[Tuple[List[ProfileData], List[MemAllocation]]]],
    allocator_dict: Dict[int, str],
    time_scale: TimeScale,
) -> Tuple[Dict[str, List[ProfileEvent]], Dict[str, List[MemEvent]]]:

    prof_data = OrderedDict()
    mem_prof_data = OrderedDict()

    # Iterate through all the profiling blocks data that have been grouped by name.
    for name, data_list in prof_blocks.items():
        prof_data_list = []
        mem_prof_data_list = []
        # Each entry in data_list is a tuple in which the first entry is profiling data
        # and the second entry is memory allocation data, also each entry in data_list
        # represents one iteration of a code block.
        for i in range(len(data_list)):
            for idx, event in enumerate(data_list[i][0]):
                # If the event represented by the index idx already exists in the list
                # then just append the new time entry to the duration list present in
                # the event object. If it doesn't exist then create a new entry and add
                # it to the list.
                if idx < len(prof_data_list):
                    start_time, duration = adjust_time_scale(event, time_scale)
                    prof_data_list[idx].ts.append(start_time)
                    prof_data_list[idx].duration.append(duration)
                else:
                    start_time, duration = adjust_time_scale(event, time_scale)
                    prof_data_list.append(
                        ProfileEvent(
                            event.name,
                            [start_time],
                            [duration],
                            event.chain_idx,
                            event.instruction_idx,
                        )
                    )

            # Collect all the memory allocation events of this iteration of the code block
            for idx, event in enumerate(data_list[i][1]):
                if idx >= len(mem_prof_data_list):
                    mem_prof_data_list.append(event)

        # Group all the memory allocation events based on the allocator they were
        # allocated from.
        alloc_sum_dict = OrderedDict()
        for alloc in mem_prof_data_list:
            alloc_sum_dict[alloc.allocator_id] = (
                alloc_sum_dict.get(alloc.allocator_id, 0) + alloc.allocation_size
            )

        mem_prof_sum_list = []
        for allocator_id, allocation_size in alloc_sum_dict.items():
            mem_prof_sum_list.append(
                MemEvent(allocator_dict[allocator_id], allocation_size)
            )
        prof_data[name] = prof_data_list
        mem_prof_data[name] = mem_prof_sum_list

    return prof_data, mem_prof_data


def sanity_check_prof_outputs(
    prof_blocks: Dict[str, List[Tuple[List[ProfileData], List[MemAllocation]]]]
):
    for _, prof_block_vals in prof_blocks.items():
        for i in range(len(prof_block_vals) - 1):
            prof_data_list_base = prof_block_vals[i][0]
            prof_data_list_cmp = prof_block_vals[i + 1][0]

            # Profiling blocks corresponding to the same name should always be of the same
            # size as they essentially just represent one iteration of a code block that has been
            # run multiple times.
            if len(prof_data_list_base) != len(prof_data_list_cmp):
                raise ValueError(
                    "Profiling blocks corresponding to the same name shouldn't be of different lengths."
                )

            for idx in range(len(prof_data_list_base)):
                if prof_data_list_base[idx].name != prof_data_list_cmp[idx].name:
                    raise ValueError(
                        "Corresponding entries in different iterations of the "
                        "profiling block do not match"
                    )

            mem_prof_data_list_base = prof_block_vals[i][1]
            mem_prof_data_list_cmp = prof_block_vals[i + 1][1]

            if len(mem_prof_data_list_base) != len(mem_prof_data_list_cmp):
                raise ValueError(
                    "Memory profiling blocks corresponding to the same name shouldn't be of different lengths."
                )

            for idx in range(len(mem_prof_data_list_base)):
                if (
                    mem_prof_data_list_base[idx].allocator_id
                    != mem_prof_data_list_cmp[idx].allocator_id
                ):
                    raise ValueError(
                        "Corresponding entries in different iterations of the memory "
                        "profiling blocks do not have the same allocator id"
                    )
                if (
                    mem_prof_data_list_base[idx].allocation_size
                    != mem_prof_data_list_cmp[idx].allocation_size
                ):
                    raise ValueError(
                        "Corresponding entries in different iterations of the memory "
                        "profiling blocks do not have the same allocation size."
                    )


def deserialize_profile_results(
    buff: bytes, time_scale: TimeScale = TimeScale.TIME_IN_NS
) -> Tuple[Dict[str, List[ProfileEvent]], Dict[str, List[MemEvent]]]:

    prof_header_struct_size = struct.calcsize(PROF_HEADER_STRUCT_FMT)
    prof_allocator_struct_size = struct.calcsize(ALLOCATOR_STRUCT_FMT)
    prof_allocation_struct_size = struct.calcsize(ALLOCATION_STRUCT_FMT)
    prof_result_struct_size = struct.calcsize(PROF_RESULT_STRUCT_FMT)
    prof_blocks = OrderedDict()
    allocator_dict = {}
    base_offset = 0

    while base_offset < len(buff):
        # Unpack the header for this profiling block from which we can figure
        # out how many profiling entries are present in this block.
        prof_header_args = list(
            struct.unpack_from(PROF_HEADER_STRUCT_FMT, buff, offset=base_offset)
        )
        # decode name in profiler header
        prof_header_args[0] = prof_header_args[0].decode("utf-8").replace("\u0000", "")
        prof_header = ProfilerHeader(*prof_header_args)
        base_offset += prof_header_struct_size

        assert prof_header.prof_ver == ET_PROF_VER, (
            "Mismatch in version between profile dump" "and post-processing tool"
        )
        # Get all the profiling (perf events) entries
        prof_data = []
        for i in range(prof_header.prof_entries):
            name_bytes, type, id, start_time, end_time = struct.unpack_from(
                PROF_RESULT_STRUCT_FMT,
                buff,
                offset=base_offset + i * prof_result_struct_size,
            )
            prof_data.append(
                ProfileData(
                    # name_bytes is 32 bytes string, where if the real log event is less
                    # than 32 characters it'll be filled with 0 chars => trimming it
                    name_bytes.decode("utf-8").replace("\u0000", ""),
                    type,
                    id,
                    start_time,
                    end_time,
                )
            )

        # Move forward in the profiling block to start parsing memory allocation events.
        base_offset += prof_result_struct_size * prof_header.max_prof_entries

        # Parse the allocator entries table, this table maps the allocator id to the
        # string containing the name designated to this allocator.
        for i in range(0, prof_header.allocator_entries):
            allocator_name, allocator_id = struct.unpack_from(
                ALLOCATOR_STRUCT_FMT,
                buff,
                offset=base_offset + i * prof_allocator_struct_size,
            )
            allocator_dict[allocator_id] = allocator_name.decode("utf-8").replace(
                "\u0000", ""
            )

        base_offset += prof_allocator_struct_size * prof_header.max_allocator_entries

        # Get all the profiling (memory allocation events) entries
        mem_prof_data = []
        for i in range(0, prof_header.mem_prof_entries):
            mem_prof_entry = list(
                struct.unpack_from(
                    ALLOCATION_STRUCT_FMT,
                    buff,
                    offset=base_offset + i * prof_allocation_struct_size,
                )
            )
            mem_prof_data.append(MemAllocation(*mem_prof_entry))

        base_offset += prof_allocation_struct_size * prof_header.max_mem_prof_entries

        # Get the name of this profiling block and append the profiling data and memory
        # allocation data we just parsed to the list that maps to this block name.
        prof_blocks[prof_header.name] = prof_blocks.get(prof_header.name, []) + [
            (prof_data, mem_prof_data)
        ]

    sanity_check_prof_outputs(prof_blocks)
    return parse_prof_blocks(prof_blocks, allocator_dict, time_scale)


def profile_table(
    profile_data: Dict[str, List[ProfileEvent]], model_buffer=None
) -> List[PrettyTable]:

    results = []
    max_len = 0

    for name, prof_entries_list in profile_data.items():
        table = PrettyTable()
        table.title = name
        table.add_rows(
            [
                (
                    entry.name,
                    entry.chain_idx,
                    entry.instruction_idx,
                    None,
                )
                + tuple(val for val in entry.duration)
                for entry in prof_entries_list
            ]
        )
        max_len = max(max_len, len(prof_entries_list[0].duration))
        table.field_names = [
            "Name",
            "Chain",
            "Instr",
            "Frame",
        ] + ["Iteration " + str(i) for i in range(max_len)]
        results.append(table)
    return results


def mem_profile_table(mem_allocations: Dict[str, List[MemEvent]]) -> List[PrettyTable]:
    tables = []
    for name, prof_data_list in mem_allocations.items():
        table = PrettyTable()
        table.title = name
        table_rows = []
        for mem_event in prof_data_list:
            table_rows += [(mem_event.allocator_name, mem_event.total_allocations_done)]
        table.add_rows(table_rows)
        table.field_names = ["Allocator name"] + ["Total size of allocations done"]
        tables.append(table)
    return tables


def profile_aggregate_framework_tax(
    prof_data: Dict[str, List[ProfileEvent]]
) -> Dict[str, ProfileEventFrameworkTax]:
    prof_framework_tax = OrderedDict()

    for name, prof_data_list in prof_data.items():
        execute_max = []
        kernel_and_delegate_sum = []

        for d in prof_data_list:
            if "Method::execute" in d.name:
                execute_max = max(execute_max, d.duration)

            if "native_call" in d.name or "delegate_execute" in d.name:
                for idx in range(len(d.duration)):
                    if idx < len(kernel_and_delegate_sum):
                        kernel_and_delegate_sum[idx] += d.duration[idx]
                    else:
                        kernel_and_delegate_sum.append(d.duration[idx])

        if len(execute_max) == 0 or len(kernel_and_delegate_sum) == 0:
            continue

        framework_tax_list = [
            round((execute_time - kernel_delegate_call) / execute_time, 4) * 100
            for execute_time, kernel_delegate_call in zip(
                execute_max, kernel_and_delegate_sum
            )
        ]

        prof_framework_tax[name] = ProfileEventFrameworkTax(
            execute_max, kernel_and_delegate_sum, framework_tax_list
        )

    return prof_framework_tax


def profile_framework_tax_table(
    prof_framework_tax_data: Dict[str, ProfileEventFrameworkTax]
):
    tables = []
    for name, prof_data_list in prof_framework_tax_data.items():
        tables = []
        table_agg = PrettyTable()
        table_agg.title = name + " framework tax calculations"

        table_agg.add_rows(
            [
                ("Model execution time", *prof_data_list.exec_time),
                (
                    "Time spent in kernels and delegates",
                    *prof_data_list.kernel_and_delegate_time,
                ),
                ("Framework tax (%)", *prof_data_list.framework_tax),
            ]
        )
        table_agg.field_names = [""] + [
            "Iteration " + str(i) for i in range(len(prof_data_list.exec_time))
        ]
        tables.append(table_agg)
    return tables


def deserialize_profile_results_files(
    profile_results_path: str,
    bundled_program_ff_path: str,
    time_scale: TimeScale = TimeScale.TIME_IN_NS,
):
    with open(profile_results_path, "rb") as prof_res_file, open(
        bundled_program_ff_path, "rb"
    ) as model_ff_file:
        prof_res_buf = prof_res_file.read()
        bundled_program_ff_buf = model_ff_file.read()

    prof_data, mem_allocations = deserialize_profile_results(prof_res_buf, time_scale)
    framework_tax_data = profile_aggregate_framework_tax(prof_data)

    prof_tables = profile_table(prof_data, bundled_program_ff_buf)
    for table in prof_tables:
        print(table)

    prof_tables_agg = profile_framework_tax_table(framework_tax_data)
    for table in prof_tables_agg:
        print(table)

    mem_prof_table = mem_profile_table(mem_allocations)
    for table in mem_prof_table:
        print(table)

    return prof_data, mem_allocations
