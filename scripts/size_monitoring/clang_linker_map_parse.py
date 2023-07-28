#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parse an LLD linker map and log the ELF file breakdown by section & input files.
"""
import re
from collections import defaultdict
from typing import Dict, Tuple

REDUNDANT_PREFIXES = [
    "/home/engshare/",
    "buck-out/(?:(?:dev|opt|dbgo|opt-clang-thinlto)/)?gen/(?:[0-9a-f]{8}/)?",
    "buck-out/v2/gen/[^/]+/(?:[0-9a-f]{16}/)?",
]


def drop_redundant_prefixes(path: str) -> Tuple[str, str]:
    for prefix in REDUNDANT_PREFIXES:
        m = re.match(f"({prefix})(.+)", path)
        if m:
            prefix, suffix = m.groups()
            return prefix, suffix
    return "", path


def clang_parse_linker_map(path: str) -> Dict:
    """Extract [(section, source, size)] from linker map"""

    # Here's a sample of an 64-bit LLD linker map profile
    #         VMA              LMA     Size Align Out     In      Symbol
    #      20b780           20b780    31008    64 .rodata
    #      20b780           20b780     8bf3     1         <internal>:(.rodata)
    #      214378           214378       c0     8         buck-out/v2/gen/<...>.build_info.o:(.rodata)
    #      214378           214378        8     1                 BuildInfo_kBuildMode
    #      214380           214380        8     1                 BuildInfo_kBuildTool
    #
    # Code to follow: https://github.com/llvm-mirror/lld/blob/master/ELF/MapFile.cpp
    #
    # The 64-bit LLD linker map file starts with a header like this:
    #   os << right_justify("VMA", w) << ' ' << right_justify("LMA", w)
    #      << "     Size Align Out     In      Symbol\n";
    #
    # Each line starts with 16+1+16+1+8+1+5+1=49 B prefix:
    #   os << format("%16llx %16llx %8llx %5lld ", vma, lma, size, align);
    #
    # Lines that describe a section are followed by the section name
    #   os << osec->name << '\n';
    # Lines that describe an input file's contribution to a section are indented by 8
    #   os << indent8 << toString(isec) << '\n';
    # Lines that describe each symbol from that input section are indented by 16
    #   os << indent16 << toString(*syms[i]);
    #
    # Out / In columns are only emitted when a new output section / input file starts.
    # For the above example will produce [(".rodata", "buck-out/v2/gen/<...>.build_info.o", 0xc0)]
    file_section_sizes_dict = defaultdict(defaultdict)
    with open(path) as f:
        next(f)  # skip header
        start_size = 16 + 1 + 16 + 1  # "%16llx %16llx "
        end_size = start_size + 8  # "%8llx"
        start_section = 16 + 1 + 16 + 1 + 8 + 1 + 5 + 1  # "%16llx %16llx %8llx %5lld "
        start_file = start_section + 8  # indent8
        curr_section_name = ""
        curr_file_name = ""
        curr_size = 0
        for line in f:
            if line[start_section] != " ":
                # Starting a new output section.
                new_section_name = line[start_section:].strip()
                if curr_section_name and new_section_name == curr_section_name:
                    raise Exception(f"Repeating section name: {new_section_name}")

                if curr_section_name and curr_file_name and curr_size:
                    file_dict = file_section_sizes_dict[curr_file_name]
                    file_dict[curr_section_name] = (
                        file_dict.get(curr_section_name, 0) + curr_size
                    )
                    curr_section_name = ""
                    curr_file_name = ""
                    curr_size = 0

                curr_section_name = new_section_name
                curr_file_name = ""
                curr_size = 0
                continue
            if curr_section_name and line[start_file] != " ":
                # Starting a new source file's input section that's part of this output section.
                # e.g.: path-to-file.o:(.text._ZN5folly7dynamicaSERKS0_)
                new_name = drop_redundant_prefixes(
                    line[start_file:].strip().split(":(")[0]
                )[1]
                new_size = int(line[start_size:end_size].strip(), base=16)
                if new_name == curr_file_name:
                    curr_size += new_size
                else:
                    if curr_section_name and curr_file_name and curr_size:
                        temp_dict = file_section_sizes_dict[curr_file_name]
                        temp_dict[curr_section_name] = (
                            temp_dict.get(curr_section_name, 0) + curr_size
                        )
                    curr_file_name = new_name
                    curr_size = new_size
                continue
            # Ignore lines that describe each symbol's contribution.
            # Only collect breakdown data at output-section x input-file granularity.

    if curr_section_name and curr_file_name and curr_size:
        temp_dict = file_section_sizes_dict[curr_file_name]
        temp_dict[curr_section_name] = temp_dict.get(curr_section_name, 0) + curr_size

    return file_section_sizes_dict
