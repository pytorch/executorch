# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import dataclasses
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class Symbol:
    name: Union[str, None]
    addr: Union[str, None] = None
    size: Union[str, None] = None
    filename: Union[str, None] = None
    size_before_relaxing: Union[str, None] = None
    details: List = dataclasses.field(default_factory=list)
    fill = int = 0


@dataclass
class Section:
    name: Union[str, None]
    addr: Union[str, None] = None
    size: Union[str, None] = None
    symbols: List = dataclasses.field(default_factory=list)


def tokenize(line: str):
    line = re.sub(" +", " ", line)
    return line.split()


def find_first_startswith(lines, pattern):
    for i, line in enumerate(lines):
        if line.startswith(pattern):
            return i
    return -1


def sort_dict_by_value(x: Dict):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}


def _build_section_and_filename_tables(sections: List[Section]):
    # build table for section names and filenames
    section_names = {}
    filenames = {}

    for section in sections:
        section_names[section.name] = len(section_names)
        for symbol in section.symbols:
            if symbol.filename not in filenames:
                filenames[symbol.filename] = len(filenames)
    return section_names, filenames


def _build_symbol_table(sections: List[Section]):
    symbols = []
    file_section_sizes_dict = defaultdict(defaultdict)
    for section in sections:
        for symbol in section.symbols:
            file_dict = file_section_sizes_dict[symbol.filename]
            file_dict[section.name] = file_dict.get(section.name, 0) + int(
                symbol.size, 16
            )

    symbols = sorted(symbols, key=lambda sym: sym[0])
    return file_section_sizes_dict


def gcc_parse_linker_map(filename):
    sections = []
    with open(filename, "r") as f:
        lines = [line.rstrip() for line in f]
    del filename

    sizes_by_filename = collections.defaultdict(int)

    # mem_config_line_id = [i for i, line in enumerate(lines) if line == "Memory Configuration"]

    lines = lines[find_first_startswith(lines, ".") :]

    for line in lines:
        # parse section
        if line.startswith("."):
            tokens = tokenize(line)
            section = Section(*tokens)
            sections.append(section)

        # parse the first line of symbol
        elif line.startswith(" ."):
            tokens = tokenize(line)
            symbol = Symbol(*tokens)
            sections[-1].symbols.append(symbol)

        # parse the second line of symbol
        elif re.match(r"\s+0x[0-9a-f]+\s+0x[0-9a-f]+", line):
            if len(sections[-1].symbols) == 0:
                continue

            last_symbol = sections[-1].symbols[-1]
            tokens = tokenize(line)

            if "*fill*" in line:
                last_symbol.fill = tokens[-1]
                continue

            if last_symbol.addr is None and last_symbol.size is None:
                addr, size, filename = tokens
                last_symbol.addr = addr
                last_symbol.size = size
                last_symbol.filename = filename

                sizes_by_filename[filename] += int(size, 16)

        # parse details of the symbol
        elif line.lstrip().startswith("0x"):
            if len(sections[-1].symbols) == 0:
                continue
            last_symbol = sections[-1].symbols[-1]
            addrs = re.findall("0x[0-9a-f]+", line)
            if len(addrs) != 1:
                continue
            addr = addrs[0]
            info = line.replace(addr, "").strip()
            last_symbol.details.append([info, addr])

        # special handling
        elif "(size before relaxing)" in line:
            last_symbol = sections[-1].symbols[-1]
            last_symbol.size_before_relaxing = re.findall("0x[0-9a-f]+", line)[0]

    # build table for section names and filenames
    section_names, filenames = _build_section_and_filename_tables(sections)

    # build table for symbols
    symbols = _build_symbol_table(sections)

    return symbols


def post_process(sizes_by_filename):
    sizes_by_filename = sort_dict_by_value(sizes_by_filename)
    table = []
    for filename, size in sizes_by_filename.items():
        matches = re.findall(r"\((.*)\)", filename)
        if len(matches) == 0:
            obj_file = os.path.basename(filename)
        else:
            obj_file = matches[0]
        table.append([size, obj_file, filename])
    return table
