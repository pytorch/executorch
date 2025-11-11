#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
A script to help check binary dependencies and disallowed symbols in intermediate build files.
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, NoReturn, Optional, Tuple

# Script output statuses.
STATUS_OK = 0
STATUS_SCRIPT_ERROR = 1
STATUS_ERROR = 2
STATUS_WARNING = 3

# Object file suffix.
OBJECT_SUFFIX = ".o"

# Project root, assuming this script is in `<root>/scripts/`
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Regex to strip info from nm and readelf.
NM_REGEX = re.compile(r"\d*\s+(?P<status>\S)\s+(?P<symbol>.*)")
READELF_DEP_REGEX = re.compile(r".*\(NEEDED\)\s+(?P<so>.*)")
READELF_DYN_SYM_REGEX = re.compile(r"(UND|\d+)\s+(?P<symbol>[^@\s:]+)(@.*)?$")

# Disallow list of prefixes for standard library symbols.
DISALLOW_LIST = [
    "operator new",
    "operator delete",
    "std::__cxx11::basic_string",
    "std::__throw",
    "std::deque",
    "std::exception",
    "std::forward_list",
    "std::list",
    "std::map",
    "std::multimap",
    "std::multiset",
    "std::priority_queue",
    "std::queue",
    "std::set",
    "std::stack",
    "std::unordered_map",
    "std::unordered_multimap",
    "std::unordered_multiset",
    "std::unordered_set",
    "std::vector",
]


@dataclass
class Symbol:
    """Symbol scraped from ELF binary object."""

    mangled: str
    demangled: str
    defined: bool
    disallowed: bool
    sources: List[Path]


# Cached symbols dictionary.
symbols_cache: Optional[Dict[str, Symbol]] = None


def error(message: str) -> NoReturn:
    """Emit an error message and kill the script."""
    print(message)
    sys.exit(STATUS_SCRIPT_ERROR)


def get_tool_output(args: List[str]) -> str:
    """Execute a command in the shell and return the output."""
    result = subprocess.run(args, stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    return output


def read_nm(
    nm: str, file: Path, exclude: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """Read a set of symbols using the nm tool."""
    if exclude is None:
        exclude = ["N"]

    output = get_tool_output([nm, file])
    result = []
    for line in output.splitlines():
        match = re.search(NM_REGEX, line)
        if not match:
            continue

        status = match.group("status").upper()
        if exclude is None or status not in exclude:
            result.append((status, match.group("symbol")))
    return result


def get_object_symbols(
    nm: str, symbols: Dict[str, Symbol], object_file: Path, source_file: Path
) -> None:
    """Scrape symbols from a binary object."""
    symbol_table = read_nm(nm, object_file)
    for t, symbol in symbol_table:
        if symbol not in symbols:
            symbols[symbol] = Symbol(
                mangled=symbol,
                demangled="",
                defined=(t != "U"),
                disallowed=False,
                sources=[],
            )
        if source_file in symbols[symbol].sources:
            continue
        symbols[symbol].sources.append(source_file)


def get_elf_dependencies(readelf: str, binary_file: Path) -> List[str]:
    """Get the shared object dependencies of a binary executable."""
    shared_objects = []
    output = get_tool_output([readelf, "-d", binary_file])
    for line in output.splitlines():
        match = re.search(READELF_DEP_REGEX, line)
        if not match:
            continue
        shared_objects.append(match.group("so"))

    return shared_objects


def get_binary_dynamic_symbols(readelf: str, binary_file: Path) -> List[str]:
    """Get the dynamic symbols required by a binary executable."""
    dynamic_symbols = []
    output = get_tool_output([readelf, "--dyn-syms", "--wide", binary_file])
    for line in output.splitlines():
        match = re.search(READELF_DYN_SYM_REGEX, line)
        if not match:
            continue
        dynamic_symbols.append(match.group("symbol"))
    return list(set(dynamic_symbols))


def demangle_symbols(cxxfilt: str, mangled_symbols: Iterable[Symbol]) -> None:
    """Demangle a collection of symbols using the cxxfilt tool."""
    output = get_tool_output([cxxfilt] + [symbol.mangled for symbol in mangled_symbols])
    for symbol, demangled in zip(mangled_symbols, output.splitlines()):
        symbol.demangled = demangled


def check_disallowed_symbols(cxxfilt: str, symbols: Iterable[Symbol]) -> None:
    """Check a collection of symbols for disallowed prefixes."""
    for symbol in symbols:
        assert len(symbol.demangled) > 0
        if symbol.demangled.startswith(tuple(DISALLOW_LIST)):
            symbol.disallowed = True


def get_cached_symbols(nm: str, build_root: Path) -> Dict[str, Symbol]:
    """Return a dictionary of symbols scraped from build files"""
    global symbols_cache

    if symbols_cache is not None:
        return symbols_cache
    symbols = {}

    if not build_root.is_dir():
        error("Specified buck-out is not a directory")

    for root, _, files in os.walk(build_root):
        root_path = Path(root)
        for file_name in files:
            file_path = root_path / file_name
            if file_path.suffix == OBJECT_SUFFIX:
                object_file_path = file_path
                source_file_name = object_file_path.name[: -len(OBJECT_SUFFIX)]

                object_file_rel = Path(os.path.relpath(object_file_path, build_root))
                if "codegen" in str(object_file_path):
                    source_file_path = source_file_name + " (generated)"
                else:
                    source_file_path = (
                        PROJECT_ROOT / object_file_rel.parent.parent / source_file_name
                    )
                get_object_symbols(nm, symbols, object_file_path, source_file_path)

    symbols_cache = symbols
    return symbols_cache


def check_dependencies(readelf: str, binary_file: Path) -> int:
    """Check that there are no shared object dependencies of a binary executable."""
    elf_dependencies = get_elf_dependencies(readelf, binary_file)
    if len(elf_dependencies) > 0:
        print("Found the following shared object dependencies:")
        for dependency in elf_dependencies:
            print(" *", dependency)
        print()
        return STATUS_ERROR
    return STATUS_OK


def check_disallowed_symbols_build_dir(nm: str, cxxfilt: str, build_root: Path) -> int:
    """Check that there are no disallowed symbols used in intermediate build files."""
    symbols = get_cached_symbols(nm, build_root)
    symbol_list = list(symbols.values())
    demangle_symbols(cxxfilt, symbol_list)
    check_disallowed_symbols(cxxfilt, symbol_list)
    disallowed_symbols = filter(lambda symbol: symbol.disallowed, symbol_list)

    disallowed_by_file = {}
    for symbol in disallowed_symbols:
        for file in symbol.sources:
            if file not in disallowed_by_file:
                disallowed_by_file[file] = []
            disallowed_by_file[file].append(symbol)

    for file, symbols in disallowed_by_file.items():
        print(f"{file} contains disallowed symbols:")
        for symbol in symbols:
            print(" *", symbol.demangled)
        print()

    if len(disallowed_by_file) > 0:
        return STATUS_ERROR

    return STATUS_OK


def check_dynamic(
    nm: str, readelf: str, cxxfilt: str, binary_file: Path, build_root: Optional[Path]
) -> int:
    """Check for dynamic symbols required by an executable, categorizing them from the
    intermediate files that may have included those symbols.
    """
    symbols = get_cached_symbols(nm, build_root) if build_root is not None else {}

    dynamic_symbols = []
    binary_dyn_sym = get_binary_dynamic_symbols(readelf, binary_file)
    for symbol in binary_dyn_sym:
        if symbols is not None and symbol in symbols:
            dynamic_symbols.append(symbols[symbol])
        else:
            dynamic_symbols.append(Symbol(symbol, "", False, False, []))
    demangle_symbols(cxxfilt, dynamic_symbols)
    check_disallowed_symbols(cxxfilt, dynamic_symbols)

    dynamic_by_file = {}
    global_dynamic = []
    for symbol in dynamic_symbols:
        if len(symbol.sources) == 0:
            global_dynamic.append(symbol)
            continue

        for file in symbol.sources:
            if file not in dynamic_by_file:
                dynamic_by_file[file] = []
            dynamic_by_file[file].append(symbol)

    print("Executable relies on the following dynamic symbols:")
    for file, symbols in dynamic_by_file.items():
        print(f"{file} contains dynamic symbols:")
        for symbol in symbols:
            print(" *", symbol.demangled)
        print()

    if len(dynamic_by_file) > 0:
        return STATUS_ERROR

    return STATUS_OK


def bubble_error(program_status, routine_status) -> int:
    """Bubble a routine's error status up to the program status."""
    # A non-OK error status overrides an OK error status.
    if routine_status == STATUS_OK:
        return program_status
    elif program_status == STATUS_OK:
        return routine_status
    else:
        return min(program_status, routine_status)


def main() -> int:
    """Parse command line arguments and execute tool."""
    parser = argparse.ArgumentParser(
        description="A tool to help check binary dependencies and statically included symbols."
    )
    parser.add_argument(
        "--nm",
        metavar="executable",
        type=str,
        help="Path of the nm tool executable",
        default="nm",
    )
    parser.add_argument(
        "--readelf",
        metavar="executable",
        type=str,
        help="Path of the readelf tool executable",
        default="readelf",
    )
    parser.add_argument(
        "--cxxfilt",
        metavar="executable",
        type=str,
        help="Path of the cxxfilt tool executable",
        default="c++filt",
    )
    parser.add_argument("--binary", metavar="binary", type=str, help="Binary to check")
    parser.add_argument(
        "--buck-out", metavar="dir", type=str, help="Buck output directory"
    )
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Check shared library dependencies for a binary",
    )
    parser.add_argument(
        "--check-disallowed-symbols",
        action="store_true",
        help="Check for usage of disallowed symbols",
    )
    parser.add_argument(
        "--check-dynamic",
        action="store_true",
        help="Check for usage of dynamic symbols",
    )

    args = parser.parse_args()

    exit_status = STATUS_OK

    if args.check_dependencies:
        if args.binary is None:
            error("--binary flag must be specified when checking dependencies")
        status = check_dependencies(args.readelf, Path(args.binary))
        exit_status = bubble_error(exit_status, status)

    if args.check_disallowed_symbols:
        if args.buck_out is None:
            error("--buck-out flag must be specified when checking disallowed symbols")
        status = check_disallowed_symbols_build_dir(
            args.nm, args.cxxfilt, Path(args.buck_out)
        )
        exit_status = bubble_error(exit_status, status)

    if args.check_dynamic:
        if args.binary is None:
            error("--binary flag must be specified when checking dynamic symbol usage")
        status = check_dynamic(
            args.nm,
            args.readelf,
            args.cxxfilt,
            Path(args.binary),
            Path(args.buck_out) if args.buck_out is not None else None,
        )
        exit_status = bubble_error(exit_status, status)

    return exit_status


if __name__ == "__main__":
    sys.exit(main())
