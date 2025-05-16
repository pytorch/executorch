# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class Symbol:
    name: str
    addr: int
    size: int
    symbol_type: str


class Parser:
    def __init__(self, elf: str, toolchain_prefix: str = "", filter=None):
        self.elf = elf
        self.toolchain_prefix = toolchain_prefix
        self.symbols: Dict[str, Symbol] = self._get_nm_output()
        self.filter = filter

    @staticmethod
    def run_nm(
        elf_file_path: str, args: Optional[List[str]] = None, nm: str = "nm"
    ) -> str:
        """
        Run the nm command on the specified ELF file.
        """
        args = [] if args is None else args
        cmd = [nm] + args + [elf_file_path]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        except FileNotFoundError:
            print(f"Error: 'nm' command not found. Please ensure it's installed.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error running nm on {elf_file_path}: {e}")
            print(f"stderr: {e.stderr}")
            sys.exit(1)

    def _get_nm_output(self) -> Dict[str, Symbol]:
        args = [
            "--print-size",
            "--size-sort",
            "--reverse-sort",
            "--demangle",
            "--format=bsd",
        ]
        output = Parser.run_nm(
            self.elf,
            args,
            nm=self.toolchain_prefix + "nm" if self.toolchain_prefix else "nm",
        )
        lines = output.splitlines()
        symbols = []
        symbol_pattern = re.compile(
            r"(?P<addr>[0-9a-fA-F]+)\s+(?P<size>[0-9a-fA-F]+)\s+(?P<type>\w)\s+(?P<name>.+)"
        )

        def parse_line(line: str) -> Optional[Symbol]:

            match = symbol_pattern.match(line)
            if match:
                addr = int(match.group("addr"), 16)
                size = int(match.group("size"), 16)
                type_ = match.group("type").strip().strip("\n")
                name = match.group("name").strip().strip("\n")
                return Symbol(name=name, addr=addr, size=size, symbol_type=type_)
            return None

        for line in lines:
            symbol = parse_line(line)
            if symbol:
                symbols.append(symbol)

        assert len(symbols) > 0, "No symbols found in nm output"
        if len(symbols) != len(lines):
            print(
                "** Warning: Not all lines were parsed, check the output of nm. Parsed {len(symbols)} lines, given {len(lines)}"
            )
        if any(symbol.size == 0 for symbol in symbols):
            print("** Warning: Some symbols have zero size, check the output of nm.")

        # TODO: Populate the section and module fields from the linker map if available (-Wl,-Map=linker.map)
        return {symbol.name: symbol for symbol in symbols}

    def print(self):
        print(f"Elf: {self.elf}")

        def print_table(filter=None, filter_name=None):
            print("\nAddress\t\tSize\tType\tName")
            # Apply filter and sort symbols
            symbols_to_print = {
                name: sym
                for name, sym in self.symbols.items()
                if not filter or filter(sym)
            }
            sorted_symbols = sorted(
                symbols_to_print.items(), key=lambda x: x[1].size, reverse=True
            )

            # Print symbols and calculate total size
            size_total = 0
            for name, sym in sorted_symbols:
                print(f"{hex(sym.addr)}\t\t{sym.size}\t{sym.symbol_type}\t{sym.name}")
                size_total += sym.size

            # Print summary
            symbol_percent = len(symbols_to_print) / len(self.symbols) * 100
            print("-----")
            print(f"> Total bytes: {size_total}")
            print(
                f"Counted: {len(symbols_to_print)}/{len(self.symbols)}, {symbol_percent:0.2f}% (filter: '{filter_name}')"
            )
            print("=====\n")

        # Print tables with different filters
        def is_executorch_symbol(s):
            return "executorch" in s.name or s.name.startswith("et")

        FILTER_NAME_TO_FILTER_AND_LABEL = {
            "all": (None, "All"),
            "executorch": (is_executorch_symbol, "ExecuTorch"),
            "executorch_text": (
                lambda s: is_executorch_symbol(s) and s.symbol_type.lower() == "t",
                "ExecuTorch .text",
            ),
        }

        filter_func, label = FILTER_NAME_TO_FILTER_AND_LABEL.get(
            self.filter, FILTER_NAME_TO_FILTER_AND_LABEL["all"]
        )
        print_table(filter_func, label)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ELF file and linker map file."
    )
    parser.add_argument(
        "-e", "--elf-file-path", required=True, help="Path to the ELF file"
    )
    parser.add_argument(
        "-f",
        "--filter",
        required=False,
        default="all",
        help="Filter symbols by pre-defined filters",
        choices=["all", "executorch", "executorch_text"],
    )
    parser.add_argument(
        "-p",
        "--toolchain-prefix",
        required=False,
        default="",
        help="Optional toolchain prefix for nm",
    )

    args = parser.parse_args()
    p = Parser(args.elf_file_path, args.toolchain_prefix, filter=args.filter)
    p.print()
