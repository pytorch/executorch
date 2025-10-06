# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Check log files for memory metrics and compare them against thresholds.

Usage example:
  python3 test_memory_allocator_log.py \
    --log path/to/log.txt \
    --require "Total SRAM used" "<= 310 KiB" \
    --require "method_allocator_input" "<= 4 B"
"""

import argparse
import re
import sys
from typing import List, Optional, Tuple


def unit_factor(u: str) -> float:
    if not u:
        return 1.0
    ul = u.strip().lower()
    table = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    if ul in table:
        return float(table[ul])
    return 1.0


def parse_value(text_num: str, text_unit: Optional[str]) -> float:
    return float(text_num) * unit_factor(text_unit or "")


def parse_cond(cond: str) -> Tuple[str, float, str]:
    # Regexp explained. Example of things it will parse:
    # "< 310 KiB", ">=10MB", "== 42", "!=3 bytes", "<=0.5 MiB"

    # The regexp explained in detail:
    # ^: anchor the match to the start and end of the string (no extra chars allowed).
    # \s*: optional whitespace (spaces, tabs, etc.).
    # (<=|>=|==|!=|<|>): capturing group 1. One of the comparison operators: <=, >=, ==, !=, <, >.
    # \s*: optional whitespace.
    # ([0-9]+(?:\.[0-9]+)?): capturing group 2. A number:
    #   [0-9]+: one or more digits (the integer part).
    #   (?:\.[0-9]+)?: optional non-capturing group for a fractional part like .25.
    # \s*: optional whitespace between number and unit
    # ([A-Za-z]+)?: capturing group 3, optional. A unit made of letters only (e.g., B, KB, KiB, MB, MiB). Case# insensitive by class choice.
    # \s*: optional trailing whitespace.
    m = re.match(
        r"^\s*(<=|>=|==|!=|<|>)\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?\s*$", cond
    )
    if not m:
        raise ValueError(f"Invalid condition: {cond}")
    op, num, unit = m.groups()
    return op, float(num), (unit or "")


def compare(a: float, b: float, op: str) -> bool:
    return {
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
        "==": abs(a - b) < 1e-9,
        "!=": abs(a - b) >= 1e-9,
    }[op]


def find_metric_value(line: str, label: str) -> Tuple[Optional[str], Optional[str]]:
    # Same regexp as parse_cond() but without the first group of matching comparison operators
    # First go, search for the pattern but escape and ignore cases
    # The regexp:
    # ([0-9]+(?:\.[0-9]+)?) — capturing group 1: a decimal number
    # [0-9]+ — one or more digits (integer part)
    # (?:\.[0-9]+)? — optional fractional part like .25 (non-capturing)
    # \s* — optional whitespace between number and unit
    # ([A-Za-z]+)? — capturing group 2 (optional): a unit made only of letters (e.g., B, KB, KiB, MB)
    m = re.search(
        re.escape(label) + r".*?([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?",
        line,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2)
    # Second go, same regexp as above but not caring about label. If
    # no number was tied to a label be happy just salvaging it from
    # the line
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?", line)
    if m:
        return m.group(1), m.group(2)
    return None, None


def first_line_with_label(lines: List[str], label: str) -> Optional[str]:
    label_lc = label.lower()
    return next((ln for ln in lines if label_lc in ln.lower()), None)


def check_requirement(label: str, cond: str, lines: List[str]) -> Optional[str]:
    op, thr_num, thr_unit = parse_cond(cond)
    matched = first_line_with_label(lines, label)
    if matched is None:
        return f"{label}: not found in log"

    num_str, unit_str = find_metric_value(matched, label)
    if num_str is None:
        return f"{label}: value not found on line: {matched.strip()}"

    left_bytes = parse_value(num_str, unit_str)
    right_bytes = parse_value(str(thr_num), thr_unit or (unit_str or ""))
    ok = compare(left_bytes, right_bytes, op)

    human_left = f"{num_str} {unit_str or 'B'}"
    human_right = f"{thr_num:g} {thr_unit or (unit_str or 'B')}"
    print(
        f"[check] {label}: {human_left} {op} {human_right} -> {'OK' if ok else 'FAIL'}"
    )

    if ok:
        return None
    return f"{label}: {human_left} not {op} {human_right}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to log file")
    parser.add_argument(
        "--require",
        action="append",
        nargs=2,
        metavar=("LABEL", "COND"),
        default=[],
        help="""Required label and condition consisting
                         of a number and unit. Example: \"Total DRAM
                         used\" \"<= 0.06 KiB\"""",
    )
    args = parser.parse_args()

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    failures: List[str] = []
    for label, cond in args.require:
        msg = check_requirement(label, cond, lines)
        if msg:
            failures.append(msg)

    if failures:
        print("Failures:")
        for msg in failures:
            print(" - " + msg)
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
