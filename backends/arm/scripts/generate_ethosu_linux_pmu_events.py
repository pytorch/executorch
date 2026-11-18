# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate symbolic PMU event tables in Linux driver configuration order."""

import argparse
import re
from pathlib import Path


def parse_events(header: str) -> list[str]:
    """Read event names in Linux configuration order.

    Args:
        header (str): Product-specific Linux driver interface header.

    Returns:
        list[str]: Event-name suffixes in lookup-table order.

    Raises:
        ValueError: The lookup table has an unsupported shape or size.

    """
    header = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.DOTALL)
    count = re.search(r"#define\s+PMU_EVENT_MAX_IDX\s+(\d+)\b", header)
    table = re.search(
        r"static\s+(?:const\s+)?uint32_t\s+pmu_event_lookup\s*"
        r"\[\s*PMU_EVENT_MAX_IDX\s*\]\s*=\s*\{(.*?)\}\s*;",
        header,
        flags=re.DOTALL,
    )
    if count is None or table is None:
        raise ValueError("Missing Linux PMU lookup table or event count")
    entries = table[1].strip().removesuffix(",").split(",")
    events = []
    for entry in entries:
        match = re.fullmatch(r"\s*PMU_EVENT_([A-Z0-9_]+)\s*", entry)
        if match is None:
            raise ValueError(f"Unsupported Linux PMU lookup entry: {entry!r}")
        events.append(match[1])
    if len(events) != int(count[1]) or len(set(events)) != len(events):
        raise ValueError("Linux PMU lookup table count or unique names mismatch")
    if events[0] != "NO_EVENT":
        raise ValueError("Linux PMU lookup index zero must be NO_EVENT")
    return events


def main() -> None:
    """Generate the header from the selected Linux driver checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u65-header", type=Path, required=True)
    parser.add_argument("--u85-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sections = [
        "// Generated from the selected Linux driver PMU lookup tables.\n",
        "#pragma once\n\n#include <ethosu_pmu_types.h>\n\n",
        "namespace executorch::backends::arm {\n",
    ]
    for product, header in (("U65", args.u65_header), ("U85", args.u85_header)):
        try:
            events = parse_events(header.read_text())
        except ValueError as error:
            parser.error(f"{header}: {error}")
        sections.append(
            f"constexpr ethosu_pmu_event_type kLinux{product}PmuEvents[] = {{\n"
        )
        sections.extend(f"    ETHOSU_PMU_{event},\n" for event in events)
        sections.append("};\n\n")
    sections.append("} // namespace executorch::backends::arm\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(sections))


if __name__ == "__main__":
    main()
