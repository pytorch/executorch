# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Write the `webgpu_scatter_test` fixture corpus from the CPU authority."""

from __future__ import annotations

import struct
import sys

from pathlib import Path
from typing import Sequence

from executorch.backends.webgpu.test.ops.scatter.test_scatter import (
    base_row,
    destinations_are_pairwise_distinct,
    EQUIVALENT_CASES,
    has_official_provenance,
    PROVENANCE_CASES,
    scatter_cases,
    scatter_parallel,
    scatter_serial,
    SELECTED_COUNT,
)

BASE_FIXTURE = "base.bin"
CASES_MANIFEST = "cases.txt"


def _write_f32(path: Path, values: Sequence[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def _write_i32(path: Path, values: Sequence[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def export_scatter_artifacts(output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_row()
    _write_f32(output_dir / BASE_FIXTURE, base)

    cases = scatter_cases()
    lines: list[str] = []
    for name in sorted(cases):
        case = cases[name]
        indices = case["indices"]
        source = case["source"]
        assert isinstance(indices, list) and isinstance(source, list)
        if len(indices) != SELECTED_COUNT or len(source) != SELECTED_COUNT:
            raise ValueError(f"scatter case {name} has the wrong width")
        equivalent = destinations_are_pairwise_distinct(indices)
        provenance = has_official_provenance(indices)
        if equivalent != (name in EQUIVALENT_CASES):
            raise ValueError(
                f"scatter case {name} mislabels parallel-route equivalence"
            )
        if provenance != (name in PROVENANCE_CASES):
            raise ValueError(f"scatter case {name} mislabels official provenance")
        expected = scatter_serial(base, indices, source)
        if (scatter_parallel(base, indices, source) == expected) != equivalent:
            raise ValueError(
                f"scatter case {name} disagrees with its parallel-route label"
            )
        _write_i32(output_dir / f"{name}.index.bin", indices)
        _write_f32(output_dir / f"{name}.source.bin", source)
        _write_f32(output_dir / f"{name}.expected.bin", expected)
        lines.append(f"{name} {1 if equivalent else 0} {1 if provenance else 0}")

    (output_dir / CASES_MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [line.split(" ", 1)[0] for line in lines]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: export_scatter_artifacts.py <output-dir>")
    export_scatter_artifacts(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
