# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Write the `webgpu_topk_test` fixture corpus from the sealed CPU authority."""

from __future__ import annotations

import json
import struct
import sys

from pathlib import Path
from typing import Mapping, Sequence

from executorch.backends.webgpu.test.ops.topk.test_topk import (
    authority_digest,
    AUTHORITY_SHA256,
    INPUT_WIDTH,
    OUTPUT_WIDTH,
)

CASES_MANIFEST = "cases.txt"


def _write_u32(path: Path, words: Sequence[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(words)}I", *words))


def _write_i32(path: Path, values: Sequence[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def _load_sealed_authority(authority_path: Path) -> Mapping[str, object]:
    sealed = json.loads(authority_path.read_text(encoding="utf-8"))
    digest = authority_digest(sealed)
    if digest != AUTHORITY_SHA256 or sealed.get("sha256") != AUTHORITY_SHA256:
        raise ValueError(
            f"top-k authority {authority_path} does not match the committed "
            f"digest {AUTHORITY_SHA256} (body {digest}, seal "
            f"{sealed.get('sha256')!r}); a deliberate authority change has to "
            "update AUTHORITY_SHA256 in test_topk.py"
        )
    return sealed


def export_topk_artifacts(output_dir: Path, authority_path: Path) -> list[str]:
    authority = _load_sealed_authority(authority_path)
    cases = authority["cases"]
    assert isinstance(cases, dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(cases)
    for name in names:
        case = cases[name]
        assert isinstance(case, dict)
        scores = case["scores_bits"]
        values = case["values_bits"]
        indices = case["indices"]
        assert isinstance(scores, list) and isinstance(values, list)
        assert isinstance(indices, list)
        if len(scores) != INPUT_WIDTH or len(values) != OUTPUT_WIDTH:
            raise ValueError(f"top-k authority case {name} has the wrong width")
        if len(indices) != OUTPUT_WIDTH:
            raise ValueError(f"top-k authority case {name} has the wrong index width")
        _write_u32(output_dir / f"{name}.scores.bin", scores)
        _write_u32(output_dir / f"{name}.values.bin", values)
        _write_i32(output_dir / f"{name}.indices.bin", indices)

    (output_dir / CASES_MANIFEST).write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit(
            "usage: export_topk_artifacts.py <output-dir> <topk-authority.json>"
        )
    export_topk_artifacts(Path(args[0]), Path(args[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
