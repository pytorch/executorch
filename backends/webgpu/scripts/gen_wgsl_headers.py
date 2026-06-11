#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate runtime/ops/<op>/<stem>_wgsl.h from each <stem>.wgsl.

Each header embeds the shader verbatim as `inline constexpr const char*
k<Pascal>WGSL` plus `k<Pascal>WorkgroupSize` (parsed from @workgroup_size).

Usage:
  gen_wgsl_headers.py            # (re)write all <stem>_wgsl.h
  gen_wgsl_headers.py --check    # exit 1 if any committed header is stale

Stdlib only (the devserver has no third-party pip).
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

_SHA_RE = re.compile(r"// wgsl-sha256: ([0-9a-f]{64})")

_BSD_HEADER = """\
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */"""


def symbol_base(stem: str) -> str:
    """snake_case shader stem -> PascalCase symbol base (binary_add -> BinaryAdd)."""
    return "".join(part.capitalize() for part in stem.split("_"))


_INT_LITERAL_RE = re.compile(r"^(\d+)[uUiI]?$")


def _resolve_dim(tok: str, src: str) -> int:
    """Resolve one @workgroup_size dim token: a literal or an override/const ident.

    Accepts WGSL suffix-typed integer literals (e.g. `64u`, `64i`) both as the
    token and on the right-hand side of an `override`/`const` (type optional).
    """
    lit = _INT_LITERAL_RE.match(tok)
    if lit:
        return int(lit.group(1))
    m = re.search(
        r"(?:override|const)\s+"
        + re.escape(tok)
        + r"\s*(?::\s*u32\s*)?=\s*(\d+)[uUiI]?",
        src,
    )
    if not m:
        raise ValueError(f"cannot resolve @workgroup_size identifier '{tok}'")
    return int(m.group(1))


def parse_workgroup_size(src: str) -> tuple[int, int, int]:
    """Resolve the (x, y, z) dims of @workgroup_size; y and z default to 1."""
    m = re.search(r"@workgroup_size\s*\(([^)]*)\)", src)
    if not m:
        raise ValueError("no @workgroup_size found")
    toks = [t.strip() for t in m.group(1).split(",") if t.strip()]
    if not toks or len(toks) > 3:
        raise ValueError(f"@workgroup_size takes 1-3 dims, got {len(toks)}")
    dims = [_resolve_dim(t, src) for t in toks]
    while len(dims) < 3:
        dims.append(1)
    return (dims[0], dims[1], dims[2])


def wgsl_sha256(wgsl_text: str) -> str:
    return hashlib.sha256(wgsl_text.encode("utf-8")).hexdigest()


def embedded_sha256(header_text: str) -> str:
    m = _SHA_RE.search(header_text)
    return m.group(1) if m else ""


def render_header(wgsl_path, wgsl_text: str) -> str:
    """Render the full <stem>_wgsl.h text for a shader (shader embedded verbatim)."""
    if ')"' in wgsl_text:
        raise ValueError('shader contains )" which would close the R"( literal')
    stem = Path(wgsl_path).stem
    base = symbol_base(stem)
    x, y, z = parse_workgroup_size(wgsl_text)

    head = [
        _BSD_HEADER,
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace executorch::backends::webgpu {",
        "",
        f"// @generated from {stem}.wgsl - DO NOT EDIT.",
        f"// wgsl-sha256: {wgsl_sha256(wgsl_text)}",
        f'inline constexpr const char* k{base}WGSL = R"(',
    ]
    return (
        "\n".join(head)
        + "\n"
        + wgsl_text
        + ')";'
        + "\n\n"
        + f"inline constexpr uint32_t k{base}WorkgroupSizeX = {x};\n"
        + f"inline constexpr uint32_t k{base}WorkgroupSizeY = {y};\n"
        + f"inline constexpr uint32_t k{base}WorkgroupSizeZ = {z};\n\n"
        + "} // namespace executorch::backends::webgpu\n"
    )


def discover():
    """All shader sources under runtime/ops, sorted."""
    return sorted((BACKEND_ROOT / "runtime/ops").glob("**/*.wgsl"))


def _report_drift(missing, stale) -> None:
    """Print the --check report for missing/stale committed headers."""
    if missing:
        print("Missing embedded WGSL headers (run scripts/gen_wgsl_headers.py):")
        for h in missing:
            print(f"  {h.relative_to(BACKEND_ROOT)}")
    if stale:
        print("Stale embedded WGSL headers (run scripts/gen_wgsl_headers.py):")
        for h in stale:
            print(f"  {h.relative_to(BACKEND_ROOT)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed headers match (exit 1 on drift)",
    )
    args = parser.parse_args(argv)

    stale = []
    missing = []
    errors = []
    for wgsl in discover():
        wgsl_text = wgsl.read_text()
        try:
            want = render_header(wgsl, wgsl_text)
        except ValueError as e:
            errors.append(f"{wgsl.relative_to(BACKEND_ROOT)}: {e}")
            continue
        header = wgsl.with_name(wgsl.stem + "_wgsl.h")
        # Full-content compare (not just the sha) catches generator-logic drift too.
        if header.exists() and header.read_text() == want:
            continue
        if args.check:
            (missing if not header.exists() else stale).append(header)
        else:
            header.write_text(want)

    if errors:
        print("Cannot generate header (malformed shader):")
        for e in errors:
            print(f"  {e}")
        return 1
    if args.check and (stale or missing):
        _report_drift(missing, stale)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
