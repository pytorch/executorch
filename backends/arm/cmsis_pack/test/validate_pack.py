#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Structural validation of a built CMSIS Pack archive.

Asserts the pack is well-formed before any consumer attempts to build
against it: a PDSC is present and parses as XML, the runtime + kernel
registration sources are shipped, no duplicate or leaked-Python
entries, and every <file name="..."/> in the PDSC resolves to a real
entry in the archive (or a directory prefix covering one).

"""

import argparse
import sys
import xml.etree.ElementTree as ET  # nosec  # noqa: B405,S405
import zipfile
from collections import Counter


def validate(pack_file: str) -> None:  # noqa: C901
    with zipfile.ZipFile(pack_file, "r") as z:
        names = z.namelist()
        names_set = set(names)

        # Use explicit sys.exit (not assert) so `python -O` does not
        # silently strip these critical checks.
        pdsc_names = [n for n in names if n.endswith(".pdsc")]
        if not pdsc_names:
            sys.exit("No PDSC file found in pack")
        if not any("runtime" in n for n in names):
            sys.exit("No runtime sources found in pack")
        if not any("RegisterAllKernels" in n for n in names):
            sys.exit("No RegisterAllKernels.cpp found")

        # Build-tree-sourced files: copy_sources.sh takes these from the CMake
        # build dir, and a pack without them compiles nothing on the consumer
        # side (a skipped cross-compile step once produced such a pack).
        for required in (
            "include/flatbuffers/flatbuffers.h",
            # Redistributing the flatbuffers headers requires shipping their
            # licence; a pack without it must not be published.
            "include/flatbuffers/LICENSE",
            "include/executorch/schema/program_generated.h",
            "include/executorch/schema/scalar_type_generated.h",
        ):
            if not any(n.endswith(required) for n in names):
                sys.exit(
                    f"Missing build-generated file: {required} "
                    "(was the CMake configure+build step run before packing?)"
                )

        if len(names) != len(names_set):
            counts = Counter(names)
            dupes = sorted(n for n, c in counts.items() if c > 1)
            sys.exit(f"ERROR: duplicate entries in pack: {dupes[:5]}")

        py = [n for n in names if n.endswith(".py")]
        if py:
            sys.exit(f"Python files leaked into pack: {py[:5]}")

        pdsc = pdsc_names[0]
        content = z.read(pdsc).decode()
        try:
            # The XML input is the PDSC we just generated and ZIP'd,
            # so it is trusted; defusedxml is not pulled in.
            root = ET.fromstring(content)  # nosec
        except ET.ParseError as e:
            sys.exit(f"ERROR: PDSC is not well-formed XML: {e}")

        # Precompute every directory prefix that exists in the archive so
        # exists() is two O(1) set lookups instead of an O(n) scan per
        # <file> reference (the PDSC has hundreds; the archive has
        # thousands of entries).
        all_prefixes = set()
        for n in names:
            parts = n.split("/")
            for i in range(1, len(parts)):
                all_prefixes.add("/".join(parts[:i]) + "/")

        def exists(ref: str) -> bool:
            if ref in names_set:
                return True
            prefix = ref if ref.endswith("/") else ref + "/"
            return prefix in all_prefixes

        missing = [
            f.attrib["name"]
            for f in root.iter("file")
            if "name" in f.attrib and not exists(f.attrib["name"])
        ]
        if missing:
            print(f"ERROR: {len(missing)} PDSC file refs missing from archive")
            for m in missing[:10]:
                print(f"  {m}")
            sys.exit(1)

        size_kb = sum(i.file_size for i in z.infolist()) / 1024
        op_count = content.count('Csub="Portable')
        q_count = content.count('Csub="Quantized')
        print(f"Pack: {len(names)} files, {size_kb:.0f} KiB uncompressed")
        print(f"Portable operator components: {op_count}")
        print(f"Quantized operator components: {q_count}")

    print("Pack validation passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack_file", help="Path to the .pack archive to validate")
    args = ap.parse_args()
    validate(args.pack_file)


if __name__ == "__main__":
    main()
