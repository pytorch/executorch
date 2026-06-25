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


def validate(pack_file: str) -> None:
    with zipfile.ZipFile(pack_file, "r") as z:
        names = z.namelist()
        names_set = set(names)

        pdsc_names = [n for n in names if n.endswith(".pdsc")]
        assert pdsc_names, "No PDSC file found in pack"
        assert any("runtime" in n for n in names), "No runtime sources found in pack"
        assert any(
            "RegisterAllKernels" in n for n in names
        ), "No RegisterAllKernels.cpp found"

        if len(names) != len(names_set):
            dupes = sorted({n for n in names if names.count(n) > 1})
            sys.exit(f"ERROR: duplicate entries in pack: {dupes[:5]}")

        py = [n for n in names if n.endswith(".py")]
        assert not py, f"Python files leaked into pack: {py[:5]}"

        pdsc = pdsc_names[0]
        content = z.read(pdsc).decode()
        try:
            # The XML input is the PDSC we just generated and ZIP'd,
            # so it is trusted; defusedxml is not pulled in.
            root = ET.fromstring(content)  # nosec
        except ET.ParseError as e:
            sys.exit(f"ERROR: PDSC is not well-formed XML: {e}")

        def exists(ref: str) -> bool:
            if ref in names_set:
                return True
            prefix = ref if ref.endswith("/") else ref + "/"
            return any(n.startswith(prefix) for n in names)

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
