#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a .pte into a C header an Arduino sketch can include.

    python pte_to_header.py -p model.pte -o model.h

examples/arm/executor_runner/pte_to_header.py places the array in a
network_model_sec section for the Ethos-U linker script. No Arduino core
defines that section, so this emits a plain rodata array instead.
"""

import argparse
import os
import re

BANNER = """\
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generated from {source} by pte_to_header.py. Do not edit.
"""


C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def c_identifier(value: str) -> str:
    """argparse type that rejects names that cannot appear in C++."""
    if not C_IDENTIFIER.match(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a C identifier; it is emitted verbatim as an "
            "array name, so anything else fails to compile"
        )
    return value


def to_header(buffer: bytes, name: str = "model_pte", source: str = "a .pte") -> str:
    if not C_IDENTIFIER.match(name):
        raise ValueError(
            f"--name must be a C identifier, got {name!r}. The value is emitted "
            "verbatim as an array name, so anything else fails to compile."
        )
    out = [BANNER.format(source=source)]
    out.append("#pragma once")
    out.append("#include <cstddef>")
    out.append("#include <cstdint>")
    out.append("")
    out.append(f"alignas(16) static const uint8_t {name}[] = {{")
    for i in range(0, len(buffer), 16):
        out.append("    " + ",".join(f"0x{b:02x}" for b in buffer[i : i + 16]) + ",")
    out.append("};")
    out.append(f"static const size_t {name}_size = {len(buffer)};")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--pte", required=True, help="Input .pte file")
    parser.add_argument("-o", "--output", required=True, help="Output .h file")
    parser.add_argument("-d", "--outdir", default="", help="Directory for --output")
    parser.add_argument(
        "-n",
        "--name",
        default="model_pte",
        type=c_identifier,
        help="C array name (must be a valid C identifier)",
    )
    args = parser.parse_args()

    out = os.path.join(args.outdir, args.output) if args.outdir else args.output

    with open(args.pte, "rb") as f:
        buffer = f.read()

    with open(out, "w") as f:
        f.write(to_header(buffer, args.name, os.path.basename(args.pte)))

    print(f"{out}: {len(buffer)} bytes ({len(buffer) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
