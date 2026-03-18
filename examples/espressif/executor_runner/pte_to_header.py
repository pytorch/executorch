# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Converts an ExecuTorch .pte model file to a C header file containing
the model data as a byte array. This is used to embed the model directly
into the firmware binary for ESP32/ESP32-S3 targets.

Usage:
    python pte_to_header.py --pte model.pte [--outdir .] [--outfile model_pte.h]
"""

import binascii
import os
from argparse import ArgumentParser, ArgumentTypeError

bytes_per_line = 32
hex_digits_per_line = bytes_per_line * 2


def input_file_path(path):
    if os.path.exists(path):
        return path
    else:
        raise ArgumentTypeError(f"input filepath:{path} does not exist")


parser = ArgumentParser(description="Convert .pte model to C header for ESP32")
parser.add_argument(
    "-p",
    "--pte",
    help="ExecuTorch .pte model file",
    type=input_file_path,
    required=True,
)
parser.add_argument(
    "-d",
    "--outdir",
    help="Output dir for model header",
    type=str,
    required=False,
    default=".",
)
parser.add_argument(
    "-o",
    "--outfile",
    help="Output filename for model header",
    type=str,
    required=False,
    default="model_pte.h",
)
parser.add_argument(
    "-s",
    "--section",
    help="Section attribute for the data array (use 'none' for no section attribute)",
    type=str,
    required=False,
    default="none",
)

if __name__ == "__main__":
    args = parser.parse_args()
    outfile = os.path.join(args.outdir, args.outfile)

    if args.section == "none":
        # No section attribute - let the linker/compiler decide placement.
        # On ESP32 with PSRAM, the compiler/linker or EXT_RAM_BSS_ATTR
        # in the code handles placement.
        attr = "__attribute__((aligned(16))) static const unsigned char "
    else:
        attr = f'__attribute__((section("{args.section}"), aligned(16))) static const unsigned char '
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    with open(args.pte, "rb") as fr, open(outfile, "w") as fw:
        data = fr.read()
        hexstream = binascii.hexlify(data).decode("utf-8")
        
        fw.write(
            "/* Auto-generated model header for ESP32 ExecuTorch runner. */\n"
        )
        fw.write(f"/* Source: {os.path.basename(args.pte)} ({len(data)} bytes) */\n\n")
        fw.write("#pragma once\n\n")
        fw.write(attr + "model_pte[] = {")

        for i in range(0, len(hexstream), 2):
            if 0 == (i % hex_digits_per_line):
                fw.write("\n")
            fw.write("0x" + hexstream[i : i + 2] + ", ")

        fw.write("\n};\n")
    print(
        f"Input: {args.pte} with {len(data)} bytes. "
        f"Output: {outfile} with {os.path.getsize(outfile)} bytes."
    )
