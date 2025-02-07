# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import binascii

bytes_per_line = 32
hex_digits_per_line = bytes_per_line * 2

magic_attr = '__attribute__((section(".sram.data"), aligned(16))) uint8_t'


def gen_header(model_path, header_path=None):
    if header_path is not None:
        header_path = header_path + "/model_pte.h"
    else:
        header_path = "model_pte.h"

    with open(model_path, "rb") as fr, open(header_path, "w+") as fw:
        data = fr.read()
        hexstream = binascii.hexlify(data).decode("utf-8")

        hexstring = magic_attr + " model_pte[] = {"

        for i in range(0, len(hexstream), 2):
            if 0 == (i % hex_digits_per_line):
                hexstring += "\n"
            hexstring += "0x" + hexstream[i : i + 2] + ", "

        hexstring += "};\n"
        fw.write(hexstring)
        print(f"Wrote {len(hexstring)} bytes, original {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", required=True, help=".pte file to generate header from."
    )
    parser.add_argument(
        "--header_output_path", help="Output path where the header should be placed."
    )

    args = parser.parse_args()

    gen_header(args.model_path, args.header_output_path)
