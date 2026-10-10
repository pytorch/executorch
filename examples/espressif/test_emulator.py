# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the emulator log from the ESP32-S3 add/multiply smoke test."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Captured emulator output")
    args = parser.parse_args()
    output = args.log.read_text(errors="replace")

    if "Test_result: FAIL" in output:
        raise SystemExit(f"FAIL: Model output verification failed; see {args.log}")
    # A clean emulator exit alone does not prove the firmware completed.
    for marker in (
        "ESP32 ExecuTorch runner initialized.",
        "SPI SRAM memory test OK",
        "3 inferences finished",
        "TEST: BundleIO index[0] Test_result: PASS",
        "Program complete.",
    ):
        if marker not in output:
            raise SystemExit(f"FAIL: Missing {marker!r}; see {args.log}")
    print("PASS: ESP32-S3 ran three inferences and matched both PyTorch outputs.")


if __name__ == "__main__":
    main()
