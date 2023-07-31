#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Strip a binary file using the ELF `strip` tool specified by a Skycastle workflow.

Usage:
    strip_binary.py input_path output_path

    Strip the ELF binary given by `input_path`, outputting the stripped
    binary to `output_path`.
"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Must specify input and output file paths")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Assumes `strip` tool is in the path (should be specified by Skycastle workflow).
    # GNU `strip`, or equivalent, should work for x86 and ARM ELF binaries. This might
    # not be appropriate for more exotic, non-ELF toolchains.
    completed = subprocess.run(["strip", "--strip-all", input_file, "-o", output_file])
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
