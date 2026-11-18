# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from executorch_native import cmake_prefix_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate the ExecuTorch Native SDK")
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument("--cmake-prefix-path", action="store_true")
    paths.add_argument("--example-path", action="store_true")
    args = parser.parse_args()
    prefix = Path(cmake_prefix_path())
    print(prefix / "share/examples/vision" if args.example_path else prefix)


if __name__ == "__main__":
    main()
