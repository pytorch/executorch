# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.devtools import Inspector


def generate_csv(etdump_path, output):

    inspector = Inspector(etdump_path)
    df = inspector.to_dataframe()
    df.to_csv(output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate profiling CSV from a model's etdump"
    )
    parser.add_argument(
        "--etdump_path",
        type=str,
        default="./model.etdump",
        help="Path to the etdump file",
        required=False,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./model_profiling.csv",
        help="Path to the output CSV file",
        required=False,
    )

    args = parser.parse_args()
    print(f"Generating CSV from {args.etdump_path}")
    generate_csv(args.etdump_path, args.output)
    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    main()
