# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.sdk import Inspector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--etdump_path",
        required=True,
        help="Provide an ETDump file path.",
    )
    parser.add_argument(
        "--etrecord_path",
        required=False,
        help="Provide an optional ETRecord file path.",
    )

    args = parser.parse_args()

    inspector = Inspector(etdump_path=args.etdump_path, etrecord=args.etrecord_path)
    inspector.print_data_tabular()
