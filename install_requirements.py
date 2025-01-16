# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

import install_executorch


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-pt-pinned-commit",
        action="store_true",
        help="build from the pinned PyTorch commit instead of nightly",
    )
    args = parser.parse_args(args)
    install_executorch.install_requirements(
        use_pytorch_nightly=not bool(args.use_pt_pinned_commit)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
