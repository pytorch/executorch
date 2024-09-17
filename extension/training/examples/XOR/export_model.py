# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse

import torch

from .export_model_lib import export_model


def main() -> None:
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(
        prog="export_model",
        description="Exports an nn.Module model to ExecuTorch .pte files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Path to the directory to write xor.pte files to",
    )
    args = parser.parse_args()
    export_model(args.outdir)


if __name__ == "__main__":
    main()
