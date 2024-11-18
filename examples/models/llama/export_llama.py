# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting Llama2 to flatbuffer

import logging
import sys

import torch

from .export_llama_lib import build_args_parser, export_llama

sys.setrecursionlimit(4096)


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    parser = build_args_parser()
    args = parser.parse_args()
    export_llama(args)


if __name__ == "__main__":
    main()  # pragma: no cover
