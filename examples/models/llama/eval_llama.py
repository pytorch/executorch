# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for evaluating pre-to_edge Llama

import logging

import torch

from .eval_llama_lib import build_args_parser, eval_llama

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    modelname = "llama2"
    parser = build_args_parser()
    args = parser.parse_args()
    # Overrides this arg, because evaluation requires full logits.
    args.generate_full_logits = True
    eval_llama(modelname, args)


if __name__ == "__main__":
    main()  # pragma: no cover
