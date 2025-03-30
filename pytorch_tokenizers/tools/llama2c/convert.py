# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT


# Script to rewrite tokenizer model given by sentencepiece to llama2.c format, with lightweight
# postprocessing logic. The output can be consumed by llama2c_tokenizer.cpp.

import argparse

from pytorch_tokenizers.llama2c import Llama2cTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tokenizer-model",
        type=str,
        default="tokenizer.model",
        help="path to tokenizer model, given by sentencepiece",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=None,
        help="output path of postprocessed tokenizer model",
    )
    parser.add_argument(
        "-p",
        "--prepend-padding",
        action="store_true",
        help="whether to prepend a padding token to the beginning of the tokenizer",
    )

    args = parser.parse_args()

    t = Llama2cTokenizer(args.tokenizer_model)

    output_path = (
        args.output_path
        if args.output_path
        else args.tokenizer_model.replace(".model", ".bin")
    )
    t.export(output_path, prepend_padding=args.prepend_padding)
