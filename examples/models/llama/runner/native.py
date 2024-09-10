# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from typing import Optional

import torch

from examples.models.llama2.llama_transformer import ModelArgs
from executorch.extension.pybindings.portable_lib import _load_for_executorch

# Load custom ops and quantized ops.
from executorch.extension.pybindings import portable_lib  # noqa # usort: skip

# Note: import this after portable_lib
from executorch.extension.llm.custom_ops import sdpa_with_kv_cache  # noqa # usort: skip
from executorch.kernels import quantized  # noqa

from .generation import LlamaRunner


class NativeLlamaRunner(LlamaRunner):
    """
    Runs llama via ExecuTorch with provided pte file.
    """

    def __init__(self, args):
        with open(args.params, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=args.max_len,
            max_batch_size=1,
            use_kv_cache=args.kv_cache,
            **params,
        )
        super().__init__(tokenizer_path=args.tokenizer, model_args=model_args)
        self.model = _load_for_executorch(args.pte)

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return (
            self.model.forward((tokens, input_pos))
            if input_pos is not None
            else self.model.forward((tokens,))
        )[0]


def build_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--pte",
        type=str,
        default=None,
        help="path to exported executorch .pte file",
    )

    parser.add_argument(
        "-p", "--params", type=str, default=None, help="model params file"
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "-kv",
        "--kv_cache",
        default=True,
        action="store_true",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum length of the generated response sequence.",
    )

    return parser


def main() -> None:
    parser = build_args_parser()
    args = parser.parse_args()
    runner = NativeLlamaRunner(args)
    result = runner.text_completion(
        prompt=args.prompt,
        temperature=args.temperature,
    )
    print(
        "Response: \n{response}\n Tokens:\n {tokens}".format(
            response=result["generation"], tokens=result["tokens"]
        )
    )


if __name__ == "__main__":
    main()  # pragma: no cover
