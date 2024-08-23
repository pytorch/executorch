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
from executorch.examples.models.model_factory import EagerModelFactory

from .generation import LlamaRunner


class EagerLlamaRunner(LlamaRunner):
    """
    Runs llama in eager mode with provided checkpoint file.
    """

    def __init__(self, args):
        with open(args.params, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=args.max_len,
            max_batch_size=1,
            use_kv_cache=True,
            **params,
        )
        super().__init__(tokenizer_path=args.tokenizer, model_args=model_args)
        self.model, _, _ = EagerModelFactory.create_model(
            "llama2",
            "Llama2Model",
            checkpoint=args.checkpoint,
            params=args.params,
            use_kv_cache=True,
            fairseq2=False,
            max_seq_len=args.max_len,
            enable_dynamic_shape=True,
        )

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(tokens=tokens, input_pos=input_pos)


def build_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to model checkpoint file",
    )

    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="model params file",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum length of the generated response sequence.",
    )

    parser.add_argument(
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
        default=0,
    )

    return parser


def main() -> None:
    parser = build_args_parser()
    args = parser.parse_args()

    runner = EagerLlamaRunner(args)
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
