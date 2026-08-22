# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from typing import Optional

import torch

from executorch.examples.models.llama.export_llama_lib import (
    build_args_parser as _build_args_parser,
)
from executorch.examples.models.llama3_2_vision.runner.generation import (
    TorchTuneLlamaRunner,
)


class ExportedLlamaRunner(TorchTuneLlamaRunner):
    """
    Runs a torch-exported .pt2 Llama.
    """

    def __init__(self, args):
        with open(args.params, "r") as f:
            params = json.loads(f.read())
        super().__init__(
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_length,
            max_batch_size=1,
            use_kv_cache=args.use_kv_cache,
            vocab_size=params["vocab_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"Loading model from {args.pt2}")
        self.model = torch.export.load(args.pt2).module()
        print("Model loaded")

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.use_kv_cache:
            return self.model(tokens, input_pos=input_pos, mask=mask)
        else:
            return self.model(tokens)


def build_args_parser() -> argparse.ArgumentParser:
    parser = _build_args_parser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
    )

    parser.add_argument(
        "--pt2",
        type=str,
        required=True,
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

    runner = ExportedLlamaRunner(args)
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
