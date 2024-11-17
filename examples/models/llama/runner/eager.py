# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from typing import Optional, Type

import torch

from executorch.examples.models.llama.export_llama_lib import (
    _prepare_for_llama_export,
    build_args_parser as _build_args_parser,
)
from executorch.examples.models.llama.runner.generation import LlamaRunner
from executorch.extension.llm.export.builder import LLMEdgeManager


class EagerLlamaRunner(LlamaRunner):
    """
    Runs llama in eager mode with provided checkpoint file.
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
        manager: LLMEdgeManager = _prepare_for_llama_export(args)
        self.model = manager.model.eval().to(device=self.device)

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(tokens=tokens, input_pos=input_pos)


def build_args_parser() -> argparse.ArgumentParser:
    parser = _build_args_parser()

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--show_tokens",
        action="store_true",
        default=False,
        help="Show the tokens that were generated",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help="Have multi-turn chat with the model",
    )

    return parser


def execute_runner(runner_class: Type[LlamaRunner]) -> None:
    parser = build_args_parser()
    args = parser.parse_args()

    with torch.no_grad():
        runner = runner_class(args)  # pyre-ignore: Missing argument [20]
        generated_tokens = (
            runner.chat_completion(
                max_seq_len=1000000 if args.use_attention_sink else args.max_seq_length,
                temperature=args.temperature,
                show_progress=args.show_tokens,
            )
            if args.chat
            else runner.text_completion(
                prompt=args.prompt,
                temperature=args.temperature,
                echo=True,
            )
        )
        if args.show_tokens:
            print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")


def main() -> None:
    execute_runner(EagerLlamaRunner)


if __name__ == "__main__":
    main()  # pragma: no cover
