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
    _convert_args_to_config,
    _prepare_for_llama_export,
    build_args_parser as _build_args_parser,
)
from executorch.examples.models.llama.runner.generation import LlamaRunner
from executorch.extension.llm.export.builder import LLMEdgeManager
from omegaconf import DictConfig, OmegaConf


class EagerLlamaRunner(LlamaRunner):
    """
    Runs llama in eager mode with provided checkpoint file.
    """

    def __init__(self, config):
        with open(config.model.params, "r") as f:
            params = json.loads(f.read())
        super().__init__(
            tokenizer_path=config.model.tokenizer_path,
            tokenizer_config_path=config.eager.tokenizer_config_path,
            max_seq_len=config.sequence.max_seq_length,
            max_batch_size=1,
            use_kv_cache=config.kv_cache.use_kv_cache,
            vocab_size=params["vocab_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        manager: LLMEdgeManager = _prepare_for_llama_export(config)
        self.model = manager.model.eval().to(device=self.device)

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(tokens, {"input_pos": input_pos})


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

    parser.add_argument(
        "--tokenizer_config_path",
        type=str,
        default=None,
        help="Path to an accompanying tokenizer_config.json, which provides metadata for the main tokenizer.json",
    )

    return parser


def _convert_cli_to_config_format(args) -> DictConfig:
    """Convert CLI arguments to config format."""
    # First convert common args using the shared function
    config = _convert_args_to_config(args)

    # Add evaluation-specific settings
    config.eager = OmegaConf.create()
    config.eager.prompt = args.prompt
    config.eager.temperature = args.temperature
    config.eager.show_tokens = args.show_tokens
    config.eager.chat = args.chat
    config.eager.tokenizer_config_path = args.tokenizer_config_path

    return config


def execute_runner(runner_class: Type[LlamaRunner]) -> None:
    parser = build_args_parser()
    args = parser.parse_args()
    config = _convert_cli_to_config_format(args)
    with torch.no_grad():
        runner = runner_class(config)  # pyre-ignore: Missing argument [20]
        generated_tokens = (
            runner.chat_completion(
                max_seq_len=(
                    1000000
                    if config.misc.use_attention_sink
                    else config.sequence.max_seq_length
                ),
                temperature=config.eager.temperature,
                show_progress=config.eager.show_tokens,
            )
            if config.eager.chat
            else runner.text_completion(
                prompt=config.eager.prompt,
                temperature=config.eager.temperature,
                echo=True,
            )
        )
        if config.eager.show_tokens:
            print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")


def main() -> None:
    execute_runner(EagerLlamaRunner)


if __name__ == "__main__":
    main()  # pragma: no cover
