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

from executorch.extension.llm.export.config.llm_config import LlmConfig


class EagerLlamaRunner(LlamaRunner):
    """
    Runs llama in eager mode with provided checkpoint file.
    """

    def __init__(
        self,
        llm_config: LlmConfig,
        tokenizer_config_path: Optional[str] = None,
        use_attention_sink: bool = False,
    ):
        with open(llm_config.base.params, "r") as f:
            params = json.loads(f.read())
        super().__init__(
            tokenizer_path=llm_config.base.tokenizer_path,
            tokenizer_config_path=tokenizer_config_path,
            max_seq_len=llm_config.export.max_seq_length,
            max_batch_size=1,
            use_kv_cache=llm_config.model.use_kv_cache,
            vocab_size=params["vocab_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        manager: LLMEdgeManager = _prepare_for_llama_export(llm_config)
        self.model = manager.model.eval().to(device=self.device)

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(tokens, {"input_pos": input_pos})


def build_args_parser() -> argparse.ArgumentParser:
    parser = _build_args_parser()

    # Runner-specific arguments that aren't part of LlmConfig
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


def execute_runner(runner_class: Type[LlamaRunner]) -> None:
    parser = build_args_parser()
    args = parser.parse_args()

    # Convert args to LlmConfig for model configuration.
    llm_config = LlmConfig.from_args(args)

    # Extract runner-specific parameters.
    prompt = args.prompt
    temperature = args.temperature
    show_tokens = args.show_tokens
    chat_mode = args.chat
    tokenizer_config_path = args.tokenizer_config_path
    use_attention_sink = args.use_attention_sink

    with torch.no_grad():
        # Create runner with LlmConfig and separate runner parameters.
        runner = runner_class(
            llm_config=llm_config,
            tokenizer_config_path=tokenizer_config_path,
            use_attention_sink=use_attention_sink,
        )

        generated_tokens = (
            runner.chat_completion(
                max_seq_len=(
                    1000000 if use_attention_sink else llm_config.export.max_seq_length
                ),
                temperature=temperature,
                show_progress=show_tokens,
            )
            if chat_mode
            else runner.text_completion(
                prompt=prompt,
                temperature=temperature,
                echo=True,
            )
        )
        if show_tokens:
            print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")


def main() -> None:
    execute_runner(EagerLlamaRunner)


if __name__ == "__main__":
    main()  # pragma: no cover
