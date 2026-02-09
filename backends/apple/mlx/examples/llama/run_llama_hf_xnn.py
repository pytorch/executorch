#!/usr/bin/env python3
# @nocommit
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run exported Llama model (XNNPACK delegate) using optimum-executorch runner.

This script runs models exported using export_llama_hf_xnn.py.

Usage:
    python -m executorch.backends.apple.mlx.examples.llama.run_llama_hf_xnn \
        --pte xnn_llama_hf_int4_fp32/model.pte \
        --model-id unsloth/Llama-3.2-1B-Instruct \
        --prompt "Tell me a story"
"""

import argparse
import logging

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def run_inference(
    pte_path: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Run inference on the exported HuggingFace model with XNNPACK delegate."""
    from optimum.executorch.modeling import ExecuTorchModelForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from HuggingFace: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading model from {pte_path}...")
    model = ExecuTorchModelForCausalLM.from_pretrained(pte_path)

    logger.info(f"Encoding prompt: {prompt!r}")
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    generated_text = model.text_generation(
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        echo=False,
        max_seq_len=len(tokenizer.encode(formatted_prompt)) + max_new_tokens,
    )

    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Run exported HuggingFace Llama model (XNNPACK delegate)"
    )
    parser.add_argument(
        "--pte",
        type=str,
        default="xnn_llama_hf_int4_bf16/model.pte",
        help="Path to the .pte file",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID (used to load tokenizer)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()

    generated_text = run_inference(
        pte_path=args.pte,
        model_id=args.model_id,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n" + "=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
