#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run exported Llama model (from HuggingFace) using ExecuTorch pybindings.

This script runs models exported using export_llama_hf.py. It loads the tokenizer
directly from HuggingFace using the same model ID used during export.

Usage:
    python -m executorch.backends.apple.mlx.examples.llama.run_llama_hf \
        --pte llama_hf.pte \
        --model-id unsloth/Llama-3.2-1B-Instruct \
        --prompt "Hello, world!"
"""

import argparse
import logging
import time

import torch
from executorch.runtime import Runtime, Verification
from transformers import AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def run_inference(
    pte_path: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Run inference on the exported HuggingFace model."""
    logger.info(f"Loading tokenizer from HuggingFace: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading model from {pte_path}...")
    et_runtime = Runtime.get()
    program = et_runtime.load_program(pte_path, verification=Verification.Minimal)
    forward = program.load_method("forward")

    logger.info(f"Encoding prompt: {prompt!r}")
    # Apply chat template for instruct models
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    logger.info(f"Input shape: {input_ids.shape}")

    generated_tokens = input_ids[0].tolist()
    seq_len = input_ids.shape[1]

    # Prefill: process all input tokens at once
    logger.info("Running prefill...")
    start_time = time.time()

    # cache_position must match the sequence length of input_ids
    # For prefill with N tokens, cache_position = [0, 1, 2, ..., N-1]
    cache_position = torch.arange(seq_len, dtype=torch.long)
    outputs = forward.execute([input_ids, cache_position])
    logits = outputs[0]

    prefill_time = time.time() - start_time
    logger.info(f"Prefill time: {prefill_time:.3f}s")
    logger.info(f"Output logits shape: {logits.shape}")

    # Get the next token from the last position
    next_token_logits = logits[0, -1, :]
    next_token = torch.argmax(next_token_logits).item()
    generated_tokens.append(next_token)

    # Decode: generate tokens one at a time
    logger.info(f"Generating {max_new_tokens} tokens...")
    decode_start = time.time()

    for i in range(max_new_tokens - 1):
        # Position for the token we're about to process
        # After prefill of N tokens and generating 1 token, generated_tokens has N+1 items
        # The token we're processing (next_token) is at position len(generated_tokens)-1
        pos = len(generated_tokens) - 1
        input_pos = torch.tensor([pos], dtype=torch.long)
        # Input is just the last generated token
        token_input = torch.tensor([[next_token]], dtype=torch.long)

        outputs = forward.execute([token_input, input_pos])
        logits = outputs[0]

        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        generated_tokens.append(next_token)

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            logger.info(f"EOS token reached at position {i + 1}")
            break

    decode_time = time.time() - decode_start
    tokens_per_sec = (len(generated_tokens) - input_ids.shape[1]) / decode_time
    logger.info(f"Decode time: {decode_time:.3f}s ({tokens_per_sec:.1f} tokens/sec)")

    # Decode only the newly generated tokens (not the input prompt)
    new_tokens = generated_tokens[input_ids.shape[1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Run exported HuggingFace Llama model")
    parser.add_argument(
        "--pte",
        type=str,
        default="llama_hf.pte",
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
