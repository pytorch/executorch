#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run exported HuggingFace LLM using ExecuTorch pybindings.

This script runs models exported using export_llm_hf.py. It loads the tokenizer
or processor directly from HuggingFace using the same model ID used during
export.

Usage:
    python -m executorch.backends.mlx.examples.llm.run_llm_hf \
        --pte llama_hf.pte \
        --model-id unsloth/Llama-3.2-1B-Instruct \
        --prompt "Hello, world!"
"""

import argparse
import logging
import time

import torch

from executorch.backends.mlx.examples.llm.runtime_meta import (
    apply_chat_template,
    chunked_prefill,
    get_eos_token_id,
    load_text_processor,
    read_model_limits,
)
from executorch.runtime import Runtime, Verification

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def _get_max_input_seq_len(program) -> int:
    """Inspect the .pte program metadata to determine the max input_ids seq len.

    Fallback for .pte files exported before get_prefill_chunk_size existed.
    Returns the static seq-len dimension of the first input tensor (input_ids).
    """
    meta = program.metadata("forward")
    input_ids_info = meta.input_tensor_meta(0)
    sizes = input_ids_info.sizes()
    # sizes is (batch, seq_len)
    return sizes[1] if len(sizes) >= 2 else 1


def run_inference(
    pte_path: str,
    model_id: str,
    revision: str | None,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Run inference on the exported HuggingFace model."""
    text_processor = load_text_processor(model_id, revision)

    logger.info(f"Loading model from {pte_path}...")
    et_runtime = Runtime.get()
    program = et_runtime.load_program(pte_path, verification=Verification.Minimal)

    max_ctx_len, prefill_chunk_size = read_model_limits(program)
    if prefill_chunk_size is None:
        prefill_chunk_size = _get_max_input_seq_len(program)
    logger.info(
        f"Model limits: max_ctx_len={max_ctx_len}, "
        f"prefill_chunk_size={prefill_chunk_size}"
    )

    forward = program.load_method("forward")

    logger.info(f"Encoding prompt: {prompt!r}")
    input_ids = apply_chat_template(text_processor, prompt)
    logger.info(f"Input shape: {input_ids.shape}")

    generated_tokens = input_ids[0].tolist()
    seq_len = input_ids.shape[1]
    eos_token_id = get_eos_token_id(text_processor)

    start_time = time.time()

    logger.info(f"Running prefill ({seq_len} tokens, chunk {prefill_chunk_size})...")
    outputs = chunked_prefill(forward, input_ids, prefill_chunk_size)
    logits = outputs[0]

    prefill_time = time.time() - start_time
    logger.info(
        f"Prefill time: {prefill_time:.3f}s "
        f"({seq_len / prefill_time:.1f} tokens/sec)"
    )

    # Get the next token from the last position
    next_token_logits = logits[0, -1, :]
    next_token = torch.argmax(next_token_logits).item()
    generated_tokens.append(next_token)

    # Decode: generate tokens one at a time
    logger.info(f"Generating up to {max_new_tokens} tokens...")
    decode_start = time.time()

    for i in range(max_new_tokens - 1):
        pos = len(generated_tokens) - 1
        cache_position = torch.tensor([pos], dtype=torch.long)
        token_input = torch.tensor([[next_token]], dtype=torch.long)

        outputs = forward.execute([token_input, cache_position])
        logits = outputs[0]

        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        generated_tokens.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            logger.info(f"EOS token reached at position {i + 1}")
            break

    decode_time = time.time() - decode_start
    num_generated = len(generated_tokens) - seq_len
    tokens_per_sec = num_generated / decode_time if decode_time > 0 else 0

    print(f"\nPrefill time: {prefill_time:.3f}s ({seq_len / prefill_time:.1f} tok/s)")
    print(
        f"Decode time:  {decode_time:.3f}s ({num_generated} tokens, {tokens_per_sec:.1f} tok/s)"
    )

    # Decode only the newly generated tokens (not the input prompt)
    new_tokens = generated_tokens[seq_len:]
    generated_text = text_processor.decode(new_tokens, skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Run exported HuggingFace LLM")
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
        help="HuggingFace model ID (used to load tokenizer or processor)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HuggingFace model revision/commit to pin",
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
        revision=args.revision,
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
