#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run exported Qwen 3.5 MoE model using ExecuTorch pybindings.

Companion to export.py --backend mlx. Supports both real model inference
(with HuggingFace tokenizer) and fake-weight validation (random tokens).

Usage:
    # Run with real tokenizer:
    python -m executorch.examples.models.qwen3_5_moe.run \
        --pte qwen35_moe_mlx.pte \
        --tokenizer Qwen/Qwen3.5-35B-A3B \
        --prompt "Hello, world!"

    # Run with random tokens (fake weights, no tokenizer needed):
    python -m executorch.examples.models.qwen3_5_moe.run \
        --pte qwen35_moe_mlx.pte \
        --prompt-len 8 \
        --max-new-tokens 20
"""

import argparse
import logging
import time

import torch
from executorch.runtime import Runtime, Verification

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def run_inference(
    pte_path: str,
    tokenizer_id: str = None,
    prompt: str = None,
    prompt_len: int = 4,
    max_new_tokens: int = 10,
    vocab_size: int = 248320,
    temperature: float = 0.0,
) -> None:
    """Run inference on the exported Qwen 3.5 MoE model."""
    logger.info(f"Loading model from {pte_path}...")
    et_runtime = Runtime.get()
    program = et_runtime.load_program(pte_path, verification=Verification.Minimal)
    forward = program.load_method("forward")
    logger.info("Model loaded successfully")

    # Read vocab size from model metadata if available
    try:
        meta_method = program.load_method("get_vocab_size")
        result = meta_method.execute([])
        model_vocab_size = result[0] if isinstance(result[0], int) else result[0].item()
        logger.info(f"Vocab size from model metadata: {model_vocab_size}")
        vocab_size = model_vocab_size
    except Exception:
        logger.info(f"No vocab size in metadata, using default: {vocab_size}")

    # Tokenize or generate random tokens
    tokenizer = None
    if tokenizer_id and prompt:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

        # Apply chat template so the model sees proper conversation boundaries
        # and knows when to stop generating (at <|im_end|>)
        messages = [{"role": "user", "content": prompt}]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(templated, return_tensors="pt").to(torch.long)
        prompt_len = input_ids.shape[1]
        logger.info(f"Prompt: {prompt!r} ({prompt_len} tokens)")

        # Collect stop token ids (EOS + any end-of-turn markers)
        stop_token_ids = set()
        if tokenizer.eos_token_id is not None:
            stop_token_ids.add(tokenizer.eos_token_id)
        # <|im_end|> is the stop token for Qwen chat models
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            stop_token_ids.add(im_end_id)
    else:
        stop_token_ids = set()
        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
        logger.info(f"Random prompt ({prompt_len} tokens): {input_ids[0].tolist()}")

    # --- Warmup run (JIT compile Metal kernels, warm GPU caches) ---
    logger.info("Running warmup...")
    warmup_tokens = torch.zeros((1, 1), dtype=torch.long)
    warmup_pos = torch.tensor([0], dtype=torch.long)
    forward.execute([warmup_tokens, warmup_pos])

    # --- Prefill ---
    logger.info(f"Running prefill ({prompt_len} tokens)...")
    start_time = time.time()

    input_pos = torch.arange(prompt_len, dtype=torch.long)
    outputs = forward.execute([input_ids, input_pos])
    logits = outputs[0]

    prefill_time = time.time() - start_time
    logger.info(
        f"Prefill: {prefill_time:.3f}s " f"({prompt_len / prefill_time:.1f} tokens/sec)"
    )

    # First generated token
    next_token_logits = logits[0, -1, :]
    if temperature > 0:
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
    else:
        next_token = torch.argmax(next_token_logits).item()
    generated_tokens = [next_token]

    # --- Decode ---
    logger.info(f"Generating up to {max_new_tokens} tokens...")
    decode_start = time.time()
    t_execute = 0
    t_prep = 0
    t_post = 0

    for _i in range(max_new_tokens - 1):
        t0 = time.time()
        pos = prompt_len + len(generated_tokens) - 1
        input_pos = torch.tensor([pos], dtype=torch.long)
        token_input = torch.tensor([[next_token]], dtype=torch.long)
        t1 = time.time()
        t_prep += t1 - t0

        outputs = forward.execute([token_input, input_pos])
        logits = outputs[0]
        t2 = time.time()
        t_execute += t2 - t1

        next_token_logits = logits[0, -1, :]
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()
        generated_tokens.append(next_token)
        t3 = time.time()
        t_post += t3 - t2

        # Stop on EOS / end-of-turn token
        if next_token in stop_token_ids:
            break

    decode_time = time.time() - decode_start
    num_generated = len(generated_tokens)
    tokens_per_sec = num_generated / decode_time if decode_time > 0 else 0

    # Print decode timing breakdown
    n_decode = num_generated - 1  # exclude first token (from prefill)
    if n_decode > 0:
        print(f"\nDecode timing breakdown ({n_decode} steps):")
        print(
            f"  Prep (tensor creation):  {t_prep*1000:.1f}ms total, {t_prep/n_decode*1000:.2f}ms/step"
        )
        print(
            f"  Execute (forward.execute): {t_execute*1000:.1f}ms total, {t_execute/n_decode*1000:.2f}ms/step"
        )
        print(
            f"  Post (argmax/sample):    {t_post*1000:.1f}ms total, {t_post/n_decode*1000:.2f}ms/step"
        )

    # Print results
    print(f"\nPrefill: {prefill_time:.3f}s ({prompt_len / prefill_time:.1f} tok/s)")
    print(
        f"Decode:  {decode_time:.3f}s "
        f"({num_generated} tokens, {tokens_per_sec:.1f} tok/s)"
    )

    if tokenizer:
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    else:
        print(f"\nGenerated token ids: {generated_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Run exported Qwen 3.5 MoE model")
    parser.add_argument(
        "--pte",
        type=str,
        required=True,
        help="Path to the .pte file from export.py --backend mlx",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer ID (e.g. Qwen/Qwen3.5-35B-A3B)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation (requires --tokenizer)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=4,
        help="Number of random tokens for the prompt (when no --prompt given)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=248320,
        help="Vocab size for random token generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )

    args = parser.parse_args()

    if args.prompt and not args.tokenizer:
        parser.error("--prompt requires --tokenizer")

    run_inference(
        pte_path=args.pte,
        tokenizer_id=args.tokenizer,
        prompt=args.prompt,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        vocab_size=args.vocab_size,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
