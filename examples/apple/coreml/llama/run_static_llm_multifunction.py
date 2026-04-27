# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run script for multifunction static attention Llama models exported with coreml_static_llama.py.

This script tests multifunction CoreML models that have separate "forward" (decode) and
"prefill" methods sharing weights.

Usage:
    python run_static_llm_multifunction.py \
        --model $HOME/Desktop/multifunction_test.pte \
        --params $HOME/models/llama1b/params.json \
        --tokenizer $HOME/models/llama1b/tokenizer.model \
        --prompt "Once upon a time" \
        --max_new_tokens 100
"""

import argparse
import json
import time
from typing import List

import torch

from executorch.examples.apple.coreml.llama.utils import (
    create_pte_wrapper,
    setup_multifunction_managers,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.runner.generation import next_token
from executorch.runtime import Runtime
from pytorch_tokenizers import get_tokenizer


def get_stop_tokens(tokenizer) -> List[int]:
    """Get stop tokens from tokenizer, falling back to eos_id if not available."""
    if hasattr(tokenizer, "stop_tokens"):
        return tokenizer.stop_tokens
    return [tokenizer.eos_id]


def main():
    parser = argparse.ArgumentParser(
        description="Run multifunction static attention Llama model"
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to exported .pte model",
    )
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to params.json",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--tokenizer_config",
        type=str,
        default=None,
        help="Path to tokenizer config (required for HuggingFace tokenizers)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
        help="Input prompt",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold",
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=32,
        help="Input sequence length for prefill (must match export)",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=1024,
        help="Maximum context length (must match export)",
    )
    parser.add_argument(
        "--lookahead",
        action="store_true",
        help="Enable lookahead (speculative) decoding",
    )
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=5,
        help="N-gram size for lookahead decoding",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=4,
        help="Window size for lookahead decoding",
    )
    parser.add_argument(
        "--n_verifications",
        type=int,
        default=4,
        help="Number of verification branches for lookahead decoding",
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_config)
    stop_tokens = get_stop_tokens(tokenizer)

    # Load model params
    with open(args.params, "r") as f:
        params = json.loads(f.read())

    # Create model args
    # Multifunction models use generate_full_logits=False (only last token logits)
    model_args = ModelArgs(
        max_context_len=args.max_context_len,
        generate_full_logits=False,
        **params,
    )
    model_args.attention_type = "static_mha"

    print(f"Model config: {model_args.n_layers} layers, dim={model_args.dim}")
    print(f"Max context length: {args.max_context_len}, Input length: {args.input_len}")

    # Both prefill and decode use the same cache_len (decode's cache size).
    # This allows them to share the same cache buffer without any copying.
    # The export script exports both methods with cache_len = max_context_len - 1.
    prefill_input_len = args.input_len  # e.g., 64
    decode_input_len = 1
    shared_cache_len = args.max_context_len - decode_input_len  # e.g., 1023

    print(f"Prefill: input_len={prefill_input_len}, cache_len={shared_cache_len}")
    print(f"Decode: input_len={decode_input_len}, cache_len={shared_cache_len}")

    # Create managers with shared cache buffers
    mgr, prefill_mgr, prefill_mask = setup_multifunction_managers(
        model_args,
        prefill_input_len,
        decode_input_len,
        shared_cache_len,
        dtype=torch.float16,
        mask_val=float("-inf"),
    )

    # Load PTE model with multifunction support
    print(f"Loading multifunction model from {args.model}...")
    runtime = Runtime.get()
    program = runtime.load_program(args.model)

    # List available methods
    method_names = program.method_names
    # Separate executable methods from constant methods (metadata)
    executable_methods = {"forward", "prefill"}
    actual_methods = executable_methods & method_names
    constant_methods = method_names - executable_methods
    print(f"Executable methods: {actual_methods}")
    print(f"Metadata methods: {constant_methods}")

    # Check for expected multifunction methods
    if "forward" not in method_names or "prefill" not in method_names:
        print(
            f"Warning: Expected 'forward' and 'prefill' methods, found: {method_names}"
        )
        print("Falling back to single 'forward' method...")
        decode_method = program.load_method("forward")
        prefill_method = decode_method
    else:
        # Load both methods
        print("Loading 'forward' (decode) method...")
        decode_method = program.load_method("forward")
        print("Loading 'prefill' method...")
        prefill_method = program.load_method("prefill")

    decode_metadata = decode_method.metadata
    print(
        f"Decode method metadata: num_inputs={decode_metadata.num_inputs()}, num_outputs={decode_metadata.num_outputs()}"
    )

    prefill_metadata = prefill_method.metadata
    print(
        f"Prefill method metadata: num_inputs={prefill_metadata.num_inputs()}, num_outputs={prefill_metadata.num_outputs()}"
    )

    # Get cache keys in insertion order (NOT sorted alphabetically!)
    # Pytree preserves dict insertion order in Python 3.7+
    # The caches are created in layer order (0, 1, 2, ..., n_layers-1)
    # Note: cache keys are obtained inside create_pte_wrapper

    # Create wrapper function that adapts PTE to eager interface
    # This wrapper will select between prefill and decode based on seq_len
    model_fn = create_pte_wrapper(
        decode_method,
        prefill_method,
        mgr,
        prefill_input_len,
        prefill_mask,
    )

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print("-" * 50)

    # Reset both managers
    mgr.reset()
    prefill_mgr.reset()

    # Prefill using prefill_mgr which has the correct input_len and mask shapes
    # This will call model_fn with seq_len=input_len, which selects the prefill method
    print("Prefilling (using 'prefill' method)...", end=" ", flush=True)
    start_time = time.time()

    # Use prefill_mgr for prefill since it has the correct input_len=64 and mask shapes
    logits = prefill_mgr.prefill(model_fn, prompt_tokens)

    prefill_time = time.time() - start_time
    print(f"done in {prefill_time:.2f}s")

    # Get first token from prefill logits
    # With generate_full_logits=False, logits is 2D [batch, vocab]
    # With generate_full_logits=True, logits is 3D [batch, seq_len, vocab]
    if logits.dim() == 2:
        first_token = next_token(logits, args.temperature, args.top_p)
    else:
        first_token = next_token(logits[:, -1, :], args.temperature, args.top_p)

    # No cache copying needed! prefill_mgr and mgr share the same cache buffers
    # because both use the same cache_len (shared_cache_len).

    # Sync position from prefill manager to decode manager
    mgr.pos = prefill_mgr.pos

    # Update decode manager's masks to reflect current position after prefill
    for mask in mgr._masks.values():
        mask.reset()
        mask.unmask(mgr.pos)

    # Decode using mgr.decode() which will call model_fn
    # The wrapper will detect seq_len=1 and route to decode method
    print(f"\n{args.prompt}", end="", flush=True)
    print(tokenizer.decode_token(first_token), end="", flush=True)

    decode_start = time.time()

    if args.lookahead:
        # Use lookahead (speculative) decoding
        print(
            f"\n[Using lookahead decoding: ngram={args.ngram_size}, window={args.window_size}, verifications={args.n_verifications}]"
        )
        generated_tokens = mgr.lookahead_decode(
            model_fn,
            first_token,
            n=args.max_new_tokens - 1,  # -1 because first_token counts
            ngram_size=args.ngram_size,
            window_size=args.window_size,
            n_verifications=args.n_verifications,
            stop_tokens=stop_tokens,
        )
    else:
        # Use standard autoregressive decoding (uses 'forward' method)
        print("\n[Using 'forward' (decode) method for generation]")
        generated_tokens = mgr.decode(
            model_fn,
            first_token,
            n=args.max_new_tokens - 1,  # -1 because first_token counts
            stop_tokens=stop_tokens,
        )

    # Print generated tokens (skip first as it's the init_token we already printed)
    for token in generated_tokens[1:]:
        if token in stop_tokens:
            break
        print(tokenizer.decode_token(token), end="", flush=True)

    decode_time = time.time() - decode_start
    total_generated = len(generated_tokens)
    tokens_per_sec = total_generated / decode_time if decode_time > 0 else 0

    print("\n" + "-" * 50)
    print(f"Prefill: {len(prompt_tokens)} tokens in {prefill_time:.2f}s")
    print(
        f"Decode: {total_generated} tokens in {decode_time:.2f}s ({tokens_per_sec:.2f} tok/s)"
    )

    # Print detailed timing breakdown
    model_fn.print_timing_stats()

    print("\nMultifunction model test completed successfully!")


if __name__ == "__main__":
    main()
