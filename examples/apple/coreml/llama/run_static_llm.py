# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run script for static attention LLM models exported with export_static_llm_coreml.py.

Usage:
    python run_static_llm.py \
        --model llama1b_static.pte \
        --params $HOME/models/llama1b/params.json \
        --tokenizer $HOME/models/llama1b/tokenizer.model \
        --prompt "Once upon a time" \
        --max_new_tokens 100
"""

import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.utils._pytree as pytree

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.runner.generation import next_token
from executorch.examples.models.llama.static_attention import StaticAttentionIOManager
from executorch.runtime import Runtime
from pytorch_tokenizers import get_tokenizer


def get_stop_tokens(tokenizer) -> List[int]:
    """Get stop tokens from tokenizer, falling back to eos_id if not available."""
    if hasattr(tokenizer, "stop_tokens"):
        return tokenizer.stop_tokens
    return [tokenizer.eos_id]


def create_pte_wrapper(
    method,
    k_cache_keys: List[str],
    v_cache_keys: List[str],
):
    """
    Create a wrapper function that adapts PTE execution to the interface
    expected by StaticAttentionIOManager.

    The wrapper:
    - Takes (tokens, options_dict) like the eager model
    - Flattens inputs using pytree
    - Executes the PTE method
    - Reconstructs outputs to match eager model format: (logits, {"out_cache_state": (k_dict, v_dict)})
    """

    def wrapper(
        tokens: torch.Tensor, options: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Build the same input structure as during export
        inputs = (tokens, options)

        # Flatten using pytree (same order as torch.export)
        flat_inputs, _ = pytree.tree_flatten(inputs)

        # Execute PTE model
        outputs = method.execute(flat_inputs)

        # First output is logits
        logits = outputs[0]

        # Remaining outputs are k_cache updates then v_cache updates
        num_layers = len(k_cache_keys)
        k_updates = outputs[1 : 1 + num_layers]
        v_updates = outputs[1 + num_layers : 1 + 2 * num_layers]

        # Reconstruct the output cache state dicts
        k_cache_dict = dict(zip(k_cache_keys, k_updates))
        v_cache_dict = dict(zip(v_cache_keys, v_updates))

        attn_updates = {"out_cache_state": (k_cache_dict, v_cache_dict)}

        return logits, attn_updates

    return wrapper


def main():
    parser = argparse.ArgumentParser(description="Run static attention Llama model")

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
        help="Input sequence length (must match export)",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        default=992,
        help="Cache length (must match export: max_context_len - input_len)",
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
    parser.add_argument(
        "--method",
        type=str,
        default="forward",
        help="Method name to run (default: 'forward', use 'base' or 'lora' for multimethod models)",
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_config)
    stop_tokens = get_stop_tokens(tokenizer)

    # Load model params
    with open(args.params, "r") as f:
        params = json.loads(f.read())

    # Create model args
    model_args = ModelArgs(
        max_context_len=args.cache_len + args.input_len,
        generate_full_logits=True,
        **params,
    )
    model_args.attention_type = "static_mha"

    print(f"Model config: {model_args.n_layers} layers, dim={model_args.dim}")
    print(f"Input length: {args.input_len}, Cache length: {args.cache_len}")

    # Create StaticAttentionIOManager
    mgr = StaticAttentionIOManager(
        model_args,
        input_len=args.input_len,
        cache_lens=args.cache_len,
        batch_size=1,
        dtype=torch.float16,
        style="smart_mask",  # Use smart_mask to match C++ StaticTransformerRunner
        mask_val=float("-inf"),
    )

    # Load PTE model
    print(f"Loading model from {args.model}...")
    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    print(f"Loading method '{args.method}'...")
    method = program.load_method(args.method)

    metadata = method.metadata
    print(
        f"Method metadata: num_inputs={metadata.num_inputs()}, num_outputs={metadata.num_outputs()}"
    )

    # Get cache keys in insertion order (NOT sorted alphabetically!)
    # Pytree preserves dict insertion order in Python 3.7+
    # The caches are created in layer order (0, 1, 2, ..., n_layers-1)
    k_cache_keys = list(mgr.k_caches.keys())
    v_cache_keys = list(mgr.v_caches.keys())

    # Create wrapper function that adapts PTE to eager interface
    model_fn = create_pte_wrapper(method, k_cache_keys, v_cache_keys)

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print("-" * 50)

    # Reset manager
    mgr.reset()

    # Prefill using StaticAttentionIOManager.prefill
    print("Prefilling...", end=" ", flush=True)
    start_time = time.time()
    logits = mgr.prefill(model_fn, prompt_tokens)
    prefill_time = time.time() - start_time
    print(f"done in {prefill_time:.2f}s")

    # Get first token from prefill logits
    first_token = next_token(logits[:, -1, :], args.temperature, args.top_p)

    # Decode using StaticAttentionIOManager.decode or lookahead_decode
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
        # Use standard autoregressive decoding
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


if __name__ == "__main__":
    main()
