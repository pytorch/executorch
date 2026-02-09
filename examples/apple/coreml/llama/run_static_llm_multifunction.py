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
    decode_method,
    prefill_method,
    mgr: "StaticAttentionIOManager",
    prefill_seq_len: int,
    prefill_mask: Dict[str, torch.Tensor],
):
    """
    Create a wrapper function that adapts PTE execution to the interface
    expected by StaticAttentionIOManager.

    This multifunction version selects between prefill and decode methods
    based on the input sequence length. Both methods use the SAME cache_len,
    so the cache buffer is shared directly without any slicing or copying.

    The wrapper:
    - Takes (tokens, options_dict) like the eager model
    - Selects prefill or decode method based on token count
    - Uses the same cache buffer for both methods (no slicing needed)
    - Flattens inputs using pytree
    - Executes the appropriate PTE method
    - Reconstructs outputs to match eager model format: (logits, {"out_cache_state": (k_dict, v_dict)})

    Args:
        decode_method: The PTE method for decode (seqlen=1)
        prefill_method: The PTE method for prefill (seqlen=input_len)
        mgr: StaticAttentionIOManager with caches sized for shared cache_len
        prefill_seq_len: The sequence length for prefill
        prefill_mask: Pre-computed mask tensor for prefill method
    """

    k_cache_keys = list(mgr.k_caches.keys())
    v_cache_keys = list(mgr.v_caches.keys())

    timing_stats = {
        "flatten_time": 0.0,
        "execute_time": 0.0,
        "reconstruct_time": 0.0,
        "detection_time": 0.0,
        "options_build_time": 0.0,
        "call_count": 0,
    }

    def wrapper(
        tokens: torch.Tensor, options: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        import time as time_module

        timing_stats["call_count"] += 1

        t0 = time_module.perf_counter()

        # Detect actual sequence length.
        # StaticAttentionIOManager._run_once pads tokens with zeros on the right.
        # For decode (1 actual token), positions 1+ are all zeros.
        padded_seq_len = tokens.shape[1]
        if padded_seq_len > 1 and (tokens[0, 1:] == 0).all():
            actual_seq_len = 1
        else:
            actual_seq_len = padded_seq_len

        is_prefill = actual_seq_len == prefill_seq_len

        t1 = time_module.perf_counter()
        timing_stats["detection_time"] += t1 - t0

        t0 = time_module.perf_counter()

        # Get the input cache state from options
        in_k_caches, in_v_caches = options["in_cache_state"]

        # Both prefill and decode use the same cache_len, so no slicing needed!
        # Just select the appropriate method and mask.
        if is_prefill:
            method = prefill_method
            adapted_mask = prefill_mask
        else:
            method = decode_method
            adapted_mask = mgr.masks

        adapted_options = {
            "masks": adapted_mask,
            "freqs_cos_override": options["freqs_cos_override"],
            "freqs_sin_override": options["freqs_sin_override"],
            "in_cache_state": (in_k_caches, in_v_caches),  # Same cache for both!
        }

        if "last_valid_token_pos" in options:
            adapted_options["last_valid_token_pos"] = options["last_valid_token_pos"]

        inputs = (tokens, adapted_options)

        t1 = time_module.perf_counter()
        timing_stats["options_build_time"] += t1 - t0

        t0 = time_module.perf_counter()
        flat_inputs, _ = pytree.tree_flatten(inputs)
        t1 = time_module.perf_counter()
        timing_stats["flatten_time"] += t1 - t0

        t0 = time_module.perf_counter()
        outputs = method.execute(flat_inputs)
        t1 = time_module.perf_counter()
        timing_stats["execute_time"] += t1 - t0

        t0 = time_module.perf_counter()

        logits = outputs[0]

        num_layers = len(k_cache_keys)
        k_updates = outputs[1 : 1 + num_layers]
        v_updates = outputs[1 + num_layers : 1 + 2 * num_layers]

        k_cache_dict = dict(zip(k_cache_keys, k_updates))
        v_cache_dict = dict(zip(v_cache_keys, v_updates))

        attn_updates = {"out_cache_state": (k_cache_dict, v_cache_dict)}

        t1 = time_module.perf_counter()
        timing_stats["reconstruct_time"] += t1 - t0

        return logits, attn_updates

    def print_timing_stats():
        n = timing_stats["call_count"]
        if n > 0:
            print(f"\n=== Wrapper Timing Stats ({n} calls) ===")
            print(
                f"  Detection time:   {timing_stats['detection_time']*1000:.2f}ms total, {timing_stats['detection_time']/n*1000:.4f}ms avg"
            )
            print(
                f"  Options build:    {timing_stats['options_build_time']*1000:.2f}ms total, {timing_stats['options_build_time']/n*1000:.4f}ms avg"
            )
            print(
                f"  Flatten time:     {timing_stats['flatten_time']*1000:.2f}ms total, {timing_stats['flatten_time']/n*1000:.4f}ms avg"
            )
            print(
                f"  Execute time:     {timing_stats['execute_time']*1000:.2f}ms total, {timing_stats['execute_time']/n*1000:.3f}ms avg"
            )
            print(
                f"  Reconstruct time: {timing_stats['reconstruct_time']*1000:.2f}ms total, {timing_stats['reconstruct_time']/n*1000:.4f}ms avg"
            )
            total = (
                timing_stats["detection_time"]
                + timing_stats["options_build_time"]
                + timing_stats["flatten_time"]
                + timing_stats["execute_time"]
                + timing_stats["reconstruct_time"]
            )
            print(
                f"  Total wrapper:    {total*1000:.2f}ms total, {total/n*1000:.3f}ms avg"
            )
            print(
                f"  Execute is {timing_stats['execute_time']/total*100:.1f}% of wrapper time"
            )
            expected_tps = 1000 / (timing_stats["execute_time"] / n * 1000)
            print(f"  Expected tok/s from execute alone: {expected_tps:.1f}")

    wrapper.print_timing_stats = print_timing_stats
    wrapper.timing_stats = timing_stats

    return wrapper


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

    # Create decode manager (input_len=1) - used for decode phase
    mgr = StaticAttentionIOManager(
        model_args,
        input_len=decode_input_len,
        cache_lens=shared_cache_len,
        batch_size=1,
        dtype=torch.float16,
        style="smart_mask",
        mask_val=float("-inf"),
    )

    # Create prefill manager (input_len=64) with the SAME cache_len.
    # Since both use the same cache_len, we can share the cache buffer directly.
    prefill_mgr = StaticAttentionIOManager(
        model_args,
        input_len=prefill_input_len,
        cache_lens=shared_cache_len,  # Same cache_len as decode!
        batch_size=1,
        dtype=torch.float16,
        style="smart_mask",
        mask_val=float("-inf"),
    )

    # Share cache buffers: point prefill_mgr's caches to mgr's caches.
    # No copying needed since both managers use the same cache_len!
    prefill_mgr.k_caches = mgr.k_caches
    prefill_mgr.v_caches = mgr.v_caches

    prefill_mask = prefill_mgr.masks

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
