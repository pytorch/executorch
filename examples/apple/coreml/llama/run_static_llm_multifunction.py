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

import sentencepiece as spm
import torch
import torch.utils._pytree as pytree

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.runner.generation import next_token
from executorch.examples.models.llama.static_attention import StaticAttentionIOManager
from executorch.runtime import Runtime


class Tokenizer:
    """Wrapper to support both SentencePiece and Tiktoken tokenizers."""

    def __init__(self, model_path: str):
        try:
            print("Trying to load sentencepiece")
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            self.tokenizer = sp
            self._is_sentencepiece = True
        except Exception:
            print("Trying to load tiktoken")
            from executorch.examples.models.llama.tokenizer import tiktoken

            self.tokenizer = tiktoken.Tokenizer(model_path)
            self._is_sentencepiece = False

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        if self._is_sentencepiece:
            bos_string = "<s>" if bos else ""
            eos_string = "</s>" if eos else ""
            return self.tokenizer.encode(f"{bos_string}{text}{eos_string}")
        return self.tokenizer.encode(text, bos=bos, eos=eos)

    def decode(self, tokens: List[int]) -> str:
        if self._is_sentencepiece:
            return self.tokenizer.decode(tokens)
        return self.tokenizer.decode(tokens)

    def decode_token(self, token: int) -> str:
        if self._is_sentencepiece:
            return self.tokenizer.decode([token])
        try:
            return self.tokenizer.decode_token(token)
        except UnicodeDecodeError:
            return f"<{token}>"

    @property
    def stop_tokens(self) -> List[int]:
        if self._is_sentencepiece:
            return [self.tokenizer.eos_id()]
        return self.tokenizer.stop_tokens


def create_pte_wrapper(
    decode_method,
    prefill_method,
    prefill_mgr: "StaticAttentionIOManager",
    decode_mgr: "StaticAttentionIOManager",
    prefill_seq_len: int,
    prefill_cache_len: int,
    decode_cache_len: int,
):
    """
    Create a wrapper function that adapts PTE execution to the interface
    expected by StaticAttentionIOManager.

    This multifunction version selects between prefill and decode methods
    based on the input sequence length. It also uses the appropriate
    StaticAttentionIOManager for each method since they have different
    cache lengths.

    The wrapper:
    - Takes (tokens, options_dict) like the eager model
    - Selects prefill or decode method based on token count
    - Adapts the options to use the correct manager's cache structure
    - Flattens inputs using pytree
    - Executes the appropriate PTE method
    - Reconstructs outputs to match eager model format: (logits, {"out_cache_state": (k_dict, v_dict)})
    """

    # Get cache keys from the prefill manager (same structure as decode)
    k_cache_keys = list(prefill_mgr.k_caches.keys())
    v_cache_keys = list(prefill_mgr.v_caches.keys())

    # Timing accumulators
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

        # TIME: Detection logic
        t0 = time_module.perf_counter()

        # Detect actual sequence length BEFORE padding.
        # StaticAttentionIOManager._run_once pads tokens with zeros on the right:
        #   tokens = F.pad(tokens, (0, self.input_len - n_tokens))
        # So for decode (1 actual token), positions 1+ are all zeros.
        # For prefill (32 actual tokens), positions have real token values.
        padded_seq_len = tokens.shape[1]
        if padded_seq_len > 1 and (tokens[0, 1:] == 0).all():
            # Single token padded to prefill_seq_len - this is decode
            actual_seq_len = 1
        else:
            actual_seq_len = padded_seq_len

        # Select method and manager based on actual (pre-padding) sequence length
        if actual_seq_len == prefill_seq_len:
            method = prefill_method
            mgr = prefill_mgr
            # Use tokens and freqs as-is for prefill
            adapted_tokens = tokens
            adapted_freqs_cos = options["freqs_cos_override"]
            adapted_freqs_sin = options["freqs_sin_override"]
            # Use cache state as-is (prefill manager's cache size matches prefill method)
            adapted_cache_state = options["in_cache_state"]
        else:
            method = decode_method
            mgr = decode_mgr
            # For decode, use tokens and freqs as-is (decode_mgr passes correct shapes)
            # Note: decode_mgr.input_len=1, so tokens are NOT padded, just (1, 1)
            adapted_tokens = tokens
            adapted_freqs_cos = options["freqs_cos_override"]
            adapted_freqs_sin = options["freqs_sin_override"]
            # Use cache state as-is (decode_mgr's cache size matches decode method)
            adapted_cache_state = options["in_cache_state"]

        t1 = time_module.perf_counter()
        timing_stats["detection_time"] += t1 - t0

        # TIME: Build options
        t0 = time_module.perf_counter()

        # Build options with the correct mask and freqs for this method
        adapted_options = {
            "masks": mgr.masks,  # Use correct manager's mask (has right shape)
            "freqs_cos_override": adapted_freqs_cos,
            "freqs_sin_override": adapted_freqs_sin,
            "in_cache_state": adapted_cache_state,
        }

        # Pass through last_valid_token_pos if present (needed for generate_full_logits=False)
        if "last_valid_token_pos" in options:
            adapted_options["last_valid_token_pos"] = options["last_valid_token_pos"]

        # Build the same input structure as during export
        inputs = (adapted_tokens, adapted_options)

        t1 = time_module.perf_counter()
        timing_stats["options_build_time"] += t1 - t0

        # TIME: Flatten using pytree (same order as torch.export)
        t0 = time_module.perf_counter()
        flat_inputs, _ = pytree.tree_flatten(inputs)
        t1 = time_module.perf_counter()
        timing_stats["flatten_time"] += t1 - t0

        # TIME: Execute PTE model
        t0 = time_module.perf_counter()
        outputs = method.execute(flat_inputs)
        t1 = time_module.perf_counter()
        timing_stats["execute_time"] += t1 - t0

        # TIME: Reconstruct outputs
        t0 = time_module.perf_counter()

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
    tokenizer = Tokenizer(args.tokenizer)

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

    # Calculate cache lengths for each method
    # The export script uses: cache_len = max_context_len - input_len
    # So for multifunction models:
    # - prefill: input_len=64, cache_len=960 (total=1024)
    # - decode: input_len=1, cache_len=1023 (total=1024)
    prefill_input_len = args.input_len  # e.g., 64
    prefill_cache_len = args.max_context_len - args.input_len  # e.g., 960
    decode_input_len = 1
    decode_cache_len = args.max_context_len - decode_input_len  # e.g., 1023

    print(f"Prefill: input_len={prefill_input_len}, cache_len={prefill_cache_len}")
    print(f"Decode: input_len={decode_input_len}, cache_len={decode_cache_len}")

    # Create StaticAttentionIOManager for prefill
    # This manager handles the main prefill/decode loop state
    prefill_mgr = StaticAttentionIOManager(
        model_args,
        input_len=prefill_input_len,
        cache_lens=prefill_cache_len,
        batch_size=1,
        dtype=torch.float16,
        style="smart_mask",
        mask_val=float("-inf"),
    )

    # Create a separate decode manager with correct cache length for mask shapes
    # This is needed because decode method expects mask shape (1, 1, 1024)
    # which requires cache_len=1023 when input_len=1
    decode_mgr = StaticAttentionIOManager(
        model_args,
        input_len=decode_input_len,
        cache_lens=decode_cache_len,
        batch_size=1,
        dtype=torch.float16,
        style="smart_mask",
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
        prefill_mgr,
        decode_mgr,
        prefill_input_len,
        prefill_cache_len,
        decode_cache_len,
    )

    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print("-" * 50)

    # Reset manager (use prefill_mgr as the main state manager)
    prefill_mgr.reset()
    decode_mgr.reset()

    # Prefill using StaticAttentionIOManager.prefill
    # This will call model_fn with seq_len=input_len, which selects the prefill method
    print("Prefilling (using 'prefill' method)...", end=" ", flush=True)
    start_time = time.time()
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

    # After prefill, copy the cache state from prefill_mgr to decode_mgr
    # This is necessary because decode_mgr has larger caches (1023 vs 960)
    # and we'll be using decode_mgr.decode() for generation
    for key in prefill_mgr.k_caches:
        src_k = prefill_mgr.k_caches[key]
        src_v = prefill_mgr.v_caches[key]
        # Copy to decode_mgr's larger cache
        decode_mgr.k_caches[key][:, :, :prefill_cache_len, :] = src_k
        decode_mgr.v_caches[key][:, :, :prefill_cache_len, :] = src_v

    # Sync the position counter
    decode_mgr.pos = prefill_mgr.pos

    # Update decode_mgr's masks to reflect current position
    # The mask needs to unmask the positions that have been filled by prefill
    for mask in decode_mgr._masks.values():
        mask.reset()
        mask.unmask(prefill_mgr.pos)

    # Decode using decode_mgr.decode() which will call model_fn
    # The wrapper will detect seq_len=1 (after we unpad) and route to decode method
    # Since we're using decode_mgr, the cache shapes will match
    print(f"\n{args.prompt}", end="", flush=True)
    print(tokenizer.decode_token(first_token), end="", flush=True)

    decode_start = time.time()

    if args.lookahead:
        # Use lookahead (speculative) decoding
        print(
            f"\n[Using lookahead decoding: ngram={args.ngram_size}, window={args.window_size}, verifications={args.n_verifications}]"
        )
        generated_tokens = decode_mgr.lookahead_decode(
            model_fn,
            first_token,
            n=args.max_new_tokens - 1,  # -1 because first_token counts
            ngram_size=args.ngram_size,
            window_size=args.window_size,
            n_verifications=args.n_verifications,
            stop_tokens=tokenizer.stop_tokens,
        )
    else:
        # Use standard autoregressive decoding (uses 'forward' method)
        print("\n[Using 'forward' (decode) method for generation]")
        generated_tokens = decode_mgr.decode(
            model_fn,
            first_token,
            n=args.max_new_tokens - 1,  # -1 because first_token counts
            stop_tokens=tokenizer.stop_tokens,
        )

    # Print generated tokens (skip first as it's the init_token we already printed)
    for token in generated_tokens[1:]:
        if token in tokenizer.stop_tokens:
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
