# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager inference on a prequantized Gemma 4 31B-IT model (CUDA + torch.compile).

Loads a quantized checkpoint (from ``quantize_and_save.py``), packs for CUDA,
materializes runtime buffers, optionally compiles with ``torch.compile``, and
generates text autoregressively. The model performs Gumbel-max sampling
on-device, so each forward returns the next token ID as a float tensor of
shape ``[B, 1]``.

Usage:
    python inference.py \\
        --prequantized ./gemma4_31b_int4 \\
        --prompt "Write a short joke about saving RAM." \\
        --max-new-tokens 128 \\
        --temperature 0.8
"""

import argparse
import os
import time

import torch

from executorch.examples.models.gemma4_31b.export import load_prequantized_model
from executorch.examples.models.gemma4_31b.model import materialize_runtime_buffers


def _move_to_cuda(model, config) -> None:
    """Move the prequantized model to CUDA and materialize runtime buffers there.

    Parameters are moved individually (not via ``model.cuda()``) to preserve
    ``Int4TilePackedTo4dTensor`` subclass identity. Non-meta buffers (e.g.
    ``layer_scalar``) are moved to CUDA. Meta-device buffers (KV cache, RoPE,
    constants) are materialized directly on CUDA via
    ``materialize_runtime_buffers``.
    """
    for name, p in model.named_parameters():
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        setattr(
            parent,
            parts[-1],
            torch.nn.Parameter(p.data.to("cuda"), requires_grad=False),
        )

    for fqn, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(parts[-1], buf.to("cuda"), persistent=False)

    materialize_runtime_buffers(model, dtype=torch.bfloat16, device="cuda")


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    eos_token_ids=None,
    bos_token_id: int = 2,
) -> str:
    """Autoregressive generation. Prefill is one-token-at-a-time so a single
    compiled graph handles every step; the exported PTE uses a separate
    multi-token prefill method, but for eager+compile a uniform decode-shape
    forward is simpler and benefits from CUDA-graph friendly shapes.

    ``tokenizers.Tokenizer.from_file`` does not auto-prepend BOS — and Gemma 4
    is unusable without it (the model's logits collapse to a single
    high-frequency vocab token if the very first input isn't BOS). We prepend
    explicitly here; pass ``bos_token_id=None`` to disable.
    """
    if eos_token_ids is None:
        eos_token_ids = set()

    input_ids = tokenizer.encode(prompt).ids
    if bos_token_id is not None and (not input_ids or input_ids[0] != bos_token_id):
        input_ids = [bos_token_id] + input_ids

    temp_val = max(temperature, 1e-6)  # avoid div-by-zero in the on-device sampler
    temp_tensor = torch.tensor([temp_val], dtype=torch.float32, device="cuda")

    sampled = None
    with torch.no_grad():
        # Prefill, one token at a time.
        for i, tok_id in enumerate(input_ids):
            tok = torch.tensor([[tok_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([i], dtype=torch.long, device="cuda")
            sampled = model(tok, pos, temp_tensor)

        # First generated token from the last prefill step.
        next_id = int(sampled.item())
        generated = [next_id]

        # Decode loop.
        seq_len = len(input_ids)
        for i in range(max_new_tokens - 1):
            tok = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([seq_len + i], dtype=torch.long, device="cuda")
            sampled = model(tok, pos, temp_tensor)
            next_id = int(sampled.item())
            generated.append(next_id)
            if next_id in eos_token_ids:
                break

    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eager inference on prequantized Gemma 4 31B-IT (CUDA)."
    )
    parser.add_argument(
        "--prequantized",
        required=True,
        help="Path to a quantized checkpoint directory.",
    )
    parser.add_argument("--prompt", default="Hello", help="Input prompt.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = near-greedy).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV cache length to allocate for this run.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (slower, but easier to debug).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required for inference.")

    print(f"Loading prequantized model from {args.prequantized}...")
    model, config = load_prequantized_model(
        args.prequantized, max_seq_len=args.max_seq_len
    )
    _move_to_cuda(model, config)
    model.eval()

    if not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    tokenizer_path = os.path.join(args.prequantized, "tokenizer.json")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Gemma 4 EOS tokens (from generation_config.json: ids 1, 50, 106).
    eos_token_ids = {1, 50, 106}

    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    t0 = time.perf_counter()
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_ids=eos_token_ids,
    )
    elapsed = time.perf_counter() - t0

    print(output)
    print("-" * 40)
    print(f"Generated in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
