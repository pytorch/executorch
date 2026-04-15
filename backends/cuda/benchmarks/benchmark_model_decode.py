#!/usr/bin/env python3
"""Model-level decode benchmark for Qwen3.5 MoE split-K SDPA.

Measures prefill tok/s and decode tok/s across different prompt sizes
and decode lengths to evaluate FlashDecoding++ async softmax impact.
"""

import json
import sys
import time

import torch

# Register Triton kernels before model import
import executorch.backends.cuda.triton.kernels  # noqa: F401

from executorch.examples.models.qwen3_5_moe.export import load_prequantized_model


PROMPT_SIZES = [1, 15, 59, 143, 1694]
DECODE_LENGTHS = [16, 64, 256, 1024]
MODEL_PATH = "/home/gasoonjia/models/qwen35_moe_int4_hqq"
NUM_WARMUP = 2  # warmup runs before timing


def _move_to_cuda(model, config):
    for fqn, buf in list(model.named_buffers()):
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        if buf.device.type == "meta":
            dtype = torch.bfloat16 if buf.dtype != torch.bool else torch.bool
            parent.register_buffer(
                parts[-1], torch.zeros(buf.shape, dtype=dtype, device="cuda")
            )
        else:
            parent.register_buffer(parts[-1], buf.to("cuda"))

    for name, p in model.named_parameters():
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        setattr(
            parent,
            parts[-1],
            torch.nn.Parameter(p.data.to("cuda"), requires_grad=False),
        )

    for layer in model.layers:
        if hasattr(layer.attn, "rotary_emb"):
            rope = layer.attn.rotary_emb
            inv_freq = 1.0 / (
                config.rope_theta
                ** (
                    torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32)
                    / rope.rotary_dim
                )
            )
            rope.inv_freq = inv_freq.to("cuda")
        if hasattr(layer.attn, "mask"):
            layer.attn.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(
                        config.max_seq_len,
                        config.max_seq_len,
                        dtype=torch.bool,
                        device="cuda",
                    )
                ),
            )


def _reset_state(model):
    """Reset all KV caches, conv_state, and recurrent_state to zero."""
    for layer in model.layers:
        attn = layer.attn
        if hasattr(attn, "kv_cache"):
            attn.kv_cache.k_cache.zero_()
            attn.kv_cache.v_cache.zero_()
        if hasattr(attn, "conv_state"):
            attn.conv_state.zero_()
        if hasattr(attn, "recurrent_state"):
            attn.recurrent_state.zero_()


@torch.inference_mode()
def benchmark_prefill(model, prompt_size):
    """Prefill prompt_size tokens one at a time, return tok/s."""
    _reset_state(model)
    tokens = torch.randint(0, 1000, (1, 1), device="cuda", dtype=torch.long)

    # Warmup
    for i in range(min(prompt_size, NUM_WARMUP)):
        pos = torch.tensor([i], device="cuda")
        model(tokens, pos)

    _reset_state(model)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(prompt_size):
        pos = torch.tensor([i], device="cuda")
        model(tokens, pos)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return prompt_size / elapsed if elapsed > 0 else 0.0


@torch.inference_mode()
def benchmark_decode(model, prompt_size, decode_length):
    """Prefill prompt_size tokens, then decode decode_length tokens. Return decode tok/s."""
    _reset_state(model)
    tokens = torch.randint(0, 1000, (1, 1), device="cuda", dtype=torch.long)

    # Prefill
    for i in range(prompt_size):
        pos = torch.tensor([i], device="cuda")
        logits = model(tokens, pos)

    # Get first decode token
    next_token = logits[:, -1:, :].argmax(dim=-1)

    # Warmup decode
    # (we skip warmup here to avoid polluting KV cache beyond prompt_size + decode_length)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(decode_length):
        pos = torch.tensor([prompt_size + i], device="cuda")
        logits = model(next_token, pos)
        next_token = logits[:, -1:, :].argmax(dim=-1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return decode_length / elapsed if elapsed > 0 else 0.0


def main():
    max_seq = max(PROMPT_SIZES) + max(DECODE_LENGTHS) + 16
    print(f"Loading model from {MODEL_PATH} (max_seq_len={max_seq})...")
    model, config = load_prequantized_model(MODEL_PATH, max_seq_len=max_seq)
    _move_to_cuda(model, config)
    model.eval()

    results = {"prefill": {}, "decode": {}}

    # Prefill benchmark
    print("\n=== Prefill Benchmark ===")
    print(f"{'Prompt Size':>12} | {'tok/s':>10}")
    print("-" * 27)
    for ps in PROMPT_SIZES:
        tps = benchmark_prefill(model, ps)
        results["prefill"][ps] = round(tps, 2)
        print(f"{ps:>12} | {tps:>10.2f}")

    # Decode benchmark
    print("\n=== Decode Benchmark ===")
    header = f"{'Prompt Size':>12}"
    for dl in DECODE_LENGTHS:
        header += f" | {'dec=' + str(dl):>12}"
    print(header)
    print("-" * len(header))

    for ps in PROMPT_SIZES:
        row = f"{ps:>12}"
        results["decode"][ps] = {}
        for dl in DECODE_LENGTHS:
            tps = benchmark_decode(model, ps, dl)
            results["decode"][ps][dl] = round(tps, 2)
            row += f" | {tps:>12.2f}"
        print(row)

    # Dump JSON for easy comparison
    print("\n--- JSON ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
