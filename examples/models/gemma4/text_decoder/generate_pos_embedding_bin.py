#!/usr/bin/env python3
"""Precompute the Gemma4 RoPE position-embedding .bin for the Exynos LLM runner.

Reproduces the RoPE convention from gemma4_attention.py: inv_freq denominator is
head_dim, full-attention (global) layers use partial rotary while sliding (local)
layers use full rotary, non-rotary dims zero-padded so split-half rotate_half
pairs correctly. Emits the layout RotaryEmbedding::load_from_file expects:

    [global_cos | global_sin | local_cos | local_sin]

each table row-major [max_seq_len, head_dim] float32 (global head_dim 512, local
head_dim 256).
"""

import argparse

import torch
from gemma4_config import Gemma4Config


def rope_table(
    max_seq_len: int,
    head_dim: int,
    base: float,
    partial_rotary_factor: float,
    full_rope: bool,
):
    rotary_dim = head_dim if full_rope else int(head_dim * partial_rotary_factor)
    rope_angles = rotary_dim // 2
    inv_freq_rotated = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2).float() / head_dim)
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)])
    else:
        inv_freq = inv_freq_rotated

    pos = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="HF config.json path")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", required=True, help="output pos_embedding.bin path")
    args = ap.parse_args()

    cfg = Gemma4Config.from_json(args.config)

    g_cos, g_sin = rope_table(
        args.max_seq_len,
        cfg.global_head_dim,
        cfg.rope_theta,
        cfg.partial_rotary_factor,
        full_rope=False,
    )
    l_cos, l_sin = rope_table(
        args.max_seq_len,
        cfg.head_dim,
        cfg.rope_local_base_freq,
        cfg.partial_rotary_factor,
        full_rope=True,
    )

    with open(args.out, "wb") as f:
        for t in (g_cos, g_sin, l_cos, l_sin):
            f.write(t.contiguous().to(torch.float32).numpy().tobytes())

    nbytes = args.max_seq_len * (cfg.global_head_dim + cfg.head_dim) * 2 * 4
    print(
        f"wrote {args.out}: {nbytes} bytes "
        f"(max_seq_len={args.max_seq_len}, global_head_dim={cfg.global_head_dim}, "
        f"head_dim={cfg.head_dim}, rope_theta={cfg.rope_theta}, "
        f"rope_local_base_freq={cfg.rope_local_base_freq}, "
        f"partial_rotary_factor={cfg.partial_rotary_factor})"
    )


if __name__ == "__main__":
    main()
