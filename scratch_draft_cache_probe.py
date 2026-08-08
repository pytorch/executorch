"""Standalone probe: does DFlashDraftKVCache correctly self-heal across
rounds, the way the design walkthrough claimed?

Not part of the PR -- a throwaway diagnostic, same style as
scratch_cache_probe.py earlier. Run directly:
    python scratch_draft_cache_probe.py

Mirrors the concrete walkthrough numbers from the design discussion:
    Round 1: ctx_len starts at 20 (a 20-token prompt). Block length 8.
             Block written at [20, 28).
    Round 2: 5 of those 8 accepted -> ctx_len becomes 25. New block (8 more)
             written at [25, 33) -- overwriting round 1's now-stale [25, 28)
             tail and extending past it.

If this passes, positions 20-24 should still hold round 1's block content
(the part that got confirmed), and everything from 25 onward should be
entirely round 2's content, with zero leftover from round 1's rejected
speculative tail.
"""

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_cache import DFlashDraftKVCache


def fake_kv(num_heads, head_dim, seq_len, fill_value):
    shape = (1, num_heads, seq_len, head_dim)
    return (
        torch.full(shape, fill_value, dtype=torch.float32),
        torch.full(shape, fill_value, dtype=torch.float32),
    )


def main():
    num_layers, num_heads, head_dim, max_seq_len = 1, 2, 4, 64
    cache = DFlashDraftKVCache(num_layers, num_heads, head_dim, max_seq_len)

    # Simulate a prompt already having filled positions [0, 20) with some
    # earlier value (3.0), just so there's realistic prior content.
    k0, v0 = fake_kv(num_heads, head_dim, 20, fill_value=3.0)
    cache.write_context(0, k0, v0, start_pos=0)
    cache.advance_context(torch.tensor(20, dtype=torch.long))

    # --- Round 1: block of 8 written at [20, 28), filled with 1.0 ---
    k1, v1 = fake_kv(num_heads, head_dim, 8, fill_value=1.0)
    start_pos_1 = cache.ctx_len.item()
    assert start_pos_1 == 20, f"expected ctx_len=20 before round 1, got {start_pos_1}"
    cache.write_block(0, k1, v1, start_pos=start_pos_1)
    print(f"After round 1 block write, k_cache[18:30]: "
          f"{cache.layers[0].k_cache[0, 0, 18:30, 0].tolist()}")

    # --- Round 2: only 5 of round 1's 8 accepted -> confirmed context grows
    # by 5 (not 8). New block (8 more) written starting at the new ctx_len. ---
    k_confirmed, v_confirmed = fake_kv(num_heads, head_dim, 5, fill_value=1.0)
    start_pos_confirm = cache.ctx_len.item()
    assert start_pos_confirm == 20
    cache.write_context(0, k_confirmed, v_confirmed, start_pos=start_pos_confirm)
    cache.advance_context(torch.tensor(5, dtype=torch.long))

    start_pos_2 = cache.ctx_len.item()
    assert start_pos_2 == 25, f"expected ctx_len=25 before round 2 block, got {start_pos_2}"
    k2, v2 = fake_kv(num_heads, head_dim, 8, fill_value=2.0)
    cache.write_block(0, k2, v2, start_pos=start_pos_2)

    result = cache.layers[0].k_cache[0, 0, 18:33, 0].tolist()
    print(f"After round 2 block write, k_cache[18:33]: {result}")

    expected = [3.0, 3.0] + [1.0] * 5 + [2.0] * 8
    if result == expected:
        print("PASS: self-healing overwrite worked exactly as designed.")
    else:
        print(f"FAIL: expected {expected}, got {result}")

    # Sanity check the mask too: with ctx_len=25 and this round's block_len=8,
    # valid range should be exactly [0, 33).
    mask = cache.valid_mask(block_len=8)
    valid_count = mask.sum().item()
    print(f"valid_mask() reports {valid_count} valid positions (expected 33).")
    assert valid_count == 33, f"expected 33 valid positions, got {valid_count}"
    print("PASS: valid_mask boundary is correct.")


if __name__ == "__main__":
    main()
