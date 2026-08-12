# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that the CACHED draft path produces the same logits as the
uncached path.
"""

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_cache import DFlashDraftKVCache
from executorch.backends.mlx.examples.llm.dflash_draft_model import (
    DFlashConfig,
    DFlashDraftModel,
)


def make_model():
    torch.manual_seed(0)
    config = DFlashConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        vocab_size=100, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=128, target_layer_ids=(0,), block_size=8,
    )
    model = DFlashDraftModel(config)
    model.qwen3_config._attn_implementation = "eager"
    for layer in model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    return model.eval(), config


def simulate_rounds(model, config, num_rounds=4, use_cache=False, max_seq_len=128):
    block_len = config.block_size
    gen = torch.Generator().manual_seed(123)
    ctx = torch.randn(1, 5, config.hidden_size, generator=gen)
    logits_per_round = []

    cache = None
    if use_cache:
        cache = DFlashDraftKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=max_seq_len,
        )

    prev_ctx_len = 0
    for r in range(num_rounds):
        tokens = torch.randint(0, config.vocab_size, (1, block_len), dtype=torch.long, generator=gen)
        with torch.no_grad():
            if use_cache:
                new_ctx = ctx[:, prev_ctx_len:, :]
                cache_position = torch.arange(prev_ctx_len, ctx.shape[1], dtype=torch.long)
                logits = model(tokens, new_ctx, cache=cache, cache_position=cache_position)
                prev_ctx_len = ctx.shape[1]
            else:
                logits = model(tokens, ctx)
        logits_per_round.append(logits)

        accepted = (r % block_len) + 1
        ctx = torch.cat([ctx, torch.randn(1, accepted, config.hidden_size, generator=gen)], dim=1)

    return logits_per_round


def main():
    model, config = make_model()
    uncached = simulate_rounds(model, config, use_cache=False)
    cached = simulate_rounds(model, config, use_cache=True)

    max_diff = max((a - b).abs().max().item() for a, b in zip(uncached, cached))
    print(f"cached-vs-uncached max abs diff across {len(uncached)} rounds: {max_diff:.3e}")
    for i, (a, b) in enumerate(zip(uncached, cached)):
        print(f"  round {i}: {(a - b).abs().max().item():.3e}")

    assert max_diff < 1e-4, "cached path diverges from uncached"
    print("PASS: cached path matches uncached in eager mode")


if __name__ == "__main__":
    main()
