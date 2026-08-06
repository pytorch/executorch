"""Eager equivalence probe for the DFlash draft KV cache.

Not part of the PR -- a throwaway diagnostic. Run directly:
    python scratch_draft_cache_equiv.py

Proves the cached draft forward produces logits numerically identical to
the uncached path across several simulated speculative-decoding rounds.
"""

import torch

from executorch.backends.mlx.examples.llm.dflash_draft_model import (
    DFlashConfig,
    DFlashDraftModel,
)


def make_model():
    torch.manual_seed(0)
    config = DFlashConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=100,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        target_layer_ids=(0,),
        block_size=8,
    )
    model = DFlashDraftModel(config)
    model.qwen3_config._attn_implementation = "eager"
    for layer in model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    return model.eval(), config


def simulate_rounds(model, config, num_rounds=4, use_cache=False):
    hidden_size = config.hidden_size
    block_len = config.block_size

    gen = torch.Generator().manual_seed(123)
    ctx = torch.randn(1, 5, hidden_size, generator=gen)
    per_round_logits = []

    cache = None
    if use_cache:
        from executorch.backends.mlx.examples.llm.dflash_draft_cache import (
            DFlashDraftKVCache,
        )

        cache = DFlashDraftKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
        )

    prev_ctx_len = 0
    for r in range(num_rounds):
        tokens = torch.randint(0, config.vocab_size, (1, block_len), dtype=torch.long, generator=gen)
        S = ctx.shape[1]
        with torch.no_grad():
            if use_cache:
                new_ctx_len = S - prev_ctx_len
                cache_position = torch.arange(prev_ctx_len, S, dtype=torch.long)
                logits = model(
                    tokens, ctx,
                    cache=cache, cache_position=cache_position, new_ctx_len=new_ctx_len,
                )
                prev_ctx_len = S
            else:
                logits = model(tokens, ctx)
        per_round_logits.append(logits)

        accepted = (r % block_len) + 1
        new_ctx = torch.randn(1, accepted, hidden_size, generator=gen)
        ctx = torch.cat([ctx, new_ctx], dim=1)

    return per_round_logits


def main():
    model, config = make_model()

    reference_logits = simulate_rounds(model, config, use_cache=False)
    print(f"Captured uncached reference for {len(reference_logits)} rounds.")

    cached_logits = simulate_rounds(model, config, use_cache=True)
    print(f"Captured cached path for {len(cached_logits)} rounds.")

    max_diff = max(
        (a - b).abs().max().item()
        for a, b in zip(reference_logits, cached_logits)
    )
    print(f"\ncached-vs-uncached max abs diff across rounds: {max_diff:.3e}")
    for i, (a, b) in enumerate(zip(reference_logits, cached_logits)):
        d = (a - b).abs().max().item()
        print(f"  round {i}: diff {d:.3e}")

    if max_diff < 1e-4:
        print("\nPASS: cached path matches uncached -- wiring is numerically correct.")
    else:
        print("\nFAIL: cached path diverges from uncached -- wiring bug.")


if __name__ == "__main__":
    main()
