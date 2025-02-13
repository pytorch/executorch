import unittest

import torch
from executorch.examples.models.llama.attention import AttentionMHA, ForwardOptions
from executorch.examples.models.llama.llama_transformer import Transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope
from executorch.examples.models.llama.static_attention import (
    StaticAttention,
    StaticKVCache,
)


class StaticAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_without_cache(self):
        config = ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=8,
        )
        layer_id = 0
        rope = Rope(config)
        attn_mha = AttentionMHA(config, layer_id, rope).eval()
        static_attn = StaticAttention(config, layer_id, rope).eval()
        static_attn.load_weights_from_attention_mha(attn_mha)

        x = torch.rand(1, config.max_seq_len, config.dim)
        freqs_cos, freqs_sin = rope.get_freqs(None, config.max_seq_len)
        expected, _ = attn_mha(x, freqs_cos, freqs_sin)
        mask = torch.triu(
            torch.full((1, config.max_seq_len, config.max_seq_len), float("-inf")),
            diagonal=1,
        )
        y, _ = static_attn(
            x,
            freqs_cos,
            freqs_sin,
            mask=mask,
        )
        self.assertTrue(torch.isclose(y, expected, rtol=1e-3).all())

    def test_with_cache(self):
        config = ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=24,
        )
        layer_id = 0
        rope = Rope(config)
        attn_mha = AttentionMHA(config, layer_id, rope).eval()
        static_attn = StaticAttention(config, layer_id, rope).eval()
        static_attn.load_weights_from_attention_mha(attn_mha)

        x = torch.rand(1, config.max_seq_len, config.dim)
        freqs_cos, freqs_sin = rope.get_freqs(None, config.max_seq_len)
        expected, _ = attn_mha(x, freqs_cos, freqs_sin)

        n_chunks = 3
        chunk_len = config.max_seq_len // n_chunks
        cache_len = config.max_seq_len - chunk_len
        mask = torch.zeros(1, chunk_len, cache_len + chunk_len)
        mask[:, :, :cache_len] = float("-inf")
        mask[:, :, cache_len:] = torch.triu(
            torch.full((1, chunk_len, chunk_len), float("-inf")),
            diagonal=1,
        )
        k_caches = {
            StaticKVCache.calculate_cache_key(layer_id, i): torch.zeros(
                1, cache_len, config.head_dim
            )
            for i in range(config.n_kv_heads)
        }
        v_caches = {
            StaticKVCache.calculate_cache_key(layer_id, i): torch.zeros(
                1, cache_len, config.head_dim
            )
            for i in range(config.n_kv_heads)
        }
        ys = []
        for i in range(n_chunks):
            y_i, attn_update = static_attn(
                x[:, i * chunk_len : (i + 1) * chunk_len, :],
                freqs_cos[i * chunk_len : (i + 1) * chunk_len],
                freqs_sin[i * chunk_len : (i + 1) * chunk_len],
                mask=mask,
                in_cache_state=(k_caches, v_caches),
                out_cache_state=({}, {}),
            )
            ys.append(y_i)
            mask[:, :, cache_len - chunk_len * (i + 1) : cache_len] = 0
            k_cache_updates, v_cache_updates = attn_update["out_cache_state"]
            for cache_id, update in k_cache_updates.items():
                k_caches[cache_id] = StaticKVCache.apply_update(
                    k_caches[cache_id], update
                )
            for cache_id, update in v_cache_updates.items():
                v_caches[cache_id] = StaticKVCache.apply_update(
                    v_caches[cache_id], update
                )

        y = torch.cat(ys, dim=1)
        self.assertTrue(torch.isclose(y, expected, rtol=1e-3).all())

    def test_within_transformer(self):
        config = ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=24,
            n_layers=4,
            vocab_size=128,
        )
        mha_transformer = Transformer(config).eval()

        config.attention_type = "static"
        static_transformer = Transformer(config).eval()
        static_transformer.load_state_dict(mha_transformer.state_dict(), strict=False)
        for mha_layer, static_layer in zip(
            mha_transformer.layers, static_transformer.layers
        ):
            static_layer.attention.load_weights_from_attention_mha(mha_layer.attention)

        x = torch.randint(config.vocab_size, (1, config.max_seq_len))
        rope = Rope(config)
        freqs_cos, freqs_sin = rope.get_freqs(None, config.max_seq_len)
        expected = mha_transformer(x)

        n_chunks = 3
        chunk_len = config.max_seq_len // n_chunks
        cache_len = config.max_seq_len - chunk_len
        mask = torch.zeros(1, chunk_len, cache_len + chunk_len)
        mask[:, :, :cache_len] = float("-inf")
        mask[:, :, cache_len:] = torch.triu(
            torch.full((1, chunk_len, chunk_len), float("-inf")),
            diagonal=1,
        )
        k_caches = {
            StaticKVCache.calculate_cache_key(layer_id, i): torch.zeros(
                1, cache_len, config.head_dim
            )
            for layer_id in range(config.n_layers)
            for i in range(config.n_kv_heads)
        }
        v_caches = {
            StaticKVCache.calculate_cache_key(layer_id, i): torch.zeros(
                1, cache_len, config.head_dim
            )
            for layer_id in range(config.n_layers)
            for i in range(config.n_kv_heads)
        }
        ys = []
        for i in range(n_chunks):
            y_i, attn_update = static_transformer(
                x[:, i * chunk_len : (i + 1) * chunk_len],
                attn_options=ForwardOptions(
                    mask=mask,
                    freqs_cos_override=freqs_cos[i * chunk_len : (i + 1) * chunk_len],
                    freqs_sin_override=freqs_sin[i * chunk_len : (i + 1) * chunk_len],
                    in_cache_state=(k_caches, v_caches),
                    out_cache_state=({}, {}),
                ),
            )
            ys.append(y_i)
            mask[:, :, cache_len - chunk_len * (i + 1) : cache_len] = 0
            k_cache_updates, v_cache_updates = attn_update["out_cache_state"]
            for cache_id, update in k_cache_updates.items():
                k_caches[cache_id] = StaticKVCache.apply_update(
                    k_caches[cache_id], update
                )
            for cache_id, update in v_cache_updates.items():
                v_caches[cache_id] = StaticKVCache.apply_update(
                    v_caches[cache_id], update
                )

        self.assertTrue(torch.isclose(ys[-1], expected, rtol=1e-3).all())
