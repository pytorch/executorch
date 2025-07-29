import copy
import itertools
import unittest
from collections import defaultdict

import torch
from executorch.examples.models.llama.attention import AttentionMHA
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope
from executorch.examples.models.llama.static_attention import (
    StaticAttention,
    StaticAttentionIOManager,
    StaticAttentionMask,
    StaticKVCache,
)


class StaticAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_without_cache(self):
        def test(
            use_qk_norm, qk_norm_before_rope, split_mha, adopt_hf_rope, use_conv2d
        ):
            torch.manual_seed(42)

            # Redundant or unsupported configurations.
            if not use_qk_norm and qk_norm_before_rope:
                return
            if not split_mha and use_conv2d:
                return

            config = ModelArgs(
                dim=64,
                n_heads=4,
                n_kv_heads=2,
                max_seq_len=8,
                use_qk_norm=use_qk_norm,
                qk_norm_before_rope=qk_norm_before_rope,
                attention_type="static" if split_mha else "static_mha",
            )
            layer_id = 0
            rope = Rope(config)
            attn_mha = AttentionMHA(config, layer_id, rope).eval()
            static_attn = StaticAttention(config, layer_id, rope).eval()
            if use_qk_norm:
                with torch.no_grad():
                    attn_mha.q_norm_fn.weight.copy_(
                        torch.rand(config.head_dim) * 0.2 + 0.9
                    )
                    attn_mha.k_norm_fn.weight.copy_(
                        torch.rand(config.head_dim) * 0.2 + 0.9
                    )
            static_attn.load_weights_from_attention_mha(attn_mha)
            if adopt_hf_rope:
                static_attn.adopt_hf_rope()
            if use_conv2d:
                static_attn.linear_to_conv2d()

            x = torch.rand(1, config.max_seq_len, config.dim)
            freqs_cos, freqs_sin = rope.get_freqs(None, config.max_seq_len)
            expected, _ = attn_mha(x, freqs_cos, freqs_sin)

            if adopt_hf_rope:
                config.use_hf_rope = True
                rope = Rope(config)
                freqs_cos, freqs_sin = rope.get_freqs(None, config.max_seq_len)
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
            self.assertTrue(
                torch.isclose(y, expected, rtol=1e-3).all(),
                f"Failed for use_qk_norm={use_qk_norm}, "
                f"qk_norm_before_rope={qk_norm_before_rope}, "
                f"split_mha={split_mha}, "
                f"adopt_hf_rope={adopt_hf_rope}, "
                f"use_conv2d={use_conv2d}",
            )

        for args in itertools.product([False, True], repeat=5):
            test(*args)

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
        static_attn.adopt_hf_rope()

        x = torch.rand(1, config.max_seq_len, config.dim)
        freqs_cos, freqs_sin = rope.get_freqs(None, config.max_seq_len)
        expected, _ = attn_mha(x, freqs_cos, freqs_sin)

        n_chunks = 3
        chunk_len = config.max_seq_len // n_chunks
        cache_len = config.max_seq_len - chunk_len

        config.use_hf_rope = True
        hf_rope = Rope(config)
        hf_freqs_cos, hf_freqs_sin = hf_rope.get_freqs(None, config.max_seq_len)

        def test_with_style(style):
            mask = StaticAttentionMask(chunk_len, cache_len, style=style)
            mask.tensor[:, :, cache_len:] = torch.triu(
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
                    hf_freqs_cos[i * chunk_len : (i + 1) * chunk_len],
                    hf_freqs_sin[i * chunk_len : (i + 1) * chunk_len],
                    mask=mask.tensor,
                    in_cache_state=(k_caches, v_caches),
                    out_cache_state=({}, {}),
                )
                ys.append(y_i)
                mask.unmask(chunk_len)
                k_cache_updates, v_cache_updates = attn_update["out_cache_state"]

                if i < n_chunks - 1:
                    for cache_id, update in k_cache_updates.items():
                        k_caches[cache_id] = StaticKVCache.apply_update(
                            k_caches[cache_id], update, pos=chunk_len * i, style=style
                        )
                    for cache_id, update in v_cache_updates.items():
                        v_caches[cache_id] = StaticKVCache.apply_update(
                            v_caches[cache_id], update, pos=chunk_len * i, style=style
                        )

            y = torch.cat(ys, dim=1)
            self.assertTrue(torch.isclose(y, expected, rtol=1e-3).all())

        test_with_style("shift_pointer")
        test_with_style("smart_mask")

    def _get_test_transformers(self, config, attention_type="static"):
        mha_transformer = construct_transformer(config).eval()

        config = copy.copy(config)
        config.attention_type = attention_type
        static_transformer = construct_transformer(config).eval()
        static_transformer.load_state_dict(mha_transformer.state_dict(), strict=False)
        for mha_layer, static_layer in zip(
            mha_transformer.layers, static_transformer.layers
        ):
            static_layer.attention.load_weights_from_attention_mha(mha_layer.attention)
            static_layer.attention.adopt_hf_rope()
        config.use_hf_rope = True

        return mha_transformer, static_transformer, config

    def test_within_transformer(self):
        config = ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=24,
            n_layers=4,
            vocab_size=128,
        )
        x = torch.randint(config.vocab_size, (1, config.max_seq_len))
        n_chunks = 3
        chunk_len = config.max_seq_len // n_chunks
        cache_len = config.max_seq_len - chunk_len

        def test(style, attention_type):
            mha_transformer, static_transformer, static_config = (
                self._get_test_transformers(
                    config,
                    attention_type,
                )
            )
            expected = mha_transformer(x)

            mgr = StaticAttentionIOManager(
                static_config, chunk_len, cache_len, style=style
            )
            ys = []
            for i in range(n_chunks):
                y_i = mgr.prefill(
                    static_transformer,
                    x[0][i * chunk_len : (i + 1) * chunk_len].tolist(),
                )
                ys.append(y_i)

            self.assertTrue(torch.isclose(ys[-1], expected, rtol=1e-3).all())

        for args in itertools.product(
            ["shift_pointer", "smart_mask"], ["static", "static_mha"]
        ):
            test(*args)

    def test_lookahead_decode(self):
        config = ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            n_layers=4,
            vocab_size=128,
            generate_full_logits=True,
        )
        _, static_transformer, static_config = self._get_test_transformers(config)

        input_len = 32
        cache_len = static_config.max_seq_len - input_len
        prefill_input = torch.randint(static_config.vocab_size, (input_len,))
        ref_mgr = StaticAttentionIOManager(static_config, input_len, cache_len)
        lookahead_mgr = StaticAttentionIOManager(static_config, input_len, cache_len)

        next_tok = (
            ref_mgr.prefill(static_transformer, prefill_input.tolist())[0][-1]
            .argmax()
            .item()
        )
        ref_output = ref_mgr.decode(static_transformer, next_tok, 50)

        ngram_size = 3
        window_size = 8
        n_verifications = 8
        ngram_caches = defaultdict(
            lambda: StaticAttentionIOManager.NGramCache(n_verifications)
        )
        for _ in range(2):  # run twice, first run will populates the cache
            lookahead_mgr.reset()
            next_tok = (
                lookahead_mgr.prefill(static_transformer, prefill_input.tolist())[0][-1]
                .argmax()
                .item()
            )
            lookahead_output = lookahead_mgr.lookahead_decode(
                static_transformer,
                next_tok,
                50,
                ngram_size=ngram_size,
                window_size=window_size,
                n_verifications=n_verifications,
                ngram_caches=ngram_caches,
            )
            self.assertEqual(lookahead_output[: len(ref_output)], ref_output)
