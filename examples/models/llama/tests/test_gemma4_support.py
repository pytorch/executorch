# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.gemma4.convert_weights import gemma4_to_meta
from executorch.examples.models.llama.attention import (
    AttentionGemma4MHA,
    KVCache,
    RingKVCache,
)
from executorch.examples.models.llama.llama_transformer import (
    _get_kv_donor_layer_idx,
    construct_transformer,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope
from executorch.extension.export_util.utils import export_to_edge
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer


class _Gemma4NextTokenModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens=tokens)


class Gemma4SupportTest(unittest.TestCase):
    def test_hf_rope_preserves_input_dtype(self):
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            max_context_len=16,
            use_hf_rope=True,
        )
        rope = Rope(args)

        x = torch.randn(1, 3, 4, 8, dtype=torch.bfloat16)
        freqs_cos, freqs_sin = rope.get_freqs(None, x.shape[1])

        rotated = rope.forward_to_tensor(x, freqs_cos, freqs_sin)

        self.assertEqual(rotated.dtype, x.dtype)

    def test_dual_rope_tables_use_layer_specific_head_dims(self):
        args = ModelArgs(
            dim=64,
            hidden_dim=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            global_head_dim=32,
            max_context_len=32,
            use_hf_rope=True,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        )

        rope = Rope(args)
        sliding_cos, _ = rope.get_freqs_for_layer_type("sliding_attention", None, 3)
        full_cos, _ = rope.get_freqs_for_layer_type("full_attention", None, 3)

        self.assertEqual(sliding_cos.shape, (3, 16))
        self.assertEqual(full_cos.shape, (3, 32))

    def test_shared_layers_pick_same_type_donors(self):
        layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]

        self.assertEqual(
            _get_kv_donor_layer_idx(
                4, n_layers=6, num_kv_shared_layers=2, layer_types=layer_types
            ),
            2,
        )
        self.assertEqual(
            _get_kv_donor_layer_idx(
                5, n_layers=6, num_kv_shared_layers=2, layer_types=layer_types
            ),
            3,
        )

    def test_gemma4_attention_uses_ring_cache_for_sliding_layers(self):
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            global_head_dim=16,
            max_batch_size=1,
            max_context_len=16,
            use_kv_cache=True,
            enable_dynamic_shape=False,
            attention_type="gemma4_mha",
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=4,
        )
        rope = Rope(args)

        sliding_attn = AttentionGemma4MHA(args, layer_id=0, rope=rope)
        full_attn = AttentionGemma4MHA(args, layer_id=1, rope=rope)

        self.assertIsInstance(sliding_attn.kv_cache, RingKVCache)
        self.assertIsInstance(full_attn.kv_cache, KVCache)

    def test_gemma4_attention_uses_unit_attention_scale(self):
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            global_head_dim=16,
            attention_type="gemma4_mha",
            attention_multiplier=1.0,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
            use_kv_cache=True,
            max_batch_size=1,
            max_context_len=16,
        )
        rope = Rope(args)

        sliding_attn = AttentionGemma4MHA(args, layer_id=0, rope=rope)
        full_attn = AttentionGemma4MHA(args, layer_id=1, rope=rope)

        self.assertEqual(sliding_attn.attention_scale, 1.0)
        self.assertEqual(full_attn.attention_scale, 1.0)
        self.assertEqual(sliding_attn.SDPA.scale, 1.0)
        self.assertEqual(full_attn.SDPA.scale, 1.0)

    def test_transformer_executes_shared_layers_and_softcaps_logits(self):
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            global_head_dim=16,
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=4,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            act_fn="gelu_pytorch_tanh",
            norm_eps=1e-6,
            post_attention_norm=True,
            post_ffn_norm=True,
            apply_embedding=True,
            embedding_scale_factor=1.5,
            use_hf_rope=True,
            attention_type="gemma4_mha",
            final_logit_softcapping=2.0,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=4,
            max_batch_size=1,
            max_seq_len=16,
            max_context_len=16,
            use_kv_cache=True,
            generate_full_logits=True,
            enable_dynamic_shape=False,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        )

        torch.manual_seed(0)
        model = construct_transformer(args)
        self.assertEqual(model.layers[0].feed_forward.w1.weight.shape[0], 64)
        self.assertEqual(model.layers[2].feed_forward.w1.weight.shape[0], 128)

        shared_layer_calls = []

        def _make_hook(layer_idx):
            def _hook(_module, _inputs, _outputs):
                shared_layer_calls.append(layer_idx)

            return _hook

        hooks = [
            model.layers[2].register_forward_hook(_make_hook(2)),
            model.layers[3].register_forward_hook(_make_hook(3)),
        ]
        try:
            tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
            logits = model(
                tokens=tokens,
                attn_options={"input_pos": torch.tensor([0, 1, 2], dtype=torch.long)},
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(shared_layer_calls, [2, 3])
        self.assertEqual(logits.shape, (1, 3, 64))
        self.assertLessEqual(float(logits.detach().abs().max()), 2.0001)

    def test_per_layer_token_embeddings_are_scaled_before_mix(self):
        args = ModelArgs(
            dim=16,
            hidden_dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=4,
            vocab_size=16,
            vocab_size_per_layer_input=16,
            hidden_size_per_layer_input=4,
            attention_type="gemma4_mha",
            max_seq_len=8,
            max_context_len=8,
        )

        model = construct_transformer(args)
        model.per_layer_model_projection.weight.data.zero_()
        model.per_layer_projection_norm.weight.data.fill_(1.0)
        model.embed_tokens_per_layer.weight.data.zero_()
        model.embed_tokens_per_layer.weight.data[1].fill_(1.0)

        captured = {}

        def _capture_forward_layers(
            h, freqs_cos, freqs_sin, attn_options_, seqlen, per_layer_inputs=None
        ):
            captured["per_layer_inputs"] = per_layer_inputs.detach().clone()
            return h, None

        model._forward_layers = _capture_forward_layers
        _ = model(tokens=torch.tensor([[1]], dtype=torch.long))

        expected_scale = (args.hidden_size_per_layer_input**0.5) * (2.0**-0.5)
        expected = torch.full(
            (1, 1, args.n_layers, args.hidden_size_per_layer_input),
            expected_scale,
        )
        self.assertTrue(torch.allclose(captured["per_layer_inputs"], expected))

    def test_gemma4_layer_scalar_scales_block_output(self):
        args = ModelArgs(
            dim=16,
            hidden_dim=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=4,
            vocab_size=32,
            attention_type="gemma4_mha",
            max_seq_len=8,
            max_context_len=8,
        )

        torch.manual_seed(0)
        model = construct_transformer(args)
        layer = model.layers[0]
        self.assertTrue(hasattr(layer, "layer_scalar"))

        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        base_logits = model(tokens=tokens)

        layer.layer_scalar.zero_()
        scaled_logits = model(tokens=tokens)

        self.assertGreater(float(base_logits.detach().abs().max()), 0.0)
        self.assertTrue(torch.allclose(scaled_logits, torch.zeros_like(scaled_logits)))

    def test_tiny_gemma4_export_runtime_matches_eager(self):
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            global_head_dim=16,
            vocab_size=32,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=4,
            num_kv_shared_layers=2,
            act_fn="gelu_pytorch_tanh",
            norm_eps=1e-6,
            post_attention_norm=True,
            post_ffn_norm=True,
            apply_embedding=True,
            embedding_scale_factor=1.5,
            use_hf_rope=True,
            attention_type="gemma4_mha",
            attention_multiplier=1.0,
            final_logit_softcapping=2.0,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=4,
            max_batch_size=1,
            max_seq_len=16,
            max_context_len=16,
            generate_full_logits=False,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        )

        torch.manual_seed(0)
        model = _Gemma4NextTokenModule(construct_transformer(args)).eval()
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        edge_program = export_to_edge(model, (tokens,), strict=True, verbose=False)
        executorch_program = edge_program.to_executorch()
        executorch_model = _load_for_executorch_from_buffer(executorch_program.buffer)

        with torch.no_grad():
            eager_logits = model(tokens)
            executorch_logits = executorch_model.run_method("forward", (tokens,))[0]

        self.assertEqual(eager_logits.shape, executorch_logits.shape)
        self.assertTrue(
            torch.allclose(eager_logits, executorch_logits, rtol=1e-4, atol=1e-4)
        )
        self.assertEqual(
            int(torch.argmax(eager_logits, dim=-1).item()),
            int(torch.argmax(executorch_logits, dim=-1).item()),
        )


class Gemma4ConvertWeightsTest(unittest.TestCase):
    def test_maps_text_and_per_layer_weights_from_multimodal_checkpoint(self):
        state_dict = {
            "model.language_model.embed_tokens.weight": torch.randn(16, 8),
            "model.language_model.embed_tokens_per_layer.weight": torch.randn(16, 12),
            "model.language_model.per_layer_model_projection.weight": torch.randn(12, 8),
            "model.language_model.per_layer_projection_norm.weight": torch.randn(4),
            "model.language_model.norm.weight": torch.randn(8),
            "model.language_model.layers.0.input_layernorm.weight": torch.randn(8),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(16, 8),
            "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(8, 8),
            "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
            "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
            "model.language_model.layers.0.self_attn.q_norm.weight": torch.randn(4),
            "model.language_model.layers.0.self_attn.k_norm.weight": torch.randn(4),
            "model.language_model.layers.0.self_attn.v_norm.weight": torch.randn(4),
            "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(8),
            "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.randn(8),
            "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.randn(8),
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(12, 8),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(8, 12),
            "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(12, 8),
            "model.language_model.layers.0.layer_scalar": torch.ones(1),
            "model.language_model.layers.0.per_layer_input_gate.weight": torch.randn(4, 8),
            "model.language_model.layers.0.per_layer_projection.weight": torch.randn(8, 4),
            "model.language_model.layers.0.post_per_layer_input_norm.weight": torch.randn(8),
            "model.vision_tower.weight": torch.randn(8, 8),
        }

        converted = gemma4_to_meta(state_dict)

        self.assertIn("tok_embeddings.weight", converted)
        self.assertIn("embed_tokens_per_layer.weight", converted)
        self.assertIn("per_layer_model_projection.weight", converted)
        self.assertIn("layers.0.attention.wq.weight", converted)
        self.assertIn("layers.0.post_attention_norm.weight", converted)
        self.assertIn("layers.0.layer_scalar", converted)
        self.assertIn("layers.0.per_layer_projection.weight", converted)
        self.assertIn("output.weight", converted)
        self.assertNotIn("layers.0.attention.v_norm_fn.weight", converted)

    def test_raises_on_unexpected_text_key(self):
        state_dict = {
            "model.language_model.embed_tokens.weight": torch.randn(16, 8),
            "model.language_model.layers.0.unknown.weight": torch.randn(8, 8),
        }

        with self.assertRaisesRegex(
            ValueError, "Unexpected checkpoint key not mapped for Gemma4 export"
        ):
            gemma4_to_meta(state_dict)


if __name__ == "__main__":
    unittest.main()
