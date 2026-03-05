import unittest

import torch
from executorch.examples.models.qwen3_5.convert_weights import qwen_3_5_to_meta


class Qwen35ConvertWeightsTest(unittest.TestCase):
    def test_maps_full_and_linear_attention_weights(self):
        state_dict = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            "model.norm.weight": torch.randn(8),
            "lm_head.weight": torch.randn(16, 8),
            "model.layers.0.input_layernorm.weight": torch.randn(8),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(8),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(12, 8),
            "model.layers.0.mlp.down_proj.weight": torch.randn(8, 12),
            "model.layers.0.mlp.up_proj.weight": torch.randn(12, 8),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 8),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(8, 8),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(8, 8),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(4),
            "model.layers.0.self_attn.k_norm.weight": torch.randn(4),
            "model.layers.1.linear_attn.in_proj_qkv.weight": torch.randn(24, 8),
            "model.layers.1.linear_attn.in_proj_z.weight": torch.randn(8, 8),
            "model.layers.1.linear_attn.in_proj_b.weight": torch.randn(2, 8),
            "model.layers.1.linear_attn.in_proj_a.weight": torch.randn(2, 8),
            "model.layers.1.linear_attn.conv1d.weight": torch.randn(24, 1, 4),
            "model.layers.1.linear_attn.dt_bias": torch.randn(2),
            "model.layers.1.linear_attn.A_log": torch.randn(2),
            "model.layers.1.linear_attn.norm.weight": torch.randn(4),
            "model.layers.1.linear_attn.out_proj.weight": torch.randn(8, 8),
        }

        converted = qwen_3_5_to_meta(state_dict)
        self.assertIn("layers.0.attention.wq.weight", converted)
        self.assertIn("layers.0.attention.q_norm_fn.weight", converted)
        self.assertIn("layers.1.attention.in_proj_qkv.weight", converted)
        self.assertIn("layers.1.attention.out_proj.weight", converted)
        self.assertIn("layers.1.attention.dt_bias", converted)

    def test_raises_on_unexpected_text_key(self):
        state_dict = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            "model.norm.weight": torch.randn(8),
            "model.layers.0.unknown.weight": torch.randn(8, 8),
        }

        with self.assertRaisesRegex(
            ValueError, "Unexpected checkpoint key not mapped for Qwen3.5 export"
        ):
            qwen_3_5_to_meta(state_dict)

    def test_ignores_known_non_text_keys(self):
        state_dict = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            "model.norm.weight": torch.randn(8),
            "mtp.proj.weight": torch.randn(8, 8),
            "model.visual.patch_embed.weight": torch.randn(8, 8),
        }

        converted = qwen_3_5_to_meta(state_dict)
        self.assertIn("tok_embeddings.weight", converted)
        self.assertIn("output.weight", converted)
        self.assertNotIn("mtp.proj.weight", converted)

    def test_maps_multimodal_language_model_keys(self):
        state_dict = {
            "model.language_model.embed_tokens.weight": torch.randn(16, 8),
            "model.language_model.norm.weight": torch.randn(8),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(
                16, 8
            ),
        }

        converted = qwen_3_5_to_meta(state_dict)
        self.assertIn("tok_embeddings.weight", converted)
        self.assertIn("norm.weight", converted)
        self.assertIn("layers.0.attention.wq.weight", converted)
        self.assertIn("output.weight", converted)

    def test_ignores_linear_attention_conv1d_bias(self):
        state_dict = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            "model.norm.weight": torch.randn(8),
            "model.layers.1.linear_attn.conv1d.weight": torch.randn(24, 1, 4),
            "model.layers.1.linear_attn.conv1d.bias": torch.randn(24),
            "model.layers.1.linear_attn.out_proj.weight": torch.randn(8, 8),
        }

        converted = qwen_3_5_to_meta(state_dict)
        self.assertIn("layers.1.attention.conv1d.weight", converted)
        self.assertIn("layers.1.attention.out_proj.weight", converted)
        self.assertNotIn("layers.1.attention.conv1d.bias", converted)

    def test_ignores_rotary_emb_inv_freq(self):
        state_dict = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            "model.norm.weight": torch.randn(8),
            "model.layers.0.self_attn.rotary_emb.inv_freq": torch.randn(4),
        }

        converted = qwen_3_5_to_meta(state_dict)
        self.assertIn("tok_embeddings.weight", converted)
        self.assertIn("output.weight", converted)
        self.assertNotIn("model.layers.0.self_attn.rotary_emb.inv_freq", converted)

if __name__ == "__main__":
    unittest.main()
