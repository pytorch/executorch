# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.examples.models.llama.attention as attention_module
import torch

from executorch.examples.models.llama.attention import ATTENTION_REGISTRY
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope


class Qwen35AttentionTest(unittest.TestCase):
    def _make_args(self, **kwargs) -> ModelArgs:
        defaults = {
            "dim": 32,
            "n_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "head_dim": 8,
            "hidden_dim": 64,
            "max_seq_len": 16,
            "max_context_len": 16,
            "attention_type": "mha",
        }
        defaults.update(kwargs)
        return ModelArgs(**defaults)

    def test_qwen35_full_attention_output_proj_is_bias_free(self):
        args = self._make_args(
            use_kv_cache=False,
            use_qk_norm=False,
            qk_norm_before_rope=True,
            use_q_gate=True,
            attention_qkv_bias=True,
        )
        rope = Rope(args)
        attn = ATTENTION_REGISTRY["mha"](args, 0, rope)
        self.assertIsNone(attn.wo.bias)

    def test_rmsnorm_preserves_input_dtype_without_unit_offset(self):
        norm = RMSNorm(dim=8, add_unit_offset=False)
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        y = norm(x)
        self.assertEqual(y.dtype, x.dtype)

    def test_qwen35_full_attention_forward_shape(self):
        torch.manual_seed(0)
        args = self._make_args(
            use_kv_cache=False,
            use_hf_rope=True,
            partial_rotary_factor=0.5,
            use_qk_norm=True,
            qk_norm_before_rope=True,
            use_q_gate=True,
            rms_norm_add_unit_offset=True,
        )
        rope = Rope(args)
        attn = ATTENTION_REGISTRY["mha"](args, 0, rope)
        x = torch.randn(1, 3, args.dim)
        freqs_cos, freqs_sin = rope.get_freqs(None, x.shape[1])
        y, _ = attn(x, freqs_cos, freqs_sin)
        self.assertEqual(y.shape, x.shape)

    def test_gated_deltanet_resets_state_on_new_sequence(self):
        torch.manual_seed(0)
        args = self._make_args(
            use_kv_cache=True,
            use_q_gate=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        rope = Rope(args)
        attn = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)

        x = torch.randn(1, 1, args.dim)
        dummy_freq = torch.zeros(1, 1)

        # First token of sequence.
        attn(x, dummy_freq, dummy_freq, input_pos=torch.tensor([0], dtype=torch.long))
        state_after_first = attn.recurrent_state.clone()

        # Decode continuation updates state.
        attn(x, dummy_freq, dummy_freq, input_pos=torch.tensor([1], dtype=torch.long))
        state_after_second = attn.recurrent_state.clone()
        self.assertFalse(torch.allclose(state_after_first, state_after_second))

        # New sequence (input_pos=0) should reset internal state.
        attn(x, dummy_freq, dummy_freq, input_pos=torch.tensor([0], dtype=torch.long))
        state_after_reset = attn.recurrent_state.clone()
        self.assertTrue(torch.allclose(state_after_first, state_after_reset, atol=1e-5))

    def test_gated_deltanet_no_input_pos_does_not_leak_state(self):
        torch.manual_seed(0)
        args = self._make_args(
            use_kv_cache=True,
            use_q_gate=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        rope = Rope(args)
        attn = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)

        x = torch.randn(1, 1, args.dim)
        dummy_freq = torch.zeros(1, 1)

        attn(x, dummy_freq, dummy_freq)
        state_after_first = attn.recurrent_state.clone()

        attn(x, dummy_freq, dummy_freq)
        state_after_second = attn.recurrent_state.clone()

        self.assertTrue(
            torch.allclose(state_after_first, state_after_second, atol=1e-5)
        )

    def test_gated_deltanet_chunked_prefill_matches_full_sequence(self):
        torch.manual_seed(0)
        args = self._make_args(
            use_kv_cache=True,
            use_q_gate=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        rope = Rope(args)
        attn_full = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)
        attn_chunked = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)
        attn_chunked.load_state_dict(attn_full.state_dict())

        x = torch.randn(1, 5, args.dim)
        dummy_freq = torch.zeros(1, 1)

        full_output, _ = attn_full(
            x,
            dummy_freq,
            dummy_freq,
            input_pos=torch.tensor([0], dtype=torch.long),
        )

        chunk_outputs = []
        for start, end in ((0, 3), (3, 4), (4, 5)):
            output, _ = attn_chunked(
                x[:, start:end],
                dummy_freq,
                dummy_freq,
                input_pos=torch.tensor([start], dtype=torch.long),
            )
            chunk_outputs.append(output)

        chunked_output = torch.cat(chunk_outputs, dim=1)

        self.assertTrue(torch.allclose(chunked_output, full_output, atol=1e-5))
        self.assertTrue(
            torch.allclose(
                attn_chunked.recurrent_state, attn_full.recurrent_state, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(attn_chunked.conv_state, attn_full.conv_state, atol=1e-5)
        )

    def test_gated_deltanet_custom_op_matches_fallback(self):
        recurrent_op = attention_module._get_recurrent_gated_delta_rule_op()
        if recurrent_op is None:
            self.skipTest("llama::recurrent_gated_delta_rule is not available")

        torch.manual_seed(0)
        args = self._make_args(
            use_kv_cache=True,
            use_q_gate=True,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        rope = Rope(args)
        attn_custom = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)
        attn_fallback = ATTENTION_REGISTRY["gated_deltanet"](args, 0, rope)
        attn_fallback.load_state_dict(attn_custom.state_dict())

        query = torch.randn(1, 3, attn_custom.num_v_heads, attn_custom.head_k_dim)
        key = torch.randn(1, 3, attn_custom.num_v_heads, attn_custom.head_k_dim)
        value = torch.randn(1, 3, attn_custom.num_v_heads, attn_custom.head_v_dim)
        g = torch.randn(1, 3, attn_custom.num_v_heads)
        beta = torch.sigmoid(torch.randn(1, 3, attn_custom.num_v_heads))

        original_op = attention_module._RECURRENT_GATED_DELTA_RULE_OP
        original_tried_loading = (
            attention_module._TRIED_LOADING_RECURRENT_GATED_DELTA_RULE_OP
        )
        try:
            attention_module._RECURRENT_GATED_DELTA_RULE_OP = recurrent_op
            attention_module._TRIED_LOADING_RECURRENT_GATED_DELTA_RULE_OP = True
            custom_output = attn_custom._recurrent_gated_delta_rule(
                query, key, value, g, beta
            )

            attention_module._RECURRENT_GATED_DELTA_RULE_OP = None
            attention_module._TRIED_LOADING_RECURRENT_GATED_DELTA_RULE_OP = True
            fallback_output = attn_fallback._recurrent_gated_delta_rule(
                query, key, value, g, beta
            )
        finally:
            attention_module._RECURRENT_GATED_DELTA_RULE_OP = original_op
            attention_module._TRIED_LOADING_RECURRENT_GATED_DELTA_RULE_OP = (
                original_tried_loading
            )

        self.assertTrue(torch.allclose(custom_output, fallback_output, atol=1e-5))
        self.assertTrue(
            torch.allclose(
                attn_custom.recurrent_state, attn_fallback.recurrent_state, atol=1e-5
            )
        )


if __name__ == "__main__":
    unittest.main()
