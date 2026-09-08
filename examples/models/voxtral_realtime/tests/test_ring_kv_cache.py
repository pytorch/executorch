# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from types import ModuleType
from unittest.mock import patch

import torch
import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

with patch.dict(
    "sys.modules",
    {"executorch.extension.llm.custom_ops.custom_ops": ModuleType("custom_ops")},
):
    from executorch.examples.models.voxtral_realtime.model import (
        RingKVCache,
        StandardRingKVCache,
        StreamingAudioEncoderExport,
        VoxtralRealtimeConfig,
        VoxtralRealtimeModel,
    )


class StandardRingKVCacheTest(unittest.TestCase):
    def test_additive_mask_uses_finite_negative_values(self):
        cache = StandardRingKVCache(window_size=4, n_heads=1, head_dim=2)

        mask = cache.create_causal_mask(
            torch.tensor(0), seq_len=1, dtype=torch.bfloat16
        )

        self.assertEqual(mask.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(mask).all())
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertLess(mask[0, 1].float().item(), -1e8)

    def test_bool_mask_keeps_bool_dtype(self):
        cache = StandardRingKVCache(window_size=4, n_heads=1, head_dim=2)

        mask = cache.create_causal_mask(torch.tensor(3), seq_len=2, bool_mask=True)

        self.assertEqual(mask.dtype, torch.bool)


class VulkanRingSDPATest(unittest.TestCase):
    def test_matches_explicit_physical_cache_mask_across_wraps(self):
        window_size = 4
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8

        for start_pos, seq_len in (
            (0, 1),
            (3, 1),
            (4, 1),
            (8, 1),
            (0, 4),
            (4, 4),
            (8, 4),
        ):
            with self.subTest(start_pos=start_pos, seq_len=seq_len):
                torch.manual_seed(20260816 + start_pos + seq_len)
                cache = RingKVCache(window_size, num_kv_heads, head_dim)
                for pos in range(start_pos + seq_len):
                    cache.k_cache[:, pos % cache.buf_size].copy_(
                        torch.randn(1, num_kv_heads, head_dim)
                    )
                    cache.v_cache[:, pos % cache.buf_size].copy_(
                        torch.randn(1, num_kv_heads, head_dim)
                    )

                q = torch.randn(1, seq_len, num_heads, head_dim)
                mask = cache.create_causal_mask(start_pos, seq_len)
                k = torch.repeat_interleave(
                    cache.k_cache, num_heads // num_kv_heads, dim=2
                )
                v = torch.repeat_interleave(
                    cache.v_cache, num_heads // num_kv_heads, dim=2
                )
                q_bhsd = q.transpose(1, 2)
                k_bhsd = k.transpose(1, 2)
                v_bhsd = v.transpose(1, 2)
                attn = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1))
                attn = attn / (head_dim**0.5) + mask
                expected = torch.matmul(torch.softmax(attn, dim=-1), v_bhsd)
                expected = expected.transpose(1, 2)

                actual = torch.ops.et_vk.ring_sdpa(
                    q,
                    cache.k_cache,
                    cache.v_cache,
                    start_pos,
                    window_size,
                )
                torch.testing.assert_close(actual, expected)

    def test_two_chunk_encoder_matches_sequential_calls(self):
        config = VoxtralRealtimeConfig(
            dim=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            hidden_dim=64,
            vocab_size=64,
            ada_rms_norm_t_cond_dim=8,
            enc_dim=32,
            enc_n_layers=1,
            enc_n_heads=4,
            enc_head_dim=8,
            enc_hidden_dim=64,
            num_mel_bins=8,
            downsample_factor=4,
            max_seq_len=8,
            sliding_window=8,
            streaming=True,
            backend="cuda",
        )
        torch.manual_seed(20260817)
        model = VoxtralRealtimeModel(config).eval()
        sequential = StreamingAudioEncoderExport(
            copy.deepcopy(model), max_enc_len=8
        ).eval()
        batched = StreamingAudioEncoderExport(
            copy.deepcopy(model), max_enc_len=8
        ).eval()
        mel = torch.randn(1, config.num_mel_bins, 16)

        with torch.inference_mode():
            expected = torch.cat(
                (
                    sequential(mel[:, :, :8], torch.arange(4)),
                    sequential(mel[:, :, 8:], torch.arange(4, 8)),
                ),
                dim=1,
            )
            actual = batched(mel, torch.arange(8))

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
