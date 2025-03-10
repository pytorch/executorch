import unittest

import torch
from executorch.examples.models.llama.attention import (
    AttentionMHA,
    KVCache,
    ModelArgs,
    Rope,
    SDPA,
)


class TestAttentionMHA(unittest.TestCase):

    def create_mock_args(self):
        return ModelArgs(
            use_kv_cache=True,
            n_heads=8,
            n_kv_heads=4,
            head_dim=64,
            max_batch_size=2,
            max_context_len=16,
            dim=512,
            attention_qkv_bias=False,
            enable_dynamic_shape=False,
        )

    def test_attentionmha_init(self):
        args = self.create_mock_args()
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)

        self.assertEqual(attn.n_heads, 8)
        self.assertEqual(attn.n_kv_heads, 4)
        self.assertEqual(attn.n_local_heads, 8)
        self.assertEqual(attn.n_local_kv_heads, 4)
        self.assertEqual(attn.head_dim, 64)
        self.assertEqual(attn.dim, 512)
        self.assertEqual(attn.mask.shape, (16, 16))  # Causal mask shape check
        self.assertTrue(attn.use_kv_cache)

        if attn.use_kv_cache:
            self.assertIsInstance(attn.kv_cache, KVCache)
        self.assertIsInstance(attn.SDPA, SDPA)

    def test_attentionmha_forward(self):
        args = self.create_mock_args()
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)

        bsz, seqlen, dim = 2, 4, args.dim
        x = torch.randn(bsz, seqlen, dim)
        freqs_cos = torch.randn(seqlen, args.head_dim // 2)
        freqs_sin = torch.randn(seqlen, args.head_dim // 2)
        input_pos = torch.tensor([0, 1, 2, 3])

        output, _ = attn.forward(x, freqs_cos, freqs_sin, input_pos=input_pos)

        self.assertEqual(output.shape, (bsz, seqlen, dim))

    def test_attentionmha_forward_no_kv_cache(self):
        args = self.create_mock_args()
        args.use_kv_cache = False  # Disable KV cache for this test
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)

        bsz, seqlen, dim = 2, 4, args.dim
        x = torch.randn(bsz, seqlen, dim)
        freqs_cos = torch.randn(seqlen, args.head_dim // 2)
        freqs_sin = torch.randn(seqlen, args.head_dim // 2)

        output, _ = attn.forward(x, freqs_cos, freqs_sin)

        self.assertEqual(output.shape, (bsz, seqlen, dim))

    def test_attentionmha_invalid_kv_cache(self):
        args = self.create_mock_args()
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)

        bsz, seqlen, dim = 2, 4, args.dim
        x = torch.randn(bsz, seqlen, dim)
        freqs_cos = torch.randn(seqlen, args.head_dim // 2)
        freqs_sin = torch.randn(seqlen, args.head_dim // 2)

        # No input_pos provided, should raise assertion error
        with self.assertRaises(AssertionError):
            attn.forward(x, freqs_cos, freqs_sin)
