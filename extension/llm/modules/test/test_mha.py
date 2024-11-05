# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.extension.llm.modules.mha import (
    MultiHeadAttention as ETMultiHeadAttention,
)
from torchtune.modules.attention import MultiHeadAttention as TTMultiHeadAttention
from torchtune.modules.kv_cache import KVCache


torch.manual_seed(0)


class AttentionTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.embed_dim=2048
        self.num_heads=32
        self.num_kv_heads=8
        self.head_dim=64
        self.max_seq_len = 128
        self.tt_mha = TTMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q_proj=torch.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False),
            k_proj=torch.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False),
            v_proj=torch.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False),
            output_proj=torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            # pos_embeddings=rope,
            max_seq_len=self.max_seq_len,
            # attn_dropout=attn_dropout,
        )
        self.et_mha = ETMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q_proj=torch.nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False),
            k_proj=torch.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False),
            v_proj=torch.nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False),
            output_proj=torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            # pos_embeddings=rope,
            max_seq_len=self.max_seq_len,
            # attn_dropout=attn_dropout,
        )

    def test_self_attention_eager(self):
        seq_len = 10
        x = torch.randn(1, seq_len, self.embed_dim)
        et_res = self.et_mha(x, x) # Self attention.
        tt_res = self.tt_mha(x, x) # Self attention.
        
        self.assertTrue(torch.allclose(et_res, tt_res))

        # TODO: KV cache.
        # self.et_mha.setup_cache(1, dtype=torch.float16, max_seq_len=20)
        # self.tt_mha.setup_cache(1, dtype=torch.float16, max_seq_len=20)
        
        # et_res = self.et_mha(x, x) # Self attention.
        # tt_res = self.tt_mha(x, x) # Self attention.

        # self.assertTrue(torch.allclose(et_res, tt_res))

    def test_self_attention_export(self):
        seq_len = 10
        x = torch.randn(1, seq_len, self.embed_dim)
        seq_len_dim = torch.export.Dim("seq_len", min=1, max=100)
        dynamic_shapes = (
            {0: torch.export.Dim.STATIC, 1: seq_len_dim, 2: torch.export.Dim.STATIC},
            {0: torch.export.Dim.STATIC, 1: seq_len_dim, 2: torch.export.Dim.STATIC},
        )

        # Self attention.
        et_mha_ep = torch.export.export(
            self.et_mha,
            (x, x),
            kwargs=None,
            dynamic_shapes=dynamic_shapes,
        )
        et_res = et_mha_ep.module()(x, x)
        tt_res = self.tt_mha(x, x)
        self.assertTrue(torch.allclose(et_res, tt_res))
        
        # TODO: KV cache.

    def test_cross_attention_export(self):
        pass
