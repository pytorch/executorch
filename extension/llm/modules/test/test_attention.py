# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge

from executorch.extension.llm.modules.attention import (
    MultiHeadAttention as ETMultiHeadAttention,
)
from executorch.runtime import Runtime
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules.attention import MultiHeadAttention as TTMultiHeadAttention


torch.manual_seed(0)


class AttentionTest(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Constants
        self.embed_dim = 2048
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 64
        self.max_seq_len = 128
        self.rope_base = 500_000
        self.scale_factor = 32

        # Module dependency injections.
        self.q_proj = torch.nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            self.embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            self.embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.output_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.pos_embeddings = Llama3ScaledRoPE(
            dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.rope_base,
            scale_factor=self.scale_factor,
        )

        # Original TorchTune reference module to test accuracy against.
        self.tt_mha = TTMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            output_proj=self.output_proj,
            pos_embeddings=self.pos_embeddings,
            max_seq_len=self.max_seq_len,
        )

        # Source transformed module that we are testing.
        self.et_mha = ETMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            output_proj=self.output_proj,
            pos_embeddings=self.pos_embeddings,
            max_seq_len=self.max_seq_len,
        )

        # Common inputs.
        seq_len = 10
        self.x = torch.randn(1, seq_len, self.embed_dim)
        self.input_pos = torch.arange(seq_len).unsqueeze(0)  # shape [1, seq_len]
        seq_len_dim = torch.export.Dim("seq_len", min=1, max=100)
        self.dynamic_shapes = (
            {0: torch.export.Dim.STATIC, 1: seq_len_dim, 2: torch.export.Dim.STATIC},
            {0: torch.export.Dim.STATIC, 1: seq_len_dim, 2: torch.export.Dim.STATIC},
            {0: torch.export.Dim.STATIC, 1: seq_len_dim},
        )

    def test_attention_eager(self):
        et_res = self.et_mha(self.x, self.x)  # Self attention.
        tt_res = self.tt_mha(self.x, self.x)  # Self attention.

        self.assertTrue(torch.allclose(et_res, tt_res))

        # test with kv cache
        self.et_mha.setup_cache(1, dtype=torch.float32, max_seq_len=20)
        self.tt_mha.setup_cache(1, dtype=torch.float32, max_seq_len=20)

        et_res = self.et_mha(self.x, self.x)  # Self attention.
        tt_res = self.tt_mha(self.x, self.x)  # Self attention.

        self.assertTrue(torch.allclose(et_res, tt_res))
        self.et_mha.reset_cache()
        self.tt_mha.reset_cache()

        et_res = self.et_mha(
            self.x, self.x, input_pos=self.input_pos
        )  # Self attention with input pos.
        tt_res = self.tt_mha(
            self.x, self.x, input_pos=self.input_pos
        )  # Self attention with input pos.

        self.assertTrue(torch.allclose(et_res, tt_res))

        # test kv cache read. Input pos can be [10, 11, ..., 19]
        next_input_pos = torch.arange(10, 20).unsqueeze(0)
        et_res = self.et_mha(
            self.x, self.x, input_pos=next_input_pos
        )  # Self attention with input pos.
        tt_res = self.tt_mha(
            self.x, self.x, input_pos=next_input_pos
        )  # Self attention with input pos.
        self.assertTrue(torch.allclose(et_res, tt_res))

    def test_attention_export(self):
        # Self attention.
        et_mha_ep = torch.export.export(
            self.et_mha,
            (self.x, self.x),
            kwargs={"input_pos": self.input_pos},
            dynamic_shapes=self.dynamic_shapes,
        )
        et_res = et_mha_ep.module()(self.x, self.x, input_pos=self.input_pos)
        tt_res = self.tt_mha(self.x, self.x, input_pos=self.input_pos)
        self.assertTrue(torch.allclose(et_res, tt_res))

        # TODO: KV cache.

    def test_attention_aoti(self):
        # TODO.
        pass

    def test_attention_executorch(self):
        # Self attention.
        et_mha_ep = torch.export.export(
            self.et_mha,
            (self.x, self.x),
            kwargs={"input_pos": self.input_pos},
            dynamic_shapes=self.dynamic_shapes,
        )
        et_program = to_edge(
            et_mha_ep,
            compile_config=EdgeCompileConfig(),
        ).to_executorch()
        runtime = Runtime.get()
        program = runtime.load_program(et_program.buffer)
        method = program.load_method("forward")
        et_res = method.execute((self.x, self.x, self.input_pos))
        tt_res = self.tt_mha(self.x, self.x, input_pos=self.input_pos)

        self.assertTrue(torch.allclose(et_res[0], tt_res, atol=1e-06))

        # TODO: KV cache.
