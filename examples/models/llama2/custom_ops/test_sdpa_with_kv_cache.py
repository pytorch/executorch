# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from .sdpa_with_kv_cache import custom_ops_lib  # noqa


class SDPATest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.k_cache = torch.zeros((1, 5, 8, 4))
        self.v_cache = torch.zeros((1, 5, 8, 4))
        self.mask = torch.full(
            (5, 5),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)

    def _sdpa_with_kv_cache_ref(self, q, k, v, k_cache, v_cache, mask, start_pos):
        print(f"at start_pos:{start_pos}")
        print(q)
        print(k)
        print(v)
        attn_mask = mask[start_pos].view((1, -1))
        attn_mask = attn_mask[:, : start_pos + 1]
        q = q.transpose(1, 2)
        k_cache[:, start_pos] = k
        v_cache[:, start_pos] = v
        sliced_k_cache = k_cache[:, : start_pos + 1, :, :]
        sliced_v_cache = v_cache[:, : start_pos + 1, :, :]
        sliced_k_cache = sliced_k_cache.transpose(1, 2)
        sliced_v_cache = sliced_v_cache.transpose(1, 2)
        # print(sliced_k_cache.size())
        # print(torch.matmul(q, sliced_k_cache.transpose(2, 3)))
        # print("q @ k")
        # qk = torch.matmul(q, sliced_k_cache.transpose(2, 3))
        # qk_softmax = torch.softmax(qk, dim=-1)
        # qkv = torch.matmul(qk_softmax, sliced_v_cache)
        # print(qk)
        # print(qk_softmax)
        # print(qkv)
        out = F.scaled_dot_product_attention(
            q, sliced_k_cache, sliced_v_cache, attn_mask=attn_mask
        )
        out = out.transpose(1, 2)
        print(out)
        print(f"-------- start pos {start_pos} done -----")
        return out

    def test_sdpa_with_cache_no_mqa_1(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))
        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 0
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 0, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_no_mqa_2(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))

        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 1
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 1, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_no_mqa_3(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))

        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 2
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 2, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_no_mqa_4(self):
        q = torch.rand((1, 1, 8, 4))
        k = torch.rand((1, 1, 8, 4))
        v = torch.rand((1, 1, 8, 4))

        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 3
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 3, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))


class SDPATestWithMQA(unittest.TestCase):

    def setup_caches(self):
        self.k_cache = torch.zeros((1, 5, self.n_heads_kv, 4))
        self.v_cache = torch.zeros((1, 5, self.n_heads_kv, 4))

    def setUp(self):
        torch.manual_seed(42)
        self.n_heads_kv = 4
        self.n_heads_q = 8
        self.setup_caches()
        self.mask = torch.full(
            (5, 5),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)

    def _sdpa_with_kv_cache_ref(self, q, k, v, k_cache, v_cache, mask, start_pos):
        print(f"at start_pos:{start_pos}")
        print(q)
        print(k)
        print(v)
        attn_mask = mask[start_pos].view((1, -1))
        attn_mask = attn_mask[:, : start_pos + 1]
        q = q.transpose(1, 2)
        k_cache[:, start_pos] = k
        v_cache[:, start_pos] = v
        sliced_k_cache = k_cache[:, : start_pos + 1, :, :]
        sliced_v_cache = v_cache[:, : start_pos + 1, :, :]
        sliced_k_cache = sliced_k_cache.transpose(1, 2)
        sliced_v_cache = sliced_v_cache.transpose(1, 2)
        # print(sliced_k_cache.size())
        # print(torch.matmul(q, sliced_k_cache.transpose(2, 3)))
        # print("q @ k")
        # qk = torch.matmul(q, sliced_k_cache.transpose(2, 3))
        # qk_softmax = torch.softmax(qk, dim=-1)
        # qkv = torch.matmul(qk_softmax, sliced_v_cache)
        # print(qk)
        # print(qk_softmax)
        # print(qkv)
        num_heads_q = q.size(1)
        num_heads_kv = sliced_k_cache.size(1)
        if num_heads_q != num_heads_kv:
            assert (
                num_heads_q % num_heads_kv == 0
            ), f"{num_heads_q} not divisible by {num_heads_kv}"
        n_reps = num_heads_q // num_heads_kv
        if n_reps > 1:
            sliced_k_cache = sliced_k_cache.repeat_interleave(n_reps, dim=1)
            sliced_v_cache = sliced_v_cache.repeat_interleave(n_reps, dim=1)
        out = F.scaled_dot_product_attention(
            q, sliced_k_cache, sliced_v_cache, attn_mask=attn_mask
        )
        out = out.transpose(1, 2)
        print(out)
        print(f"-------- start pos {start_pos} done -----")
        return out

    def test_sdpa_with_cache_mqa_1(self):
        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_kv, 4))
        v = torch.rand((1, 1, self.n_heads_kv, 4))
        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 0
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 0, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_mqa_2(self):
        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_kv, 4))
        v = torch.rand((1, 1, self.n_heads_kv, 4))
        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 1
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 1, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))

    def test_sdpa_with_cache_mqa_3(self):
        self.n_heads_q = 14
        self.n_heads_kv = 7
        self.setup_caches()
        q = torch.rand((1, 1, self.n_heads_q, 4))
        k = torch.rand((1, 1, self.n_heads_kv, 4))
        v = torch.rand((1, 1, self.n_heads_kv, 4))
        ref_output = self._sdpa_with_kv_cache_ref(
            q, k, v, self.k_cache, self.v_cache, self.mask, 1
        )
        op_output = torch.ops.llama.sdpa_with_kv_cache(
            q, k, v, self.k_cache, self.v_cache, 1, 1, None, 0, False
        )
        self.assertTrue(torch.allclose(ref_output, op_output))
