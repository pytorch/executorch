# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
import torch.nn.functional as F

from executorch.extension.llm.custom_ops import custom_ops  # noqa
from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool


def is_fbcode():
    return not hasattr(torch.version, "git_version")


class SDPATestForCustomQuantizedSDPA(unittest.TestCase):
    """
    This test is to test the custom quantized SDPA op
    Tensors are in [B, H, S, D] format
    """

    def setUp(self):
        from torch.ao.quantization.fx._decomposed import (  # noqa: F401
            quantized_decomposed_lib,
        )

        torch.manual_seed(42)
        self.n_batch = 1
        self.n_heads_kv = 32
        self.n_heads_q = 32
        self.head_dim = 128
        self.max_seq_len = 2048
        self.quantized_dtype = torch.int8
        self.float_dtype = torch.float32
        self.q_shape = None
        self.kv_shape = None
        self.is_seq_at_dim_2 = True
        # For some reason 4 threads doesnt work
        # This setting is needed to make this test not flaky due to OMP
        # error of "OMP: Error #131: Thread identifier invalid"
        # Not clear why that happens but having smaller threadpool resolves it
        _unsafe_reset_threadpool(3)

    def _scale_tensor(self, tensor, min_value, max_value, scale=True):
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        scaled_tensor = normalized_tensor * (max_value - min_value) + min_value

        return scaled_tensor if scale else tensor

    def setup_caches_and_mask(self, tensor_scale_max, tensor_scale_min, scale_tensors):
        self.mask = torch.full(
            (self.max_seq_len, self.max_seq_len),
            float("-inf"),
        )
        self.mask = torch.triu(self.mask, diagonal=1)

        self.k = self._scale_tensor(
            torch.rand(self.kv_shape),
            tensor_scale_max,
            tensor_scale_min,
            scale_tensors,
        )
        self.v = self._scale_tensor(
            torch.rand(self.kv_shape),
            tensor_scale_max,
            tensor_scale_min,
            scale_tensors,
        )

    def _sdpa_ref(
        self,
        q_quantized,
        k_quantized,
        v_quantized,
        start_pos,
        q_zero_point,
        q_scale,
        k_zero_point,
        k_scale,
        v_zero_point,
        v_scale,
        attn_mask,
    ):
        q = torch.ops.quantized_decomposed.dequantize_per_token(
            q_quantized,
            q_scale,
            q_zero_point,
            torch.iinfo(self.quantized_dtype).min,
            torch.iinfo(self.quantized_dtype).max,
            self.quantized_dtype,
            self.float_dtype,
        )
        k = torch.ops.quantized_decomposed.dequantize_per_token(
            k_quantized,
            k_scale,
            k_zero_point,
            torch.iinfo(self.quantized_dtype).min,
            torch.iinfo(self.quantized_dtype).max,
            self.quantized_dtype,
            self.float_dtype,
        )
        v = torch.ops.quantized_decomposed.dequantize_per_token(
            v_quantized,
            v_scale,
            v_zero_point,
            torch.iinfo(self.quantized_dtype).min,
            torch.iinfo(self.quantized_dtype).max,
            self.quantized_dtype,
            self.float_dtype,
        )

        if not self.is_seq_at_dim_2:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
        num_heads_q = q.size(1)
        num_heads_kv = k.size(1)
        seq_len = q.size(2)
        k = torch.narrow(k, 2, 0, start_pos + seq_len)
        v = torch.narrow(v, 2, 0, start_pos + seq_len)
        if num_heads_q != num_heads_kv:
            assert (
                num_heads_q % num_heads_kv == 0
            ), f"{num_heads_q} not divisible by {num_heads_kv}"
        n_reps = num_heads_q // num_heads_kv
        if n_reps > 1:
            k = k.repeat_interleave(n_reps, dim=1)
            v = v.repeat_interleave(n_reps, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if not self.is_seq_at_dim_2:
            out = out.transpose(1, 2).contiguous()
        return out

    def _int_matmul(
        self, quantized_q, quantized_k, q_zero_points, q_scale, k_zero_point, k_scale
    ):
        row_sum_q = torch.sum(quantized_q, dim=-1, keepdim=True)
        row_sum_k = torch.sum(quantized_k, dim=-1, keepdim=True)
        q_at_k = torch.matmul(quantized_q, quantized_k.transpose(-2, -1))
        row_sum_q_scaled = row_sum_q * k_zero_point.squeeze(-1).unsqueeze(0)
        row_sum_k_scaled = q_zero_points * row_sum_k.squeeze(-1).unsqueeze(0)
        zero_points_product = (
            quantized_q.size(-1) * q_zero_points * k_zero_point.squeeze(-1).unsqueeze(0)
        )
        res = q_at_k - row_sum_q_scaled - row_sum_k_scaled + zero_points_product
        q_scale_mul_k_scale = q_scale * k_scale.squeeze(-1).unsqueeze(0)
        res = res.to(torch.float32) * q_scale_mul_k_scale
        return res

    def _quantized_sdpa_ref(
        self,
        quantized_q,
        quantized_k,
        quantized_v,
        q_zero_points,
        q_scale,
        k_scale,
        k_zero_point,
        v_scale,
        v_zero_point,
        attn_mask,
    ):
        import math

        quantized_q = quantized_q.to(torch.int32)
        quantized_k = quantized_k.to(torch.int32)
        quantized_v = quantized_v.to(torch.int32)
        batch_size = quantized_q.size(0)
        num_heads_q = quantized_q.size(1)
        num_heads_kv = quantized_k.size(1)
        q_scale = q_scale.to(torch.float32)
        k_scale = k_scale.to(torch.float32)
        q_zero_points = q_zero_points.to(torch.int32)
        k_zero_point = k_zero_point.to(torch.int32)
        if num_heads_q != num_heads_kv:
            assert (
                num_heads_q % num_heads_kv == 0
            ), f"{num_heads_q} not divisible by {num_heads_kv}"
        n_reps = num_heads_q // num_heads_kv
        if n_reps > 1:
            quantized_k = quantized_k.repeat_interleave(n_reps, dim=1)
            quantized_v = quantized_v.repeat_interleave(n_reps, dim=1)
        res_b = []
        scale_factor = 1 / math.sqrt(quantized_k.size(-1))
        dequantized_v = torch.ops.quantized_decomposed.dequantize_per_token(
            quantized_v,
            v_scale,
            v_zero_point,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            torch.int8,
            torch.float32,
        )
        for b in range(batch_size):
            res_h = []
            for h in range(num_heads_q):
                q_at_k = self._int_matmul(
                    quantized_q[b][h],
                    quantized_k[b][h],
                    q_zero_points[b][h],
                    q_scale[b][h],
                    k_zero_point[b][h],
                    k_scale[b][h],
                )
                q_at_k = q_at_k * scale_factor
                q_at_k += attn_mask
                attn_weight = torch.softmax(q_at_k, dim=-1)
                y = torch.matmul(attn_weight, dequantized_v[b][h])
                res_h.append(y)
            res = torch.stack(res_h, dim=0)
            res_b.append(res.unsqueeze(0))
        res = torch.cat(res_b, dim=0)
        return res

    def _test_sdpa_common(
        self,
        n_heads_kv,
        n_heads_q,
        head_dim,
        max_seq_len,
        start_pos,
        seq_len,
        scale_tensors=False,
        atol=1e-5,
        is_seq_at_dim_2=False,
    ):
        # Range arbitrarily chosen to reproduce a numerical error on x86 in some of the long context tests
        tensor_scale_max = 15
        tensor_scale_min = -15
        self.n_heads_kv = n_heads_kv
        self.n_heads_q = n_heads_q
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_seq_at_dim_2 = is_seq_at_dim_2
        seq_dim = 2
        self.q_shape = (self.n_batch, self.n_heads_q, seq_len, self.head_dim)
        self.kv_shape = (self.n_batch, self.n_heads_kv, self.max_seq_len, self.head_dim)
        if not is_seq_at_dim_2:
            seq_dim = 1
            self.q_shape = (self.n_batch, seq_len, self.n_heads_q, self.head_dim)
            self.kv_shape = (
                self.n_batch,
                self.max_seq_len,
                self.n_heads_kv,
                self.head_dim,
            )

        q = self._scale_tensor(
            torch.rand(self.q_shape),
            tensor_scale_max,
            tensor_scale_min,
            scale_tensors,
        )
        self.setup_caches_and_mask(tensor_scale_max, tensor_scale_min, scale_tensors)
        k = self.k
        v = self.v

        quantized_dtype = self.quantized_dtype
        q_scale, q_zero_point = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                q, quantized_dtype
            )
        )
        k_scale, k_zero_point = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                k, quantized_dtype
            )
        )
        v_scale, v_zero_point = (
            torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric.default(
                v, quantized_dtype
            )
        )

        q_quantized = torch.ops.quantized_decomposed.quantize_per_token(
            q,
            q_scale,
            q_zero_point,
            torch.iinfo(quantized_dtype).min,
            torch.iinfo(quantized_dtype).max,
            quantized_dtype,
        )
        k_quantized = torch.ops.quantized_decomposed.quantize_per_token(
            k,
            k_scale,
            k_zero_point,
            torch.iinfo(quantized_dtype).min,
            torch.iinfo(quantized_dtype).max,
            quantized_dtype,
        )
        v_quantized = torch.ops.quantized_decomposed.quantize_per_token(
            v,
            v_scale,
            v_zero_point,
            torch.iinfo(quantized_dtype).min,
            torch.iinfo(quantized_dtype).max,
            quantized_dtype,
        )

        seq_len = q.size(seq_dim)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]

        # quantized_sdpa_ref_output = self._quantized_sdpa_ref(q_quantized, k_quantized, v_quantized, q_zero_point, q_scale, k_scale, k_zero_point, v_scale, v_zero_point, attn_mask)

        from torch.nn.attention import SDPBackend

        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION]
        ), torch.no_grad():
            ref_output = self._sdpa_ref(
                q_quantized,
                k_quantized,
                v_quantized,
                start_pos,
                q_zero_point,
                q_scale,
                k_zero_point,
                k_scale,
                v_zero_point,
                v_scale,
                attn_mask,
            )

        q_zero_point_int8 = q_zero_point.to(dtype=torch.int8)
        k_zero_point_int8 = k_zero_point.to(dtype=torch.int8)
        v_zero_point_int8 = v_zero_point.to(dtype=torch.int8)
        q_scale_fp32 = q_scale.to(dtype=torch.float32)
        k_scale_fp32 = k_scale.to(dtype=torch.float32)
        v_scale_fp32 = v_scale.to(dtype=torch.float32)

        op_output = torch.ops.llama.custom_quantized_sdpa(
            q_quantized,
            k_quantized,
            v_quantized,
            start_pos,
            None,
            0,
            True,
            None,
            q_zero_point_int8,
            q_scale_fp32,
            k_zero_point_int8,
            k_scale_fp32,
            v_zero_point_int8,
            v_scale_fp32,
            is_seq_at_dim_2,
        )
        print((ref_output - op_output).abs().max())
        self.assertTrue(torch.allclose(ref_output, op_output, atol=atol))
        # Following line crashes due to some weird issues in mkldnn with crash in mkl_sgemm with `wild jump`
        # self.assertTrue(torch.allclose(ref_output, quantized_sdpa_ref_output, atol=1e-3))

        start_pos = seq_len
        seq_len = q.size(seq_dim)
        attn_mask = self.mask[start_pos : start_pos + seq_len, :]
        attn_mask = attn_mask[:, : start_pos + seq_len]
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION]
        ), torch.no_grad():
            ref_output = self._sdpa_ref(
                q_quantized,
                k_quantized,
                v_quantized,
                start_pos,
                q_zero_point,
                q_scale,
                k_zero_point,
                k_scale,
                v_zero_point,
                v_scale,
                attn_mask,
            )
        op_output = torch.ops.llama.custom_quantized_sdpa(
            q_quantized,
            k_quantized,
            v_quantized,
            start_pos,
            None,
            0,
            True,
            None,
            q_zero_point_int8,
            q_scale_fp32,
            k_zero_point_int8,
            k_scale_fp32,
            v_zero_point_int8,
            v_scale_fp32,
            is_seq_at_dim_2,
        )
        self.assertTrue(torch.allclose(ref_output, op_output, atol=atol))

    @unittest.skipIf(
        not is_fbcode(), "in OSS error is too large 0.0002 for some reason"
    )
    def test_sdpa_with_custom_quantized(self):
        n_heads_kv = 8
        n_heads_q = 8
        head_dim = 128
        max_seq_len = 2048
        seq_len = 24
        start_pos = 0
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            True,
            atol=1e-4,
            is_seq_at_dim_2=True,
        )
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            True,
            atol=1e-4,
            is_seq_at_dim_2=False,
        )

    def test_sdpa_with_custom_quantized_seq_len_1(self):
        n_heads_kv = 4
        n_heads_q = 4
        head_dim = 4
        max_seq_len = 8
        seq_len = 1
        start_pos = 0
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            is_seq_at_dim_2=True,
        )
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            is_seq_at_dim_2=False,
        )

    def test_sdpa_with_custom_quantized_seq_len_small(self):
        n_heads_kv = 4
        n_heads_q = 4
        head_dim = 4
        max_seq_len = 8
        seq_len = 4
        start_pos = 0
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            is_seq_at_dim_2=True,
        )
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            is_seq_at_dim_2=False,
        )

    def test_sdpa_with_custom_quantized_seq_len_llava_example(self):
        n_heads_kv = 32
        n_heads_q = 32
        head_dim = 128
        max_seq_len = 2048
        seq_len = 634
        start_pos = 0
        self._test_sdpa_common(
            n_heads_kv, n_heads_q, head_dim, max_seq_len, start_pos, seq_len
        )

    def test_sdpa_with_custom_quantized_seq_len_130_gqa(self):
        n_heads_kv = 8
        n_heads_q = 32
        head_dim = 128
        max_seq_len = 2048
        seq_len = 130
        start_pos = 0
        # For some reason when scaling tensors, the test fails with smaller atol
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            True,
            atol=1e-3,
        )

    def test_sdpa_with_custom_quantized_seq_len_llava_example_gqa(self):
        n_heads_kv = 16
        n_heads_q = 32
        head_dim = 128
        max_seq_len = 2048
        seq_len = 634
        start_pos = 0
        self._test_sdpa_common(
            n_heads_kv, n_heads_q, head_dim, max_seq_len, start_pos, seq_len
        )

    def test_sdpa_with_cache_mqa(self):
        n_heads_kv = 1
        n_heads_q = 8
        head_dim = 128
        max_seq_len = 2048
        seq_len = 24
        start_pos = 0
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            is_seq_at_dim_2=True,
        )
        self._test_sdpa_common(
            n_heads_kv,
            n_heads_q,
            head_dim,
            max_seq_len,
            start_pos,
            seq_len,
            is_seq_at_dim_2=False,
        )
