# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase


class CortexMScaledAttentionDiv(torch.nn.Module):
    """``softmax(bmm(q, k^T) / sqrt(d))`` -- the attention-score scale is an fp32
    ``aten.div.Tensor`` by a constant that otherwise stays between the QK^T bmm
    and softmax. It must fold into the softmax-input quantize scale
    (``quantize(x / c, S) == quantize(x, S*c)``) so no fp32 div remains and the
    chain lowers to ``cortex_m.quantized_batch_matmul`` + ``cortex_m.softmax``.
    """

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_div_Tensor": 0,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 0,
    }

    def forward(self, q, k):
        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return torch.softmax(scores, dim=-1)


class CortexMScaledAttentionMul(torch.nn.Module):
    """Same, but the scale is applied as ``* (1/sqrt(d))`` -- an
    ``aten.mul.Tensor`` by a constant, folded via ``quantize(x*c, S) ==
    quantize(x, S/c)``.
    """

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_div_Tensor": 0,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 0,
    }

    def forward(self, q, k):
        scores = torch.bmm(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
        return torch.softmax(scores, dim=-1)


test_cases = {
    "scaled_attn_div": McuTestCase(
        CortexMScaledAttentionDiv(),
        (torch.rand(1, 8, 16), torch.rand(1, 8, 16)),
    ),
    "scaled_attn_mul": McuTestCase(
        CortexMScaledAttentionMul(),
        (torch.rand(1, 8, 16), torch.rand(1, 8, 16)),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_attention_scale(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )
