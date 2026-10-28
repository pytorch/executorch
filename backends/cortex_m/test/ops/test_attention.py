# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase


class CortexMAttention(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 2,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(12, 36)

    def forward(self, x):
        qkv = self.qkv(x).reshape(1, 5, 3, 3, 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q * 0.5) @ k.transpose(-2, -1)
        return (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(1, 5, 12)


test_cases = {
    "qkv_attention": McuTestCase(CortexMAttention(), (torch.randn(1, 5, 12),)),
}


@parametrize("test_case", test_cases)
def test_dialect_attention(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=2,
    )


@parametrize("test_case", test_cases)
def test_implementation_attention(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=2)
