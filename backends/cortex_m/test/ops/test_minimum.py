# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMSelfMinimum(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x):
        return torch.minimum(x, x)


class CortexMTensorMinimum(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x, y):
        return torch.minimum(x, y)


test_cases = {
    "self_rank_1": McuTestCase(
        CortexMSelfMinimum(),
        (ramp_tensor(-5, 5, (10,)),),
    ),
    "self_rank_3": McuTestCase(
        CortexMSelfMinimum(),
        (ramp_tensor(-10, 10, (2, 3, 4)),),
    ),
    "tensor_small": McuTestCase(
        CortexMTensorMinimum(),
        (
            torch.tensor([[1.0, -2.0], [3.5, -4.5]]),
            torch.tensor([[0.5, -3.0], [3.0, -4.0]]),
        ),
    ),
    "tensor_rand": McuTestCase(
        CortexMTensorMinimum(),
        (
            torch.rand(2, 2, 2) * 4 - 2,
            torch.rand(2, 2, 2) * 4 - 2,
        ),
    ),
    "broadcast": McuTestCase(
        CortexMTensorMinimum(),
        (
            ramp_tensor(-2, 2, (2, 1, 2)),
            ramp_tensor(-3, 3, (1, 2, 1)),
        ),
    ),
    "broadcast_rank4": McuTestCase(
        CortexMTensorMinimum(),
        (
            ramp_tensor(-4, 4, (1, 2, 3, 1)),
            ramp_tensor(-6, 6, (4, 1, 1, 3)),
        ),
    ),
}


xfails = {}


@parametrize("test_case", test_cases, xfails=xfails)
def test_dialect_minimum(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


@parametrize("test_case", test_cases, xfails=xfails)
def test_implementation_minimum(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
