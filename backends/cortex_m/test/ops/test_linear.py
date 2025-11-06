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


class CortexMLinear(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class CortexMLinearX3(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 7,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 3,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear(x)
        x = self.linear(x)
        return x


class CortexMLinearBias(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear(x)


test_cases = {
    "linear_rank1": McuTestCase(
        model=CortexMLinear(1, 2),
        example_inputs=(torch.Tensor([1]),),
    ),
    "linear_rank2_pos": McuTestCase(
        model=CortexMLinear(1, 2),
        example_inputs=(ramp_tensor(-1, 1, (1, 1)),),
    ),
    "linear_rank3_neg": McuTestCase(
        model=CortexMLinear(5, 3),
        example_inputs=(ramp_tensor(-40, 0, (4, 2, 5)),),
    ),
    "linear_rank4": McuTestCase(
        model=CortexMLinear(16, 32),
        example_inputs=(ramp_tensor(-100, 100, (2, 1, 2, 16)),),
    ),
    "linear_rank5": McuTestCase(
        model=CortexMLinear(4, 3),
        example_inputs=(ramp_tensor(-2, 2, (5, 2, 1, 2, 4)),),
    ),
    "linear_bias": McuTestCase(
        model=CortexMLinearBias(61, 37),
        example_inputs=(ramp_tensor(0, 10, (8, 61)),),
    ),
    "linear_x3": McuTestCase(
        model=CortexMLinearX3(4, 4),
        example_inputs=(ramp_tensor(0, 10, (2, 4)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_linear(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_linear(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
