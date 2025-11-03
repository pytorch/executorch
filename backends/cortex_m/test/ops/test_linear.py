# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMMm(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y)

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mm_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


class CortexMBmm(torch.nn.Module):
    def forward(self, x, y):
        return torch.bmm(x, y)

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


class CortexMAddmm(torch.nn.Module):
    def forward(self, x, y, z, alpha=None, beta=None):
        return torch.addmm(beta, x, alpha, y, z)

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_addmm_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


class CortexMAt(CortexMMm):
    def forward(self, x, y):
        return x @ y


class CortexMMatmul(CortexMMm):
    def forward(self, x, y):
        return torch.matmul(x, y)


class CortexMLinear(CortexMMatmul):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=False)

    def forward(self, x):
        return self.linear(x)


class CortexMLinearBias(CortexMAddmm):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


test_cases = {
    "mm": McuTestCase(
        model=CortexMMm(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 16)),
            ramp_tensor(0, 10, (16, 16)),
        ),
    ),
    "bmm": McuTestCase(
        model=CortexMBmm(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 16, 16)),
            ramp_tensor(0, 10, (1, 16, 16)),
        ),
    ),
    "addmm": McuTestCase(
        model=CortexMAddmm(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 16)),
            ramp_tensor(0, 10, (16, 16)),
            ramp_tensor(0, 10, (16, 16)),
            2,
            4,
        ),
    ),
    "addmm_scalars": McuTestCase(
        model=CortexMAddmm(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 16)),
            ramp_tensor(0, 10, (16, 16)),
            ramp_tensor(0, 10, (16, 16)),
        ),
    ),
    "@-operator": McuTestCase(
        model=CortexMAt(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 16)),
            ramp_tensor(0, 10, (16, 16)),
        ),
    ),
    "matmul": McuTestCase(
        model=CortexMMatmul(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 16)),
            ramp_tensor(0, 10, (16, 16)),
        ),
    ),
    "linear_rank1": McuTestCase(
        model=CortexMLinear(2, 3),
        example_inputs=(ramp_tensor(-1, 1, (2,)),),
    ),
    "linear_rank2_pos": McuTestCase(
        model=CortexMLinear(8, 3),
        example_inputs=(ramp_tensor(0, 10, (2, 8)),),
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
}


@pytest.mark.skip(
    reason="Skipping until the quantized_linear_fusion_pass is updated to work with non decomposed linear ops."
)
def test_dialect_linear(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


@pytest.mark.skip(
    reason="Skipping until the quantized_linear_fusion_pass is updated to work with non decomposed linear ops."
)
def test_implementation_linear(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
