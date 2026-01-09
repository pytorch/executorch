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


class CortexMAvgPool2d(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(
        self, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=False
    ):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(
            kernel_size,
            stride,
            padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def forward(self, x):  # noqa: D102
        return self.pool(x)


# Prepare test cases: simple 2x2 pool on 4x4, and 3x3 stride 1 on 3x3
test_cases = {
    "avgpool_2x2": McuTestCase(
        CortexMAvgPool2d(kernel_size=2, stride=2), (ramp_tensor(0, 15, (1, 1, 4, 4)),)
    ),
    "avgpool_3x3_s1": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=1, padding=1),
        (ramp_tensor(0, 8, (1, 1, 3, 3)),),
    ),
    # additional pooling configurations: padding, stride, ceil_mode, count_include_pad
    "avgpool_2x2_pad1": McuTestCase(
        CortexMAvgPool2d(kernel_size=2, stride=2, padding=1),
        (ramp_tensor(0, 24, (1, 1, 5, 5)),),
    ),
    "avgpool_3x3_s2_pad1": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=2, padding=1),
        (ramp_tensor(0, 15, (1, 1, 4, 4)),),
    ),
}

test_cases_fp = {
    "avgpool_3x3_s2_pad1_ceil": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        (ramp_tensor(0, 15, (1, 1, 4, 4)),),
    ),
    "avgpool_3x3_s2_pad1_countinc": McuTestCase(
        CortexMAvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=True),
        (ramp_tensor(0, 15, (1, 1, 4, 4)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_avg_pool2d(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases_fp)
def test_dialect_avg_pool2d_fp(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1},
        {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1},
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_avg_pool2d(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
