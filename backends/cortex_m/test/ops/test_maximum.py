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


class CortexMTensorMaximum(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x, y):
        return torch.maximum(x, y)


test_cases = {
    "tensor_small": McuTestCase(
        CortexMTensorMaximum(),
        (
            torch.tensor([[1.0, -2.0], [3.5, -4.5]]),
            torch.tensor([[0.5, -1.0], [4.0, -3.5]]),
        ),
    ),
    "tensor_rand": McuTestCase(
        CortexMTensorMaximum(),
        (
            torch.rand(2, 2, 2) * 4 - 2,
            torch.rand(2, 2, 2) * 4 - 2,
        ),
    ),
    "broadcast": McuTestCase(
        CortexMTensorMaximum(),
        (
            ramp_tensor(-2, 2, (2, 1, 2)),
            ramp_tensor(-3, 3, (1, 2, 1)),
        ),
    ),
    "broadcast_rank4": McuTestCase(
        CortexMTensorMaximum(),
        (
            ramp_tensor(-4, 4, (1, 2, 3, 1)),
            ramp_tensor(-6, 6, (4, 1, 1, 3)),
        ),
    ),
    "broadcast_scalar": McuTestCase(
        CortexMTensorMaximum(),
        (
            torch.tensor(1.0),
            ramp_tensor(-6, 6, (4, 1, 1, 3)),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_maximum(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


@parametrize("test_case", test_cases)
def test_implementation_maximum(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
