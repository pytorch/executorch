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
from executorch.backends.test.suite.operators.test_mul import Model


class CortexMSelfMul(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x):
        return x * x


class CortexMScalarMul(Model):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


class CortexMTensorMul(Model):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


test_cases = {
    "self_scalar": McuTestCase(
        CortexMSelfMul(),
        (10.0,),
    ),
    "self_rank_1": McuTestCase(
        CortexMSelfMul(),
        (ramp_tensor(-5, 5, (10,)),),
    ),
    "self_rank_2_pos": McuTestCase(
        CortexMSelfMul(),
        (ramp_tensor(0, 1000, (10, 1)),),
    ),
    "self_rank_3_neg": McuTestCase(
        CortexMSelfMul(),
        (ramp_tensor(-100, 0, (2, 2, 2)),),
    ),
    "self_rank_4_small": McuTestCase(
        CortexMSelfMul(),
        (ramp_tensor(-0.1, 0.1, (2, 2, 2, 2)),),
    ),
    "self_rank_5": McuTestCase(
        CortexMSelfMul(),
        (ramp_tensor(-5, 5, (2, 2, 2, 2, 2)),),
    ),
    "scalar_scalar": McuTestCase(
        CortexMScalarMul(),
        (-0.5, 1.0),
    ),
    "tensor_scalar": McuTestCase(
        CortexMScalarMul(),
        (torch.ones(1), 1.0),
    ),
    "scalar_tensor": McuTestCase(
        CortexMScalarMul(),
        (1000.0, torch.ones(1)),
    ),
    "broadcast_1": McuTestCase(
        CortexMTensorMul(),
        (torch.ones(1), torch.ones(2, 2, 2, 2)),
    ),
    "broadcast_2": McuTestCase(
        CortexMTensorMul(),
        (torch.ones((2, 1, 1, 1)), torch.ones(1)),
    ),
    "broadcast_3": McuTestCase(
        CortexMTensorMul(),
        (
            ramp_tensor(-2, 2, (2, 1, 2, 1)),
            ramp_tensor(-5, 5, (1, 2, 1, 2)),
        ),
    ),
}


xfail_cases = {
    "self_scalar": (
        "'float' object has not attribute 'fake_mode' - scalar only ops not supported.",
        AttributeError,
    ),
    "scalar_scalar": (
        "'float' object has not attribute 'fake_mode' - scalar only ops not supported.",
        AttributeError,
    ),
    "broadcast_1": "Broadcasting not yet supported in Cortex-M backend",
    "broadcast_2": "Broadcasting not yet supported in Cortex-M backend",
    "broadcast_3": "Broadcasting not yet supported in Cortex-M backend",
}


@parametrize("test_case", test_cases, xfails=xfail_cases)
def test_dialect_mul(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases, xfails=xfail_cases)
def test_implementation_mul(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
