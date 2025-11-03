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
from executorch.backends.test.suite.operators.test_add import Model, ModelAlpha


class CortexMSelfAdd(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x):
        return x + x


class CortexMScalarAdd(Model):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


class CortexMTensorAdd(Model):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


class CortexMAlphaAdd(ModelAlpha):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }


test_cases = {
    "self_scalar": McuTestCase(
        CortexMSelfAdd(),
        (10.0,),
    ),
    "self_rank_1": McuTestCase(
        CortexMSelfAdd(),
        (torch.linspace(-5, 5, 10),),
    ),
    "self_rank_2_pos": McuTestCase(
        CortexMSelfAdd(),
        (ramp_tensor(0, 1000, (10, 1)),),
    ),
    "self_rank_3_neg": McuTestCase(
        CortexMSelfAdd(),
        (ramp_tensor(-100, 0, (2, 2, 2)),),
    ),
    "self_rank_4_small": McuTestCase(
        CortexMSelfAdd(),
        (ramp_tensor(-0.1, 0.1, (2, 2, 2, 2)),),
    ),
    "self_rank_5": McuTestCase(
        CortexMSelfAdd(),
        (ramp_tensor(-5, 5, (2, 2, 2, 2, 2)),),
    ),
    "scalar_scalar": McuTestCase(
        CortexMScalarAdd(),
        (-0.5, 1.0),
    ),
    "tensor_scalar": McuTestCase(
        CortexMScalarAdd(),
        (torch.ones(2, 2), 1.0),
    ),
    "scalar_tensor": McuTestCase(
        CortexMScalarAdd(),
        (1000.0, torch.ones(2, 2)),
    ),
    "tensor_tensor": McuTestCase(
        CortexMTensorAdd(),
        (torch.rand(2, 2) * 10, torch.rand(2, 2)),
    ),
    "broadcast_1": McuTestCase(
        CortexMTensorAdd(),
        (torch.ones(1), torch.ones(2, 2, 2, 2)),
    ),
    "broadcast_2": McuTestCase(
        CortexMTensorAdd(),
        (torch.ones((2, 1, 1, 1)), torch.ones(1)),
    ),
    "broadcast_3": McuTestCase(
        CortexMTensorAdd(),
        (
            ramp_tensor(-2, 2, (2, 1, 2, 1)),
            ramp_tensor(-5, 5, (1, 2, 1, 2)),
        ),
    ),
    "alpha": McuTestCase(
        CortexMAlphaAdd(0.5),
        (
            ramp_tensor(-10, 10, (4, 5)),
            ramp_tensor(-20, 20, (4, 5)),
        ),
    ),
}


dialect_xfails = {
    "self_scalar": (
        "'float' object has not attribute 'fake_mode' - scalar only ops not supported.",
        AttributeError,
    ),
    "scalar_scalar": (
        "'float' object has not attribute 'fake_mode' - scalar only ops not supported.",
        AttributeError,
    ),
    "tensor_scalar": (
        "Expected to find 'executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default' but did not find it - broadcasting not supported.",
        RuntimeError,
    ),
    "scalar_tensor": (
        "Expected to find 'executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default' but did not find it - broadcasting not supported.",
        RuntimeError,
    ),
    "broadcast_1": (
        "Expected to find 'executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default' but did not find it - broadcasting not supported.",
        RuntimeError,
    ),
    "broadcast_2": (
        "Expected to find 'executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default' but did not find it - broadcasting not supported.",
        RuntimeError,
    ),
    "broadcast_3": (
        "Expected to find 'executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default' but did not find it - broadcasting not supported.",
        RuntimeError,
    ),
    "alpha": (
        "Expecting kwargs for aten op IR to be empty - alpha arg not supported.",
        AssertionError,
    ),
}


@parametrize("test_case", test_cases, xfails=dialect_xfails)
def test_dialect_add(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


implementation_xfails = {
    "self_scalar": (
        "'float' object has not attribute 'fake_mode' - scalar only ops not supported.",
        AttributeError,
    ),
    "scalar_scalar": (
        "'float' object has not attribute 'fake_mode' - scalar only ops not supported.",
        AttributeError,
    ),
    "tensor_scalar": (
        "Missing operator: [2] aten::add.out - broadcasting not supported.",
        RuntimeError,
    ),
    "scalar_tensor": (
        "Missing operator: [2] aten::add.out - broadcasting not supported.",
        RuntimeError,
    ),
    "broadcast_1": (
        "Missing operator: [2] aten::add.out - broadcasting not supported.",
        RuntimeError,
    ),
    "broadcast_2": (
        "Missing operator: [2] aten::add.out - broadcasting not supported.",
        RuntimeError,
    ),
    "broadcast_3": (
        "Missing operator: [2] aten::add.out - broadcasting not supported.",
        RuntimeError,
    ),
    "alpha": (
        "Expecting kwargs for aten op IR to be empty - alpha arg not supported.",
        AssertionError,
    ),
}


@parametrize("test_case", test_cases, xfails=implementation_xfails)
def test_implementation_add(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
