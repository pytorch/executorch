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
    "tensor_scalar": McuTestCase(
        CortexMScalarAdd(),
        (torch.ones(1), 1.1),
    ),
    "scalar_tensor": McuTestCase(
        CortexMScalarAdd(),
        (1000.1, torch.ones(1)),
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
    "broadcast_channels_1": McuTestCase(
        CortexMTensorAdd(),
        (
            ramp_tensor(-2, 2, (1, 8, 1, 1)).to(memory_format=torch.channels_last),
            ramp_tensor(-5, 5, (1, 8, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "broadcast_channels_2": McuTestCase(
        CortexMTensorAdd(),
        (
            ramp_tensor(-5, 5, (2, 8, 5, 5)).to(memory_format=torch.channels_last),
            ramp_tensor(-2, 2, (1, 8, 1, 1)).to(memory_format=torch.channels_last),
        ),
    ),
    "broadcast_channels_continous": McuTestCase(
        CortexMTensorAdd(),
        (
            ramp_tensor(-5, 5, (2, 8, 5, 5)),
            ramp_tensor(-2, 2, (1, 8, 1, 1)),
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


xfails_implementation = {
    "alpha": (
        "Expecting kwargs for aten op IR to be empty - alpha arg not supported.",
        AssertionError,
    ),
}
xfails_dialect = xfails_implementation | {
    # Cortex-M quantizer will not quantize additions that require broadcasting
    # leading to the add op not being replaced by a cortex-m specific implementation
    "broadcast_1": "Broadcasting is not supported in Cortex-M backend",
    "broadcast_2": "Broadcasting is not supported in Cortex-M backend",
    "broadcast_3": "Broadcasting is not supported in Cortex-M backend",
    "broadcast_channels_continous": "Broadcasting channels is not supported in continous memory_format in Cortex-M backend.",
}


@parametrize("test_case", test_cases, xfails=xfails_dialect)
def test_dialect_add(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


@parametrize("test_case", test_cases, xfails=xfails_implementation)
def test_implementation_add(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
