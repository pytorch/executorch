# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.cortex_m_tester import CortexMTester, McuTestCase
from executorch.backends.test.suite.operators.test_add import Model, ModelAlpha


class SelfAdd(torch.nn.Module):
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


class ScalarAdd(Model):
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


class TensorAdd(Model):
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


class AlphaAdd(ModelAlpha):
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
        SelfAdd(),
        (10.0,),
    ),
    "self_rank_1": McuTestCase(
        SelfAdd(),
        (torch.linspace(-5, 5, 10),),
    ),
    "self_rank_2_pos": McuTestCase(
        SelfAdd(),
        (torch.linspace(0, 1000, 10).reshape((10, 1)),),
    ),
    "self_rank_3_neg": McuTestCase(
        SelfAdd(),
        (torch.linspace(-100, 0, 8).reshape((2, 2, 2)),),
    ),
    "self_rank_4_small": McuTestCase(
        SelfAdd(),
        (torch.linspace(-0.1, 0.1, 16).reshape(2, 2, 2, 2),),
    ),
    "self_rank_5": McuTestCase(
        SelfAdd(),
        (torch.linspace(-5, 5, 32).reshape(2, 2, 2, 2, 2),),
    ),
    "scalar_scalar": McuTestCase(
        ScalarAdd(),
        (-0.5, 1.0),
    ),
    "tensor_scalar": McuTestCase(
        ScalarAdd(),
        (torch.ones(2, 2), 1.0),
    ),
    "scalar_tensor": McuTestCase(
        ScalarAdd(),
        (1000.0, torch.ones(2, 2)),
    ),
    "broadcast_1": McuTestCase(
        TensorAdd(),
        (torch.ones(1), torch.ones(2, 2, 2, 2)),
    ),
    "broadcast_2": McuTestCase(
        TensorAdd(),
        (torch.ones((2, 1, 1, 1)), torch.ones(1)),
    ),
    "broadcast_3": McuTestCase(
        TensorAdd(),
        (
            torch.linspace(-2, 2, 4).reshape(2, 1, 2, 1),
            torch.linspace(-5, 5, 4).reshape(1, 2, 1, 2),
        ),
    ),
    "alpha": McuTestCase(
        AlphaAdd(0.5),
        (
            torch.linspace(-10, 10, 20).reshape(4, 5),
            torch.linspace(-20, 20, 20).reshape(4, 5),
        ),
    ),
}


dialect_xfails = {
    "self_scalar": ("'float' object has no attribute 'fake_mode'", AttributeError),
    "self_rank_1": ("Output 0 does not match reference output", AssertionError),
    "self_rank_2_pos": ("Output 0 does not match reference output", AssertionError),
    "self_rank_3_neg": ("Output 0 does not match reference output", AssertionError),
    "self_rank_4_small": ("Output 0 does not match reference output", AssertionError),
    "self_rank_5": ("Output 0 does not match reference output", AssertionError),
    "scalar_scalar": ("'float' object has no attribute 'fake_mode'", AttributeError),
    "broadcast_3": ("Output 0 does not match reference output", AssertionError),
    "alpha": ("Expecting kwargs for aten op IR to be empty", AssertionError),
}


@parametrize("test_case", test_cases, xfails=dialect_xfails)
def test_dialect_add(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


implementation_xfails = {
    "self_scalar": ("'float' object has no attribute 'fake_mode'", AttributeError),
    "self_rank_1": ("Output 0 does not match reference output", AssertionError),
    "self_rank_2_pos": ("Output 0 does not match reference output", AssertionError),
    "self_rank_3_neg": ("Output 0 does not match reference output", AssertionError),
    "self_rank_4_small": ("Output 0 does not match reference output", AssertionError),
    "self_rank_5": ("Output 0 does not match reference output", AssertionError),
    "scalar_scalar": ("'float' object has no attribute 'fake_mode'", AttributeError),
    "tensor_scalar": ("Output 0 does not match reference output", AssertionError),
    "scalar_tensor": ("Output 0 does not match reference output", AssertionError),
    "broadcast_1": ("Output 0 does not match reference output", AssertionError),
    "broadcast_2": ("Output 0 does not match reference output", AssertionError),
    "broadcast_3": ("Output 0 does not match reference output", AssertionError),
    "alpha": ("Expecting kwargs for aten op IR to be empty", AssertionError),
}


@parametrize("test_case", test_cases, xfails=implementation_xfails)
def test_implementation_add(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
