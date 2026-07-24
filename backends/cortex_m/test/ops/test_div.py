# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT16_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.test.harness.stages import Quantize


class _CortexMInt16Quantize(Quantize):
    """Quantize stage that drives the Cortex-M quantizer with int16 activations."""

    def __init__(self) -> None:
        super().__init__(CortexMQuantizer(per_tensor_config=INT16_PER_TENSOR_CONFIG))


class CortexMSelfDiv(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_div_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x):
        return x / x


class CortexMTensorDiv(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_div_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def forward(self, x, y):
        return x / y


# Divisors are kept strictly positive so the quantized denominator never lands
# on its zero point (which the kernel maps to a 0 quotient).
test_cases = {
    "self_rank_1": McuTestCase(
        CortexMSelfDiv(),
        (ramp_tensor(1, 5, (10,)),),
    ),
    "self_rank_4": McuTestCase(
        CortexMSelfDiv(),
        (ramp_tensor(1, 5, (2, 2, 2, 2)),),
    ),
    "tensor_pos": McuTestCase(
        CortexMTensorDiv(),
        (ramp_tensor(1, 10, (8,)), ramp_tensor(1, 5, (8,))),
    ),
    "tensor_neg_num": McuTestCase(
        CortexMTensorDiv(),
        (ramp_tensor(-10, -1, (8,)), ramp_tensor(1, 5, (8,))),
    ),
    "tensor_rank_4": McuTestCase(
        CortexMTensorDiv(),
        (
            ramp_tensor(-8, 8, (2, 3, 4, 4)),
            ramp_tensor(1, 5, (2, 3, 4, 4)),
        ),
    ),
    "broadcast_1": McuTestCase(
        CortexMTensorDiv(),
        (ramp_tensor(1, 5, (1,)), ramp_tensor(1, 5, (2, 2, 2, 2))),
    ),
    "broadcast_2": McuTestCase(
        CortexMTensorDiv(),
        (ramp_tensor(1, 5, (2, 2, 2, 2)), ramp_tensor(1, 5, (1,))),
    ),
}


xfail_cases_dialect: dict[str, xfail_type] = {
    # The Cortex-M quantizer refuses to quantize divisions that require
    # broadcasting, so the div op is not replaced by a cortex_m implementation.
    "broadcast_1": "Broadcasting is not supported in Cortex-M backend",
    "broadcast_2": "Broadcasting is not supported in Cortex-M backend",
}


@parametrize("test_case", test_cases, xfails=xfail_cases_dialect)
def test_dialect_div(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize(
    "test_case",
    test_cases,
    xfails=xfail_cases_dialect,
)
def test_implementation_div(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)


@parametrize("test_case", test_cases, xfails=xfail_cases_dialect)
def test_dialect_div_int16(test_case, cortex_m_target):
    # Same op counts as the int8 flow — the graph topology is identical; only the
    # quantize/dequantize dtype (and the div kernel's clamp range) change.
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.quantize(_CortexMInt16Quantize())
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(
        inputs=tester.example_inputs, qtol=1, atol=1e-03
    )


@parametrize("test_case", test_cases, xfails=xfail_cases_dialect)
def test_implementation_div_int16(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.quantize(_CortexMInt16Quantize())
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=tester.example_inputs, qtol=1, atol=1e-03
    )
