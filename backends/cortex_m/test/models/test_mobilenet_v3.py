# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize

from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.backends.test.harness.stages import StageType
from torchvision import models


ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 6,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 52,
    "executorch_exir_dialects_edge__ops_aten_hardswish_default": 19,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 14,
    "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 10,
    "executorch_exir_dialects_edge__ops_aten_hardsigmoid_default": 9,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 9,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 6,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 104,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 120,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 101,
}

ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 6,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 41,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 11,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 28,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 10,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
}

# Use bigger sample set for calibration.
calibration_samples = [
    (torch.randn(1, 3, 232, 232).to(memory_format=torch.channels_last),)
    for i in (range(100))
]

test_cases = {
    "mobilenet_v3_small": McuTestCase(
        model=models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        ),
        example_inputs=lambda: (
            torch.randn(1, 3, 232, 232).to(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize(
    "test_case",
    test_cases,
    xfails={
        "mobilenet_v3_small": "MLETORCH-1821 - Investigate mobilenet_v3_small flakyness"
    },
    strict=False,
)
def test_dialect_mv3(test_case):
    inputs = test_case.example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=20,
        calibration_samples=calibration_samples,
    )

    # Since qtol is high, also assert that top 1 output matches between reference quantized model and lowered model
    ref = tester.get_artifact(StageType.EXPORT).module()(*inputs)
    result = tester.stages[StageType.RUN_PASSES].run_artifact(inputs)
    assert torch.argmax(ref) == torch.argmax(result), "Mismatch in model outputs"


@parametrize(
    "test_case",
    test_cases,
    xfails={
        "mobilenet_v3_small": "MLETORCH-1821 - Investigate mobilenet_v3_small flakyness"
    },
    strict=False,
)
def test_implementation_mv3(test_case):
    inputs = test_case.example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_implementation(
        qtol=20,
        calibration_samples=calibration_samples,
    )

    # Since qtol is high, also assert that top 1 output matches between reference quantized model and lowered model
    ref = tester.get_artifact(StageType.EXPORT).module()(*inputs)
    result = tester.stages[StageType.SERIALIZE].run_artifact(inputs)
    assert torch.argmax(ref) == torch.argmax(result[0]), "Mismatch in model outputs"
