# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize

from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.backends.test.harness.stages import StageType
from torchvision import models


ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 10,
    "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 52,
    "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 35,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 104,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 79,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 67,
}

ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 10,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 35,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 17,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
}

# Use larger sample set for calibration to get better quantization
calibration_samples = [
    (torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last),)
    for _ in range(100)
]

test_cases = {
    "mobilenet_v2": McuTestCase(
        model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        example_inputs=lambda: (
            torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_mv2(test_case):
    inputs = test_case.example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
        calibration_samples=calibration_samples,
    )

    # assert that top 1 output matches
    ref = tester.get_artifact(StageType.EXPORT).module()(*inputs)
    result = tester.stages[StageType.RUN_PASSES].run_artifact(inputs)
    assert torch.argmax(ref) == torch.argmax(result), "Mismatch in model outputs"


@parametrize(
    "test_case",
    test_cases,
    xfails={"mobilenet_v2": "MLETORCH-XXX - Investigate mobilenet_v2 flakiness"},
    strict=False,
)
def test_implementation_mv2(test_case):
    inputs = test_case.example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_implementation(
        qtol=10,
        calibration_samples=calibration_samples,
    )

    # assert that top 1 output matches
    ref = tester.get_artifact(StageType.EXPORT).module()(*inputs)
    result = tester.stages[StageType.SERIALIZE].run_artifact(inputs)
    assert torch.argmax(ref) == torch.argmax(result[0]), "Mismatch in model outputs"
