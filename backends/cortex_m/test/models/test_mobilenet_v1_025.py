# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.mlperf_tiny.mobilenet_v1_025 import MobileNetV1025

ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 27,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 27,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 54,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 33,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 31,
}

ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 14,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 13,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
}

test_cases = {
    "mobilenet_v1_025": McuTestCase(
        model=MobileNetV1025().eval(),
        example_inputs=lambda: (
            (torch.rand(1, 3, 96, 96) * 2 - 1).to(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_mobilenet_v1_025(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(ops_before_transforms, ops_after_transforms, qtol=1)


@parametrize("test_case", test_cases)
def test_implementation_mobilenet_v1_025(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_implementation(qtol=1)
