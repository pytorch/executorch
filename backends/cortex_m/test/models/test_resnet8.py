# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.mlperf_tiny.resnet8 import ResNet8

ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 9,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 7,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 16,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 21,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 16,
}

ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 3,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 9,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
}

test_cases = {
    "resnet8": McuTestCase(
        model=ResNet8().eval(),
        example_inputs=lambda: (
            (torch.rand(1, 3, 32, 32) * 2 - 1).to(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_resnet8(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(ops_before_transforms, ops_after_transforms, qtol=1)


@parametrize("test_case", test_cases)
def test_implementation_resnet8(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_implementation(qtol=1)
