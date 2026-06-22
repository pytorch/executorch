# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.mlperf_tiny.ds_cnn import DSCNNKWS

ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 9,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 9,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 2,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 18,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 17,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 15,
}

ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_pad_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 4,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 5,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 2,
}

test_cases = {
    "ds_cnn": McuTestCase(
        model=DSCNNKWS().eval(),
        example_inputs=lambda: (
            (torch.rand(1, 1, 49, 10) * 2 - 1).to(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_ds_cnn(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(ops_before_transforms, ops_after_transforms, qtol=1)


@parametrize("test_case", test_cases)
def test_implementation_ds_cnn(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_implementation(qtol=1)
