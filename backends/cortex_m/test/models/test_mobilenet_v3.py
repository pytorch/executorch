# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from torchvision import models


# TODO: Update as more ops are converted by CMSIS-NN ops.
ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 34,
    "executorch_exir_dialects_edge__ops_aten_addmm_default": 2,
    "executorch_exir_dialects_edge__ops_aten_clamp_default": 56,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 52,
    "executorch_exir_dialects_edge__ops_aten_div_Tensor": 28,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 10,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 28,
    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 14,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 56,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 178,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 109,
}
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 28,  # Not lowered due to broadcasting
    "executorch_exir_dialects_edge__ops_aten_addmm_default": 0,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 6,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 2,
    "executorch_exir_dialects_edge__ops_aten_clamp_default": 56,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 52,
    "executorch_exir_dialects_edge__ops_aten_div_Tensor": 28,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 10,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 28,
    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 0,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 14,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 56,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 0,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 0,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 162,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 101,
}

model = models.mobilenet_v3_small(weights=None)
example_input = torch.randn(1, 3, 224, 224)


test_cases = {
    "mobilenet_v3_small": McuTestCase(
        model=models.mobilenet_v3_small(weights=None),
        example_inputs=(example_input,),
    ),
}


@pytest.mark.skip("Skip until add + linear fix are upstreamed.")
def test_dialect_mv3(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=1,
    )


@pytest.mark.skip("Skip until add + linear fix are upstreamed.")
def test_implementation_mv3(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
