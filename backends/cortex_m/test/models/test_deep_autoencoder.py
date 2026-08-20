# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.mlperf_tiny.deep_autoencoder import DeepAutoEncoder

ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_linear_default": 10,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 9,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 31,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 11,
}

ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 10,
}

test_cases = {
    "deep_autoencoder": McuTestCase(
        model=DeepAutoEncoder().eval(),
        example_inputs=lambda: ((torch.rand(1, 640) * 2 - 1,)),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_deep_autoencoder(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(ops_before_transforms, ops_after_transforms, qtol=1)


@parametrize("test_case", test_cases)
def test_implementation_deep_autoencoder(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_implementation(qtol=1)
