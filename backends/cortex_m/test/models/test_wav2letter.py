# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.wav2letter.model import Wav2LetterModel


ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten__log_softmax_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 12,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 12,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 24,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 13,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 13,
}
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten__log_softmax_default": 1,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_nhwc_default": 12,
    "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
}

model = Wav2LetterModel()
pt_model = model.get_eager_model()

test_cases = {
    "wav2letter": McuTestCase(
        model=pt_model,
        example_inputs=lambda: model.get_example_inputs(),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_wav2letter(test_case):
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
        use_explicit_layout=True,
    )
