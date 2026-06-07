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
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 14,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 14,
}
# Every Conv1d + ReLU pair fuses into a single cortex_m.quantized_conv2d call;
# only the final log_softmax stays in aten until a quantized log_softmax lands.
# Between consecutive Conv1ds the view_copy + _clone_dim_order wrap/unwrap
# chain collapses to a direct conv2d -> conv2d hand-off (via FuseViewCopyTransform
# + FoldInverseDimOrderClonePass). The model-boundary view_copy pair survives
# (one at the model input, one at output). Only one _clone_dim_order survives:
# the first Conv1d has in_channels==1, so its input NHWC clone is skipped
# (channels-last == contiguous), leaving just the output-boundary clone.
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten__log_softmax_default": 1,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 12,
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 1,
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
    """Wav2Letter is a pure Conv1d+ReLU stack with a log_softmax tail; the
    Conv1d-via-Conv2d-reshape lowering now collapses every layer into a single
    cortex_m.quantized_conv2d. Only log_softmax stays unfused."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
