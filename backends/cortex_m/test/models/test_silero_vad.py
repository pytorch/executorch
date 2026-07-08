# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.silero_vad.export_silero_vad import (
    CONTEXT_SIZE,
    HIDDEN_DIM,
    SileroVAD16k,
    WINDOW_SIZE,
)


ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_abs_default": 2,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 6,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 5,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
    "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 4,
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 1,
    "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 15,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 14,
}
# The final `sigmoid(final_conv(x))` now lowers to cortex_m.quantized_activation.
# The 3 remaining sigmoids and 2 tanhs are LSTMCell gates: PyTorch export
# captures nn.LSTMCell as a single high-level op, so the quantizer never sees
# the gate activations and can't annotate them. They're decomposed only at
# to_edge -- which runs after the quantizer, so by then the gates have no
# qparams to fold and the lowering pass correctly skips them. The unblocker
# is a pre-annotation decompose pass that splits nn.LSTMCell into linear +
# split + sigmoid + tanh + add + mul *before* prepare_pt2e runs; tracked as
# the LSTMCell verification follow-up.
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_abs_default": 2,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 6,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 5,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
    "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 3,
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 1,
    "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 7,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 7,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
}


pt_model = SileroVAD16k().eval()

x = torch.randn(
    1, CONTEXT_SIZE + WINDOW_SIZE
)  # (1, 576) — 64 context + 512 audio samples
state = torch.zeros(2, 1, HIDDEN_DIM)  # (2, 1, 128) — [h, c] LSTM state

test_cases = {
    "silero_vad_16k": McuTestCase(
        model=pt_model,
        example_inputs=lambda: (x, state),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_silero_vad_16k(test_case):
    """This model currently does largely not lower to accelerated kernels due to missing LSTM and conv1d support, this test is to track development progress."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
