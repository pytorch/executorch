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
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 11,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 26,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 24,
}
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_abs_default": 2,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
    # 4 of the 5 ReLUs fuse into their preceding encoder Conv1d. The surviving
    # one is the post-LSTM `F.relu(h)` -- its producer is an elementwise mul
    # (not a fusible conv/linear tail) and there is no standalone quantized relu,
    # so it stays in aten.
    "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
    # The final `sigmoid(final_conv(x))` lowers to cortex_m.quantized_activation
    # (one below); the 3 surviving sigmoids + 2 tanhs are the LSTMCell gates,
    # which stay in aten until the pre-annotation LSTMCell decompose lands.
    "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 3,
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 1,
    "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    # The conv1d lowering inserts view_copy wraps around each conv2d; the
    # encoder's chained Conv1ds get their inter-layer view_copy <-> view_copy
    # plus inverse _clone_dim_order pairs folded out. View_copy nodes here
    # are: original model view (1) + boundary wraps that survive folding (6).
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 7,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 9,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 8,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 5,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 1,
    # Five clone_dim_order survive: boundary wraps for the conv1ds whose
    # neighbours aren't another conv1d (the STFT/magnitude-spectrum interface
    # and the final-conv boundary). The STFT conv1d has in_channels==1, so its
    # input NHWC clone is skipped (channels-last == contiguous) -- one fewer
    # than the channelled boundaries would give.
    "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default": 5,
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
    """Conv1d layers lower to cortex_m.quantized_conv2d via reshape and the
    final-conv sigmoid lowers to cortex_m.quantized_activation. The LSTMCell
    gates (3 sigmoid + 2 tanh + 2 linear) stay in aten until the pre-annotation
    LSTMCell decompose lands. This test tracks progress."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
