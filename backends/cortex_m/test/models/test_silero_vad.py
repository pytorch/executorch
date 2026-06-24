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
    # DecomposeLSTMCellPass splits the gates with slices (i/f/g/o), so the
    # opaque cell's split_with_sizes is gone and slice_copy rises to 6 (2 model
    # + 4 gate). The gate linear/sigmoid/tanh/mul/add are now quantized, hence
    # the higher q/dq counts vs the opaque-cell graph.
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 6,
    "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 45,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 35,
}
# DecomposeLSTMCellPass lowers the LSTMCell gates: the 2 gemms ->
# quantized_linear, the 5 gate activations + the final-conv sigmoid -> 6
# quantized_activation, and the cell-state elementwise -> quantized_add/mul.
# The post-LSTM `F.relu(h)` has a quantized (mul) producer so it normalizes to
# an aten.clamp (it doesn't fuse -- mul isn't a conv/linear tail). Conv1d does
# not yet lower, so the 6 convolutions stay fp32 and their 4 conv-tail relus
# stay in aten. Magnitude spectrum (abs/pow/sqrt/sub), mean, and the
# slices/views stay portable.
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_abs_default": 2,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 6,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 2,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 4,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 6,
    "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 6,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 7,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 6,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 3,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 3,
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
    """The LSTMCell gates and the final-conv sigmoid lower to cortex_m ops
    (quantized_linear / quantized_activation / quantized_add / quantized_mul)
    via DecomposeLSTMCellPass. The Conv1d encoder does not yet lower. This test
    tracks development progress."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
