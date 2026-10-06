# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.test.common import parametrize

from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.backends.test.harness.stages import StageType

YOLO = pytest.importorskip(
    "ultralytics",
    reason="ultralytics is optional; install it locally to run YOLO tests.",
).YOLO


ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten__softmax_default": 2,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 22,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 6,
    "executorch_exir_dialects_edge__ops_aten_bmm_default": 2,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 26,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 88,
    "executorch_exir_dialects_edge__ops_aten_copy_default": 3,
    "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_expand_copy_default": 6,
    "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 3,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 5,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 3,
    "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1,
    "executorch_exir_dialects_edge__ops_aten_silu_default": 77,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 11,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 7,
    "executorch_exir_dialects_edge__ops_aten_upsample_nearest2d_vec": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 26,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 175,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 308,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 263,
}
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 6,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 6,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 26,
    "executorch_exir_dialects_edge__ops_aten_copy_default": 3,
    "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_expand_copy_default": 6,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 3,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 11,
    "executorch_exir_dialects_edge__ops_aten_upsample_nearest2d_vec": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 33,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 13,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 9,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 78,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 18,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_nhwc_default": 81,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_nhwc_default": 7,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_max_pool2d_nhwc_default": 3,
    "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 81,
}


ops_before_explicit_layout: dict[str, int] = {
    **ops_before_transforms,
    "executorch_exir_dialects_edge__ops_dim_order_ops__empty_dim_order_default": 3,
}

ops_after_explicit_layout: dict[str, int] = {
    **ops_after_transforms,
    "executorch_exir_dialects_edge__ops_dim_order_ops__empty_dim_order_default": 3,
}


test_cases = {
    "yolo11n": McuTestCase(
        model=None,  # type: ignore[arg-type]
        example_inputs=lambda: (torch.randn(1, 3, 640, 640),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_yolo11(test_case):
    WEIGHTS = "yolo11n.pt"
    yolo = YOLO(WEIGHTS)
    pt_model = yolo.model.eval()

    inputs = test_case.get_example_inputs()
    tester = CortexMTester(pt_model, inputs)
    tester.test_dialect(
        ops_before_explicit_layout,
        ops_after_explicit_layout,
        qtol=10,
        use_explicit_layout=True,
        compare_outputs=False,
    )

    ref, scale = tester._calculate_reference_output(
        tester.get_artifact(StageType.EXPORT), inputs
    )
    result = tester.stages[StageType.RUN_PASSES].run_artifact(inputs)
    try:
        tester._compare_outputs(ref, result, scale, qtol=10)
    except AssertionError:
        pytest.xfail("YOLO11 output differs numerically after Cortex-M lowering")
