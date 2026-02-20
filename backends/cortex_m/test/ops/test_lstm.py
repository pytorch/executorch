# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMLSTM(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_full_default": 2,
        "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 4,
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 6,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 3,
        "executorch_exir_dialects_edge__ops_aten_addmm_default": 3,
        "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2,
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 4,
        "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 6,
        "executorch_exir_dialects_edge__ops_aten_tanh_default": 4,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 6,
        "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    }

    ops_after_transforms = {}

    def __init__(self, input_size: int = 4, hidden_size: int = 3) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return y


class CortexMQuantizableLSTM(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 4,
        "executorch_exir_dialects_edge__ops_aten_addmm_default": 4,
        "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
        "executorch_exir_dialects_edge__ops_aten_full_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 6,
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 4,
        "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
        "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 6,
        "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 2,
        "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 1,
        "executorch_exir_dialects_edge__ops_aten_tanh_default": 4,
        "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 34,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 27,
    }

    ops_after_transforms = {}

    def __init__(self, input_size: int = 4, hidden_size: int = 3) -> None:
        super().__init__()
        self.lstm = torch.ao.nn.quantizable.LSTM(
            input_size=input_size, hidden_size=hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return y


test_cases = {
    "lstm_fp32": McuTestCase(
        model=CortexMLSTM(),
        example_inputs=(ramp_tensor(-1, 1, (2, 1, 4)),),
    ),
    "lstm_quantizable": McuTestCase(
        model=CortexMQuantizableLSTM(),
        example_inputs=(ramp_tensor(-1, 1, (2, 1, 4)),),
    ),
}


@pytest.mark.skip("Not implemented yet.")
def test_dialect_lstm(test_case: McuTestCase) -> None:
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms, test_case.model.ops_after_transforms
    )


@pytest.mark.skip("Not implemented yet.")
def test_implementation_lstm(test_case: McuTestCase) -> None:
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation()
