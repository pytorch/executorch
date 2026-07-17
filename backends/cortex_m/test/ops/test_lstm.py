# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMLSTM(torch.nn.Module):
    # A single-layer unidirectional nn.LSTM is preserved as one aten.lstm.input
    # node, boundary-quantized, and fused into cortex_m.quantized_lstm.
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_lstm_input": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_aten_full_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_lstm_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(
        self, input_size: int = 4, hidden_size: int = 3, batch_first: bool = False
    ) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return y


test_cases = {
    "lstm_fp32": McuTestCase(
        model=CortexMLSTM(),
        example_inputs=(ramp_tensor(-1, 1, (2, 1, 4)),),
    ),
    # Wider hidden size exercises the CMSIS MVE main loop (hidden % 4 == 0).
    "lstm_fp32_wide": McuTestCase(
        model=CortexMLSTM(input_size=8, hidden_size=16),
        example_inputs=(ramp_tensor(-1, 1, (3, 1, 8)),),
    ),
    # batch_first -> CMSIS batch-major (time_major=0) with the [B, T, F] layout.
    "lstm_batch_first": McuTestCase(
        model=CortexMLSTM(batch_first=True),
        example_inputs=(ramp_tensor(-1, 1, (1, 3, 4)),),
    ),
    # batch > 1 exercises the per-batch state/scratch handling.
    "lstm_batched": McuTestCase(
        model=CortexMLSTM(),
        example_inputs=(ramp_tensor(-1, 1, (2, 2, 4)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_lstm(test_case: McuTestCase, cortex_m_target) -> None:
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        atol=0.05,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_lstm(test_case: McuTestCase, cortex_m_target) -> None:
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    # The reference impl matches CMSIS-NN's arm_lstm_unidirectional_s8 within a
    # LSB (its int16 activation LUT vs the float reference), so allow one int8
    # step of slack on top of the quantization tolerance.
    tester.test_implementation(atol=0.1, qtol=1)


class MultiLayerLSTM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=3, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return y


class StateReturningLSTM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=3)

    def forward(self, x: torch.Tensor):
        y, (h_n, _) = self.lstm(x)
        return y, h_n


unsupported_cases = {
    "multi_layer": MultiLayerLSTM,
    "uses_h_n": StateReturningLSTM,
}


@parametrize("model_cls", unsupported_cases)
def test_unsupported_lstm_fails_clearly(model_cls, cortex_m_target) -> None:
    # Unsupported configurations are not fused; because aten.lstm.input is
    # preserved and has no portable kernel, lowering must raise a clear error
    # rather than silently emitting a broken graph.
    tester = CortexMTester(
        model_cls(), (ramp_tensor(-1, 1, (2, 1, 4)),), target_config=cortex_m_target
    )
    with pytest.raises(Exception, match="AtenToCortexMPass"):
        tester.quantize().export().to_edge().run_passes()
