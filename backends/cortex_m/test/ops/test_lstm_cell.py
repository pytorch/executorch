# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)

# DecomposeLSTMCellPass (pre-annotation) rewrites the opaque aten.lstm_cell into
# linear / slice / sigmoid / tanh / mul / add, which the quantizer annotates and
# the cortex_m passes lower: the two gemms -> quantized_linear, the gate
# sigmoids/tanhs -> quantized_activation, the cell-state elementwise -> quantized
# add / mul. The gate split stays as portable slice_copy (data movement); the
# surrounding dq/q island is removed once the data-movement folding lands.
# (Full nn.LSTM / sequence support is still unimplemented -- see test_lstm.py.)
_H = 3

# The lowered compute is identical across variants (2 linear, 5 activation, 2
# add, 3 mul, 4 slice); only the q/dq boundary count differs by case.
_OPS_AFTER_COMPUTE = {
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 4,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 3,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 5,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 3,
}


class CortexMLSTMCell(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
        "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 3,
        "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 3,
        "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 4,
        "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 26,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 19,
    }
    ops_after_transforms = {
        **_OPS_AFTER_COMPUTE,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, input_size=4, hidden_size=_H, bias=True):
        super().__init__()
        self.cell = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)

    def forward(self, x, h, c):
        h1, _ = self.cell(x, (h, c))
        return h1


class CortexMLSTMCellNoBias(CortexMLSTMCell):
    # bias=False drops the two bias params -> two fewer dequant nodes pre-pass;
    # the lowered graph is otherwise identical (exercises the b_ih/b_hh=None path).
    ops_before_transforms = {
        **CortexMLSTMCell.ops_before_transforms,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 24,
    }

    def __init__(self):
        super().__init__(bias=False)


class CortexMLSTMCellState(CortexMLSTMCell):
    # Returns both (h, c): the extra c output adds one dequant at the boundary
    # and exercises the getitem[1] -> c' rewrite.
    ops_before_transforms = {
        **CortexMLSTMCell.ops_before_transforms,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 27,
    }
    ops_after_transforms = {
        **_OPS_AFTER_COMPUTE,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 2,
    }

    def forward(self, x, h, c):
        return self.cell(x, (h, c))


def _inputs(nonzero_state=False):
    state = ramp_tensor if nonzero_state else (lambda a, b, s: torch.zeros(s))
    return (
        ramp_tensor(-1, 1, (1, 4)),
        state(-1, 1, (1, _H)),
        state(-1, 1, (1, _H)),
    )


test_cases = {
    "lstm_cell": McuTestCase(model=CortexMLSTMCell(), example_inputs=_inputs()),
    "lstm_cell_no_bias": McuTestCase(
        model=CortexMLSTMCellNoBias(), example_inputs=_inputs()
    ),
    "lstm_cell_state": McuTestCase(
        model=CortexMLSTMCellState(), example_inputs=_inputs(nonzero_state=True)
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_lstm_cell(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=10,
    )


@parametrize("test_case", test_cases)
def test_implementation_lstm_cell(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=10)
