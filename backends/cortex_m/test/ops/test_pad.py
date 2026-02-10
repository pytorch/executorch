# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)

OPS_BEFORE_PASSES = {
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default": 1,
}

OPS_AFTER_PASSES = {
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_pad_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
}


class CortexMPad(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_PASSES
    ops_after_transforms = OPS_AFTER_PASSES

    def __init__(self, padding, value=0.0):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, mode="constant", value=self.value)


test_cases = {
    "pad_rank4_all_dims": McuTestCase(
        CortexMPad((1, 1, 2, 2, 1, 0, 0, 1)),
        (ramp_tensor(-0.5, 0.5, (1, 2, 3, 4)),),
    ),
    "pad_rank4_last_two_dims": McuTestCase(
        CortexMPad((1, 2, 3, 4)),
        (ramp_tensor(-1.0, 1.0, (1, 3, 4, 5)),),
    ),
    "pad_rank3": McuTestCase(
        CortexMPad((1, 1, 2, 2)),
        (ramp_tensor(-0.5, 0.5, (2, 3, 4)),),
    ),
    "pad_rank2": McuTestCase(
        CortexMPad((1, 2, 3, 4)),
        (ramp_tensor(-1.0, 1.0, (3, 5)),),
    ),
    "pad_rank1": McuTestCase(
        CortexMPad((2, 3)),
        (ramp_tensor(0.0, 1.0, (6,)),),
    ),
    "pad_nonzero_value": McuTestCase(
        CortexMPad((1, 1), value=0.5),
        (ramp_tensor(-1.0, 1.0, (2, 4)),),
    ),
    "pad_zero_padding": McuTestCase(
        CortexMPad((0, 0, 0, 0)),
        (ramp_tensor(-0.5, 0.5, (2, 3, 4, 5)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_pad(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=0,
    )


@parametrize("test_case", test_cases)
def test_implementation_pad(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=0)
