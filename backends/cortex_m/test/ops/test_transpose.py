# Copyright 2025 Arm Limited and/or its affiliates.
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

OPS_BEFORE_PASSES = {
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
}

OPS_AFTER_PASSES = {
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
}


class CortexMPermute(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_PASSES
    ops_after_transforms = OPS_AFTER_PASSES

    def __init__(self, perms):
        super().__init__()
        self.perms = perms

    def forward(self, x):
        return x.permute(self.perms)


class CortexMTranspose(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_PASSES
    ops_after_transforms = OPS_AFTER_PASSES

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class CortexMT(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_PASSES
    ops_after_transforms = OPS_AFTER_PASSES

    def forward(self, x):
        return x.t()


test_cases = {
    "permute_nhwc_to_nchw": McuTestCase(
        CortexMPermute((0, 3, 1, 2)),
        (ramp_tensor(-0.5, 0.5, (2, 3, 4, 2)),),
    ),
    "permute_nchw_to_nhwc_neg_index": McuTestCase(
        CortexMPermute((0, -2, -1, -3)),
        (ramp_tensor(10, 100, (2, 3, 4, 2)),),
    ),
    "permute_rank_1": McuTestCase(
        CortexMPermute((0,)),
        (ramp_tensor(10, 100, (3)),),
    ),
    "transpose_1_2": McuTestCase(
        CortexMTranspose(1, 2),
        (ramp_tensor(-1.0, 1.0, (1, 3, 4)),),
    ),
    "transpose_0_1": McuTestCase(
        CortexMTranspose(0, 1),
        (ramp_tensor(-2.0, 2.0, (2, 3, 4, 3)),),
    ),
    "t_operator": McuTestCase(
        CortexMT(),
        (ramp_tensor(-0.5, 0.5, (4, 2)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_transpose(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_transpose(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
