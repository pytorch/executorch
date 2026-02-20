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


class CortexMSoftmax(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)


test_cases = {
    "rank1": McuTestCase(
        CortexMSoftmax(dim=-1),
        (ramp_tensor(-4, 4, (16,)),),
    ),
    "rank2": McuTestCase(
        CortexMSoftmax(dim=-1),
        (ramp_tensor(-8, 8, (4, 8)),),
    ),
    "rank3": McuTestCase(
        CortexMSoftmax(dim=-1),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "dim_not_last": McuTestCase(
        CortexMSoftmax(dim=1),
        (ramp_tensor(-2, 2, (2, 3, 4)),),
    ),
    "large_tensor": McuTestCase(
        CortexMSoftmax(dim=-1),
        (ramp_tensor(-10, 10, (8, 1024)),),
    ),
}


xfail_cases_dialect = {
    "dim_not_last": (
        "Softmax stays in ATen when dim isn’t the channel dimension, so dialect expectations fail",
        Exception,
    ),
}
xfail_cases_impl = {
    "dim_not_last": (
        "Softmax on Cortex-M currently supports only the last dimension",
        Exception,
    ),
}


@parametrize("test_case", test_cases, xfails=xfail_cases_dialect)
def test_dialect_softmax(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=2,
    )


@parametrize("test_case", test_cases, xfails=xfail_cases_impl)
def test_implementation_softmax(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=2)
