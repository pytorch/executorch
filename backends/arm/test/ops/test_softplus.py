# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


class SoftplusWithParameters(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=2.0, threshold=5.0)


def test_softplus_parameters_tosa_int():
    values = torch.tensor([[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]])
    pipeline = TosaPipelineINT(
        SoftplusWithParameters(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
    )
    pipeline.count_tosa_ops({"TABLE": 1})
    pipeline.run()


def test_softplus_parameters_tosa_a16w8():
    values = torch.tensor([[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]])
    pipeline = TosaPipelineINT(
        SoftplusWithParameters(),
        (values,),
        "torch.ops.aten.softplus.default",
        "executorch_exir_dialects_edge__ops_aten_softplus_default",
        tosa_extensions=["int16"],
    )
    pipeline.count_tosa_ops({"TABLE": 1})
    pipeline.run()


def test_softplus_parameters_tosa_fp():
    values = torch.tensor([[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]])
    pipeline = TosaPipelineFP(
        SoftplusWithParameters(),
        (values,),
        "torch.ops.aten.softplus.default",
        [],
    )
    pipeline.run()
