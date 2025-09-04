# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

test_data_suite = {
    # (test_name, test_data)
    "zeros_default": lambda: (1.0, torch.zeros(1, 10, 10, 10)),
    "ones_default": lambda: (1.0, torch.ones(10, 10, 10)),
    "rand_default": lambda: (1.0, torch.rand(10, 10) - 0.5),
    "randn_pos_default": lambda: (1.0, torch.randn(1, 2, 3, 3) + 10),
    "randn_neg_default": lambda: (1.0, torch.randn(2, 4, 3) - 10),
    "ramp_default": lambda: (1.0, torch.arange(-16, 16, 0.2)),
    "large_pos_default": lambda: (1.0, torch.randn(3, 3) * 1e6 + 1e7),
    "large_neg_default": lambda: (1.0, -torch.empty(5).uniform_(1e5, 1e8)),
    "small_pos_default": lambda: (1.0, torch.empty(5).uniform_(1e-8, 1e-5)),
    "small_neg_default": lambda: (1.0, -torch.empty(5).uniform_(1e-8, 1e-5)),
    "zeros_custom": lambda: (2.0, torch.zeros(1, 10, 10, 10)),
    "ones_custom": lambda: (2.0, torch.ones(10, 10, 10)),
    "rand_custom": lambda: (2.0, torch.rand(10, 10) - 0.5),
    "randn_pos_custom": lambda: (2.0, torch.randn(1, 3, 3) + 10),
    "randn_neg_custom": lambda: (2.0, torch.randn(1, 2, 4, 3) - 10),
    "ramp_custom": lambda: (2.0, torch.arange(-16, 16, 0.2)),
    "large_pos_custom": lambda: (2.0, torch.randn(3, 3) * 1e6 + 1e7),
    "large_neg_custom": lambda: (2.0, -torch.empty(5).uniform_(1e5, 1e8)),
    "small_pos_custom": lambda: (2.0, torch.empty(5).uniform_(1e-8, 1e-5)),
    "small_neg_custom": lambda: (2.0, -torch.empty(5).uniform_(1e-8, 1e-5)),
    "zeros_zero": lambda: (0.0, torch.zeros(1, 10, 10, 10)),
    "ones_zero": lambda: (0.0, torch.ones(10, 10, 10)),
    "rand_zero": lambda: (0.0, torch.rand(10, 10) - 0.5),
    "randn_pos_zero": lambda: (0.0, torch.randn(1, 3, 3) + 10),
    "randn_neg_zero": lambda: (0.0, torch.randn(1, 2, 4, 3) - 10),
    "ramp_zero": lambda: (0.0, torch.arange(-16, 16, 0.2)),
    "large_pos_zero": lambda: (0.0, torch.randn(3, 3) * 1e6 + 1e7),
    "large_neg_zero": lambda: (0.0, -torch.empty(5).uniform_(1e5, 1e8)),
    "small_pos_zero": lambda: (0.0, torch.empty(5).uniform_(1e-8, 1e-5)),
    "small_neg_zero": lambda: (0.0, -torch.empty(5).uniform_(1e-8, 1e-5)),
}


class Elu(nn.Module):
    aten_op = "torch.ops.aten.elu.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten__elu_default"

    def __init__(self, input_alpha: float = 1.0):
        super().__init__()
        self.elu = torch.nn.ELU(alpha=input_alpha)

    def forward(self, input_: torch.Tensor):
        return self.elu(input_)


input_t1 = Tuple[torch.Tensor]


@common.parametrize("test_module", test_data_suite)
def test_elu_tosa_FP(test_module: input_t1):
    alpha, test_data = test_module()
    pipeline = TosaPipelineFP[input_t1](
        Elu(alpha), (test_data,), aten_op=Elu.aten_op, exir_op=Elu.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_suite)
def test_elu_tosa_INT(test_module: input_t1):
    alpha, test_data = test_module()
    pipeline = TosaPipelineINT[input_t1](
        Elu(alpha), (test_data,), aten_op=Elu.aten_op, exir_op=Elu.exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_module", test_data_suite)
def test_elu_u55_INT(test_module: input_t1):
    alpha, test_data = test_module()
    pipeline = EthosU55PipelineINT[input_t1](
        Elu(alpha), (test_data,), aten_ops=Elu.aten_op, exir_ops=Elu.exir_op
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_module", test_data_suite)
def test_elu_u85_INT(test_module: input_t1):
    alpha, test_data = test_module()
    pipeline = EthosU85PipelineINT[input_t1](
        Elu(alpha), (test_data,), aten_ops=Elu.aten_op, exir_ops=Elu.exir_op
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite)
def test_elu_vgf_FP(test_module: input_t1):
    alpha, test_data = test_module()
    pipeline = VgfPipeline[input_t1](
        Elu(alpha),
        (test_data,),
        aten_op=Elu.aten_op,
        exir_op=Elu.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite)
def test_elu_vgf_INT(test_module: input_t1):
    alpha, test_data = test_module()
    pipeline = VgfPipeline[input_t1](
        Elu(alpha),
        (test_data,),
        aten_op=Elu.aten_op,
        exir_op=Elu.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
