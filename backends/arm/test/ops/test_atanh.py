# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.atanh.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__atanh_default"


input_t1 = Tuple[torch.Tensor]


test_data_suite = {
    "zeros": torch.zeros(1, 10, 10, 10),
    "zeros_alt_shape": torch.zeros(1, 10, 3, 5),
    "ones": torch.ones(10, 10, 10),
    "rand": torch.rand(10, 10) - 0.5,
    "rand_alt_shape": torch.rand(1, 10, 3, 5) - 0.5,
    "ramp": torch.arange(-1, 1, 0.2),
    "near_bounds": torch.tensor([-0.999999, -0.999, -0.9, 0.9, 0.999, 0.999999]),
    "on_bounds": torch.tensor([-1.0, 1.0]),
}


class Atanh(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.atanh(x)


@common.parametrize("test_data", test_data_suite)
def test_atanh_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        Atanh(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_atanh_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        Atanh(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_atanh_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Atanh(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_atanh_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Atanh(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()
