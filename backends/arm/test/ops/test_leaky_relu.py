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

aten_op = "torch.ops.aten.leaky_relu.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_leaky_relu_default"
input_t1 = Tuple[torch.Tensor]  # Input x


class LeakyReLU(torch.nn.Module):
    def __init__(self, slope: float = 0.01):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor):
        return self.activation(x)

    test_data: dict[str, input_t1] = {
        "zeros": ((torch.zeros(1, 1, 5, 5),), 0.01),
        "ones": ((torch.ones(1, 32, 112, 112),), 0.01),
        "rand": ((torch.rand(1, 96, 56, 56),), 0.2),
        "3Dtensor": ((torch.rand(5, 5, 5),), 0.001),
        "negative_slope": ((torch.rand(1, 16, 128, 128),), -0.002),
    }


@common.parametrize("test_data", LeakyReLU.test_data)
def test_leaky_relu_tosa_MI(test_data):
    data, slope = test_data
    pipeline = TosaPipelineMI[input_t1](
        LeakyReLU(slope), data, [], use_to_edge_transform_and_lower=True
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
def test_leaky_relu_tosa_BI(test_data):
    data, slope = test_data
    pipeline = TosaPipelineBI[input_t1](
        LeakyReLU(slope), data, [], use_to_edge_transform_and_lower=True
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
@common.XfailIfNoCorstone300
def test_leaky_relu_u55_BI(test_data):
    data, slope = test_data
    pipeline = EthosU55PipelineBI[input_t1](
        LeakyReLU(slope),
        data,
        [],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
@common.XfailIfNoCorstone320
def test_leaky_relu_u85_BI(test_data):
    data, slope = test_data
    pipeline = EthosU85PipelineBI[input_t1](
        LeakyReLU(slope),
        data,
        [],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()
