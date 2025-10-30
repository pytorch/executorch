# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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
        "zeros": lambda: ((torch.zeros(1, 1, 5, 5),), 0.01),
        "ones": lambda: ((torch.ones(1, 16, 96, 96),), 0.01),
        "rand": lambda: ((torch.rand(1, 64, 56, 56),), 0.2),
        "3Dtensor": lambda: ((torch.rand(5, 5, 5),), 0.001),
        "negative_slope": lambda: ((torch.rand(1, 16, 96, 96),), -0.002),
    }


@common.parametrize("test_data", LeakyReLU.test_data)
def test_leaky_relu_tosa_FP(test_data):
    data, slope = test_data()
    pipeline = TosaPipelineFP[input_t1](
        LeakyReLU(slope),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
def test_leaky_relu_tosa_INT(test_data):
    data, slope = test_data()
    pipeline = TosaPipelineINT[input_t1](
        LeakyReLU(slope),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
@common.XfailIfNoCorstone300
def test_leaky_relu_u55_INT(test_data):
    data, slope = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        LeakyReLU(slope),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
@common.XfailIfNoCorstone320
def test_leaky_relu_u85_INT(test_data):
    data, slope = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        LeakyReLU(slope),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
@common.SkipIfNoModelConverter
def test_leaky_relu_vgf_FP(test_data):
    data, slope = test_data()
    pipeline = VgfPipeline[input_t1](
        LeakyReLU(slope),
        data,
        [],
        use_to_edge_transform_and_lower=True,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [aten_op]
    )
    pipeline.run()


@common.parametrize("test_data", LeakyReLU.test_data)
@common.SkipIfNoModelConverter
def test_leaky_relu_vgf_INT(test_data):
    data, slope = test_data()
    pipeline = VgfPipeline[input_t1](
        LeakyReLU(slope),
        data,
        [],
        use_to_edge_transform_and_lower=True,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()
