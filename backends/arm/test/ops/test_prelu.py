# Copyright 2026 Arm Limited and/or its affiliates.
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
)

aten_op = "torch.ops.aten.prelu.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_prelu_default"
input_t1 = Tuple[torch.Tensor]


class PReLU(torch.nn.Module):
    def __init__(self, num_parameters: int = 1):
        super().__init__()
        self.activation = torch.nn.PReLU(num_parameters=num_parameters)

    def forward(self, x: torch.Tensor):
        return self.activation(x)

    test_data: dict[str, tuple[input_t1, int]] = {
        "scalar_2d": ((torch.randn(4, 5),), 1),
        "scalar_4d": ((torch.randn(1, 3, 8, 8),), 1),
        "per_channel_3d": ((torch.randn(2, 4, 5),), 4),
        "per_channel_4d": ((torch.randn(1, 3, 8, 8),), 3),
    }


@common.parametrize("test_data", PReLU.test_data)
def test_prelu_tosa_FP(test_data):
    data, num_parameters = test_data
    pipeline = TosaPipelineFP[input_t1](
        PReLU(num_parameters),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", PReLU.test_data)
def test_prelu_tosa_INT(test_data):
    data, num_parameters = test_data
    pipeline = TosaPipelineINT[input_t1](
        PReLU(num_parameters),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", PReLU.test_data)
@common.XfailIfNoCorstone300
def test_prelu_u55_INT(test_data):
    data, num_parameters = test_data
    pipeline = EthosU55PipelineINT[input_t1](
        PReLU(num_parameters),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", PReLU.test_data)
@common.XfailIfNoCorstone320
def test_prelu_u85_INT(test_data):
    data, num_parameters = test_data
    pipeline = EthosU85PipelineINT[input_t1](
        PReLU(num_parameters),
        data,
        [],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()
