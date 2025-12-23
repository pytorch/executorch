# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x


class RescaleNetwork(torch.nn.Module):
    test_data = {
        "rand": (torch.rand(5), torch.rand(5)),
        "randn": (torch.randn(5, 2), torch.randn(5, 1)),
        "ones": (torch.ones(1, 10, 4, 6), torch.ones(1, 10, 4, 6)),
        "randn_ones": (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
        "randn_large": (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        a = y.exp()
        g = (a + 5).log()
        c = a + x
        d = c - g
        e = c * d
        f = e.sigmoid()

        return f


@common.parametrize("test_data", RescaleNetwork.test_data)
def test_insert_rescale_tosa_INT(test_data: tuple[torch.Tensor, torch.Tensor]):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    module = RescaleNetwork()
    pipeline = TosaPipelineINT(
        module=module,
        test_data=test_data,
        aten_op=[],
        exir_op=[],
    )
    if not conftest.is_option_enabled("tosa_ref_model"):
        pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.parametrize("test_data", RescaleNetwork.test_data)
@common.XfailIfNoCorstone300
def test_insert_rescale_u55_INT(test_data: input_t):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    module = RescaleNetwork()
    pipeline = EthosU55PipelineINT(
        module=module,
        test_data=test_data,
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", RescaleNetwork.test_data)
@common.XfailIfNoCorstone320
def test_insert_rescale_u85_INT(test_data: input_t):
    """Tests a model with many ops that requires rescales. As more ops are quantized to int32 and
    need the InsertRescalesPass, make sure that they play nicely together."""
    module = RescaleNetwork()
    pipeline = EthosU85PipelineINT(
        module=module,
        test_data=test_data,
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()
