# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.softmax.default"  # Used for checking that we do not have softmax in the graph after decompose
exir_op = "executorch_exir_dialects_edge__ops_aten__softmax_tensor"
input_t1 = Tuple[torch.Tensor]  # Input x


class Softmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)

    test_data = {
        "ones": lambda: ((torch.ones(10, 10),), 1),
        "ones_neg_dim": lambda: ((torch.ones(1, 3, 4),), -1),
        "randn_neg_dim": lambda: ((torch.randn(1, 5, 8, 7),), -3),
        "zeros": lambda: ((torch.zeros(1, 8, 5, 2),), 0),
        "zeros_neg_dim": lambda: ((torch.zeros(1, 7, 8, 9),), -4),
        "rand": lambda: ((torch.rand(1, 2, 5, 8),), 2),
        "rand_neg_dim": lambda: ((torch.rand(1, 10, 8, 10),), -2),
        "randn_mult_batches": lambda: ((torch.randn(2, 10, 10, 10),), 3),
    }


@common.parametrize("test_data", Softmax.test_data)
def test_softmax_tosa_FP(test_data):
    data, dim = test_data()
    pipeline = TosaPipelineFP[input_t1](Softmax(dim), data, [])
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
def test_softmax_tosa_INT(test_data):
    data, dim = test_data()
    pipeline = TosaPipelineINT[input_t1](Softmax(dim), data, [])
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.XfailIfNoCorstone300
def test_softmax_u55_INT(test_data):
    data, dim = test_data()
    pipeline = EthosU55PipelineINT[input_t1](Softmax(dim), data, [], run_on_fvp=True)
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.XfailIfNoCorstone320
def test_softmax_u85_INT(test_data):
    data, dim = test_data()
    pipeline = EthosU85PipelineINT[input_t1](Softmax(dim), data, [], run_on_fvp=True)
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.SkipIfNoModelConverter
def test_softmax_vgf_FP(test_data):
    data, dim = test_data()
    pipeline = VgfPipeline[input_t1](
        Softmax(dim),
        data,
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", Softmax.test_data)
@common.SkipIfNoModelConverter
def test_softmax_vgf_INT(test_data):
    data, dim = test_data()
    pipeline = VgfPipeline[input_t1](
        Softmax(dim),
        data,
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    # TODO: MLETORCH-1136 Change args of run_method_and_compare_outputs of the vgf tests
    # pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()
