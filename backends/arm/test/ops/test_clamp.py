# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number
from typing import Tuple, Union

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.clamp.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_clamp_default"

input_t = Tuple[torch.Tensor]


test_data_suite = {
    # test_name: (test_data, min, max)
    "rank_1": lambda: (torch.rand(10) * 2, -1.0, 1.0),
    "rank_2": lambda: (torch.rand(1, 35), 0.5, 0.8),
    "rank_3": lambda: (torch.ones(1, 10, 10), -1, -1),
    "rank_4": lambda: (torch.rand(1, 10, 10, 1) * 2, -0.1, 2.0),
    "rank_4_mixed_min_max_dtype": lambda: (torch.rand(1, 10, 10, 5) + 10, 8.0, 10),
    "rank_4_no_min": lambda: (torch.rand(1, 10, 10, 1) * 10, None, 5),
    "rank_4_no_max": lambda: (torch.rand(1, 10, 10, 1) - 3, -3.3, None),
}


class Clamp(torch.nn.Module):
    def __init__(
        self,
        clamp_min: Union[torch.Tensor, Number, None],
        clamp_max: Union[torch.Tensor, Number, None],
    ):
        super().__init__()

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        return torch.clamp(x, self.clamp_min, self.clamp_max)


@common.parametrize("test_data", test_data_suite)
def test_clamp_tosa_FP(test_data):

    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineFP[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clamp_tosa_INT(test_data):

    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_clamp_u55_INT(test_data):

    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = EthosU55PipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )

    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_clamp_u85_INT(test_data):

    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)

    pipeline = EthosU85PipelineINT[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_clamp_vgf_FP(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = VgfPipeline[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_clamp_vgf_INT(test_data):
    input_tensor, min_val, max_val = test_data()
    model = Clamp(min_val, max_val)
    pipeline = VgfPipeline[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    # TODO: MLETORCH-1136 Change args of run_method_and_compare_outputs of the vgf tests
    # pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()
