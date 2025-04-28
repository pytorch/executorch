# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number
from typing import Tuple, Union

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


aten_op = "torch.ops.aten.clamp.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_clamp_default"
input_t = Tuple[torch.Tensor]

test_data_suite = {
    # test_name: (test_data, min, max)
    "rank_1": (torch.rand(10) * 2, -1.0, 1.0),
    "rank_2": (torch.rand(1, 35), 0.5, 0.8),
    "rank_3": (torch.ones(1, 10, 10), -1, -1),
    "rank_4": (torch.rand(1, 10, 10, 1) * 2, -0.1, 2.0),
    "rank_4_mixed_min_max_dtype": (torch.rand(1, 10, 10, 5) + 10, 8.0, 10),
    "rank_4_no_min": (torch.rand(1, 10, 10, 1) * 10, None, 5),
    "rank_4_no_max": (torch.rand(1, 10, 10, 1) - 3, -3.3, None),
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
def test_clamp_tosa_MI(test_data):

    input_tensor, min_val, max_val = test_data
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineMI[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clamp_tosa_BI(test_data):

    input_tensor, min_val, max_val = test_data
    model = Clamp(min_val, max_val)

    pipeline = TosaPipelineBI[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        symmetric_io_quantization=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clamp_u55_BI(test_data):

    input_tensor, min_val, max_val = test_data
    model = Clamp(min_val, max_val)

    pipeline = EthosU55PipelineBI[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        run_on_fvp=False,
        symmetric_io_quantization=True,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clamp_u85_BI(test_data):

    input_tensor, min_val, max_val = test_data
    model = Clamp(min_val, max_val)

    pipeline = EthosU85PipelineBI[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        run_on_fvp=False,
        symmetric_io_quantization=True,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoCorstone300
def test_clamp_u55_BI_on_fvp(test_data):

    input_tensor, min_val, max_val = test_data
    model = Clamp(min_val, max_val)

    pipeline = EthosU55PipelineBI[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )

    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoCorstone320
def test_clamp_u85_BI_on_fvp(test_data):

    input_tensor, min_val, max_val = test_data
    model = Clamp(min_val, max_val)

    pipeline = EthosU85PipelineBI[input_t](
        model,
        (input_tensor,),
        aten_op,
        exir_op,
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)

    pipeline.run()
