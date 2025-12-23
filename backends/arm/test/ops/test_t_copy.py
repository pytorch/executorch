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

test_data_suite = {
    # test_name: (test_data, min, max)
    "rand": lambda: (torch.rand(2, 3),),
    "rand_multiplied": lambda: (torch.rand(3, 4) * 10,),
    "ones": lambda: (torch.ones(5, 10),),
    "randn": lambda: (torch.randn(1, 10) * 2,),
}


class TCopy(torch.nn.Module):
    aten_op = "torch.ops.aten.t_copy.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_permute_copy_default"

    def forward(self, x: torch.Tensor):
        return torch.t_copy(x)


input_t1 = Tuple[torch.Tensor]


@common.parametrize("test_data", test_data_suite)
def test_t_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        TCopy(),
        test_data(),
        aten_op=TCopy.aten_op,
        exir_op=TCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_t_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        TCopy(),
        test_data(),
        aten_op=TCopy.aten_op,
        exir_op=TCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_t_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        TCopy(),
        test_data(),
        aten_ops=TCopy.aten_op,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_t_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        TCopy(),
        test_data(),
        aten_ops=TCopy.aten_op,
        exir_ops=TCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_t_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        TCopy(),
        test_data(),
        aten_op=TCopy.aten_op,
        exir_op=TCopy.exir_op,
        use_to_edge_transform_and_lower=False,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_t_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        TCopy(),
        test_data(),
        aten_op=TCopy.aten_op,
        exir_op=TCopy.exir_op,
        use_to_edge_transform_and_lower=False,
        quantize=True,
    )
    pipeline.run()
