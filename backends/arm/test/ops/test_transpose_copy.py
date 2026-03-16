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
    "rank_2": lambda: (torch.rand(2, 3), 0, 1),
    "rank_2_swapped": lambda: (torch.rand(3, 4), 1, 0),
    "rank_3": lambda: (torch.ones(5, 10, 10), 1, 2),
    "rank_4": lambda: (torch.rand(1, 10, 4, 2) * 2, 2, 0),
}


class TransposeCopy(torch.nn.Module):
    aten_op = "torch.ops.aten.transpose_copy.int"
    exir_op = "executorch_exir_dialects_edge__ops_aten_permute_copy_default"

    def forward(self, x: torch.Tensor, dim0: int, dim1: int):
        return torch.transpose_copy(x, dim0=dim0, dim1=dim1)


input_t1 = Tuple[torch.Tensor]


@common.parametrize("test_data", test_data_suite)
def test_transpose_int_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        TransposeCopy(),
        test_data(),
        aten_op=TransposeCopy.aten_op,
        exir_op=TransposeCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_transpose_int_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        TransposeCopy(),
        test_data(),
        aten_op=TransposeCopy.aten_op,
        exir_op=TransposeCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_transpose_int_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        TransposeCopy(),
        test_data(),
        aten_ops=TransposeCopy.aten_op,
        exir_ops=TransposeCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_transpose_int_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        TransposeCopy(),
        test_data(),
        aten_ops=TransposeCopy.aten_op,
        exir_ops=TransposeCopy.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_transpose_int_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        TransposeCopy(),
        test_data(),
        aten_op=TransposeCopy.aten_op,
        exir_op=TransposeCopy.exir_op,
        use_to_edge_transform_and_lower=False,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_transpose_int_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        TransposeCopy(),
        test_data(),
        aten_op=TransposeCopy.aten_op,
        exir_op=TransposeCopy.exir_op,
        use_to_edge_transform_and_lower=False,
        quantize=True,
    )
    pipeline.run()
