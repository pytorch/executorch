# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import InsertInt32CastsAfterInt64PlaceholdersPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

test_data_suite = {
    "rank3_zeros_int8": (
        lambda: (
            torch.zeros((1, 3, 2), dtype=torch.int8),
            (
                torch.tensor([0, 0], dtype=torch.int64),
                torch.tensor([2, 1], dtype=torch.int64),
            ),
            torch.randint(-5, 5, (2, 2), dtype=torch.int8),
            False,
        ),
        0,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
    "rank3_accumulate": (
        lambda: (
            torch.rand((5, 9, 3), dtype=torch.float32),
            (
                torch.tensor([0, 3], dtype=torch.int64),
                torch.tensor([3, 1], dtype=torch.int64),
            ),
            torch.randint(-5, 5, (2, 3), dtype=torch.float32),
            True,
        ),
        1,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
    "rank4_rand": (
        lambda: (
            torch.rand((1, 2, 4, 5), dtype=torch.float32),
            (
                torch.tensor([0, 0], dtype=torch.int64),
                torch.tensor([1, 0], dtype=torch.int64),
                torch.tensor([2, 3], dtype=torch.int64),
            ),
            torch.tensor(([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), dtype=torch.float32),
            False,
        ),
        0,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
    "rank4_accumulate_int32": (
        lambda: (
            torch.ones((3, 4, 20, 9), dtype=torch.int32),
            (
                torch.tensor(
                    [0, 2, 2],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 1, 1],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [4, 8, 5],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 2, 3],
                    dtype=torch.int64,
                ),
            ),
            torch.zeros((3), dtype=torch.int32),
            True,
        ),
        1,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
    "rank5_ones": (
        lambda: (
            torch.ones((3, 4, 20, 9, 5), dtype=torch.float32),
            (
                torch.tensor(
                    [0, 2, 2],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 1, 1],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [4, 8, 5],
                    dtype=torch.int64,
                ),
            ),
            torch.randn((3, 9, 5), dtype=torch.float32),
            False,
        ),
        0,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
    "rank6_rand": (
        lambda: (
            torch.rand((1, 2, 3, 4, 2, 1), dtype=torch.float32),
            (
                torch.tensor(
                    [0, 0, 0],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 1, 1],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 2, 0],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 2, 1],
                    dtype=torch.int64,
                ),
            ),
            torch.randn((3, 2, 1), dtype=torch.float32),
            False,
        ),
        0,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
    "same_index": (
        lambda: (
            torch.rand((1, 2, 3), dtype=torch.float32),
            (
                torch.tensor(
                    [0, 0],
                    dtype=torch.int64,
                ),
                torch.tensor(
                    [1, 1],
                    dtype=torch.int64,
                ),
            ),
            torch.randn((2, 3), dtype=torch.float32),
            False,
        ),
        0,  # used for u55 tests to config n_expected_delgates, only 1 when accumulate is True
    ),
}


class IndexPut(torch.nn.Module):
    aten_op = "torch.ops.aten.index_put.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_index_put_default"

    def forward(
        self,
        x: torch.Tensor,
        y: tuple[torch.Tensor],
        z: torch.Tensor,
        acc: bool,
    ):
        return torch.index_put(x, indices=y, values=z, accumulate=acc)


input_t = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool], int]

xfails = {
    "same_index": "MLETORCH-1596: index_put with repeated indices not supported",
}


@common.parametrize("test_module", test_data_suite, xfails=xfails)
def test_index_put_tosa_FP(test_module: input_t):
    pipeline = TosaPipelineFP[input_t](
        IndexPut(),
        test_module[0](),
        aten_op=IndexPut.aten_op,
        exir_op=IndexPut.exir_op,
        transform_passes=[
            InsertInt32CastsAfterInt64PlaceholdersPass(),
        ],  # int64 inputs are not currently supported and need to be cast to int32
    )
    pipeline.run()


@common.parametrize("test_module", test_data_suite, xfails=xfails)
def test_index_put_tosa_INT(test_module: input_t):
    pipeline = TosaPipelineINT[input_t](
        IndexPut(),
        test_module[0](),
        aten_op=IndexPut.aten_op,
        exir_op=IndexPut.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_suite)
def test_index_put_u55_INT(test_module: input_t):
    # SCATTER op is not supported on U55
    pipeline = OpNotSupportedPipeline[input_t](
        IndexPut(),
        test_module[0](),
        {IndexPut.exir_op: 1},
        quantize=True,
        u55_subset=True,
        n_expected_delegates=test_module[1],
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_module", test_data_suite)
def test_index_put_u85_INT(test_module: input_t):
    """same_index test case already supported on u85 even though it is not supported by TOSA spec.
    This is because the SCATTER is converted to a DMA op where the destination is specificed by the index and each index
    leads to one copy to the output tensor. It has been implemented so because there is no realistic way of detecting
    repeat indices at runtime.
    """
    pipeline = EthosU85PipelineINT[input_t](
        IndexPut(),
        test_module[0](),
        aten_ops=IndexPut.aten_op,
        exir_ops=IndexPut.exir_op,
    )
    # The indices arg tensors have to be cast to int32 and the _to_dim_order_copy op is not delegated as it has int64 inputs, therefore portable ops have to be used.
    pipeline.tester.use_portable_ops = True
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite, xfails=xfails)
def test_index_put_vgf_no_quant(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        IndexPut(),
        test_module[0](),
        aten_op=IndexPut.aten_op,
        exir_op=IndexPut.exir_op,
        transform_passes=[
            InsertInt32CastsAfterInt64PlaceholdersPass(),
        ],  # int64 inputs are not currently supported and need to be cast to int32
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite, xfails=xfails)
def test_index_put_vgf_quant(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        IndexPut(),
        test_module[0](),
        aten_op=IndexPut.aten_op,
        exir_op=IndexPut.exir_op,
    )
    pipeline.run()
