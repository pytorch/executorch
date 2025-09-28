# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op_bmm = "torch.ops.aten.bmm.default"
exir_op_bmm = "executorch_exir_dialects_edge__ops_aten_bmm_default"

aten_op_mm = "torch.ops.aten.matmul.default"
exir_op_mm = "executorch_exir_dialects_edge__ops_aten_matmul_default"

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x


class BMM(torch.nn.Module):
    test_data_generators = {
        "rand_same": lambda: (torch.rand(2, 1, 1), torch.rand(2, 1, 1)),
        "rand_diff": lambda: (torch.rand(5, 3, 5), torch.rand(5, 5, 2)),
        "rand_ones": lambda: (torch.ones(1, 55, 3), torch.ones(1, 3, 44)),
        "rand_big": lambda: (10000 * torch.randn(10, 1, 10), torch.randn(10, 10, 5)),
        "rand_neg": lambda: (
            -10 * torch.randn(2, 32, 64),
            5 + 5 * torch.randn(2, 64, 32),
        ),
    }

    def forward(self, x, y):
        return torch.bmm(x, y)


class BMMSingleInput(torch.nn.Module):
    test_data_generators = {
        "rand_3d_1": lambda: (torch.rand(20, 3, 3),),
        "rand_3d_2": lambda: (torch.rand(2, 128, 128),),
        "rand_big_1": lambda: (10000 * torch.randn(4, 25, 25),),
        "rand_big_2": lambda: (5 + 5 * torch.randn(3, 64, 64),),
    }

    def forward(self, x):
        return torch.bmm(x, x)


@common.parametrize("test_data", BMM.test_data_generators)
def test_bmm_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](BMM(), test_data(), aten_op_bmm, exir_op_bmm)
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLETORCH-534)
@common.parametrize("test_data", BMMSingleInput.test_data_generators)
def test_bmm_tosa_FP_single_input(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        BMMSingleInput(), test_data(), aten_op_bmm, exir_op_bmm
    )
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
def test_bmm_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        BMM(), test_data(), aten_op_bmm, exir_op_bmm, qtol=1
    )
    pipeline.run()


@common.parametrize("test_data", BMMSingleInput.test_data_generators)
def test_bmm_tosa_INT_single_input(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        BMMSingleInput(), test_data(), aten_op_bmm, exir_op_bmm
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
@common.XfailIfNoCorstone300
def test_bmm_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        BMM(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
@common.XfailIfNoCorstone320
def test_bmm_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        BMM(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", BMMSingleInput.test_data_generators)
@common.XfailIfNoCorstone300
def test_bmm_u55_INT_single_input(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        BMMSingleInput(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", BMMSingleInput.test_data_generators)
@common.XfailIfNoCorstone320
def test_bmm_u85_INT_single_input(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        BMMSingleInput(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMM(), test_data(), aten_op_bmm, exir_op_bmm, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", BMMSingleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_FP_single_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMMSingleInput(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMM(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", BMMSingleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_INT_single_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMMSingleInput(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
