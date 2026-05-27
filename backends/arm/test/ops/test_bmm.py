# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch

from executorch.backends.arm.quantizer import get_symmetric_a16w8_quantization_config
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op_bmm = "torch.ops.aten.bmm.default"
exir_op_bmm = "executorch_exir_dialects_edge__ops_aten_bmm_default"

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
    )
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMM(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    BMMSingleInput.test_data_generators,
    flakies={"rand_big_1": 3},
)
@common.SkipIfNoModelConverter
def test_bmm_vgf_no_quant_single_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMMSingleInput(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", BMM.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMM(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", BMMSingleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_bmm_vgf_quant_single_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        BMMSingleInput(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        quantize=True,
    )
    pipeline.run()


a16w8_bmm_test_parameters = {
    "rand_same": lambda: (torch.rand(2, 1, 1), torch.rand(2, 1, 1)),
    "rand_diff": lambda: (torch.rand(5, 3, 5), torch.rand(5, 5, 2)),
    "rand_rect": lambda: (torch.rand(1, 55, 3), torch.rand(1, 3, 44)),
    "rand_batch10": lambda: (torch.rand(10, 1, 10), torch.rand(10, 10, 5)),
    "rand_neg": lambda: (
        -10 * torch.randn(2, 32, 64),
        5 + 5 * torch.randn(2, 64, 32),
    ),
}


@common.parametrize("test_data", a16w8_bmm_test_parameters)
@common.XfailIfNoCorstone300
def test_bmm_a16w8_u55_INT(test_data: input_t1):
    """U55 does not support bmm with INT16 inputs.

    Verify bmm is rejected.

    """
    pipeline = OpNotSupportedPipeline[input_t1](
        BMM(),
        test_data(),
        non_delegated_ops={exir_op_bmm: 1},
        n_expected_delegates=0,
        u55_subset=True,
        quantize=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(get_symmetric_a16w8_quantization_config())
    pipeline.run()


@common.parametrize("test_data", a16w8_bmm_test_parameters)
@common.XfailIfNoCorstone320
def test_bmm_a16w8_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        BMM(),
        test_data(),
        aten_op_bmm,
        exir_op_bmm,
        a16w8_quantization=True,
        symmetric_io_quantization=True,
        qtol=1,
        epsilon=2**-16,
    )
    pipeline.run()
