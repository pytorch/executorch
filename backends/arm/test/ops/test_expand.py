# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the expand op which copies the data of the input tensor (possibly with new data format)
#


from typing import Sequence, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.expand.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_expand_copy_default"

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, Input y


class Expand(torch.nn.Module):
    # (input tensor, multiples)
    base_test_set = {
        "rand_1d_both": lambda: (torch.rand(1), (2,)),
        "rand_1d": lambda: (torch.randn(1), (2, 2, 4)),
        "rand_4d": lambda: (torch.randn(1, 1, 1, 5), (1, 4, -1, -1)),
        "rand_batch_1": lambda: (torch.randn(1, 1), (1, 2, 2, 4)),
        "rand_batch_2": lambda: (torch.randn(1, 1), (2, 2, 2, 4)),
        "rand_mix_neg": lambda: (torch.randn(10, 1, 1, 97), (-1, 4, -1, -1)),
        "rand_small_neg": lambda: (torch.rand(1, 1, 2, 2), (4, 3, -1, 2)),
    }

    test_u55_reject_set = {
        "randbool_1d": lambda: (torch.randint(0, 2, (1,), dtype=torch.bool), (5,)),
    }
    test_reject_set = {
        "rand_2d": lambda: (torch.randn(1, 4), (1, -1)),
        "rand_neg_mul": lambda: (torch.randn(1, 1, 192), (1, -1, -1)),
    }
    test_parameters = base_test_set | test_u55_reject_set

    def forward(self, x: torch.Tensor, m: Sequence):
        return x.expand(m)


@common.parametrize("test_data", Expand.test_parameters)
def test_expand_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_parameters)
def test_expand_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    Expand.base_test_set,
)
@common.XfailIfNoCorstone300
def test_expand_u55_INT(test_data: Tuple):
    inputs = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        Expand(),
        inputs,
        aten_op,
        exir_ops=[],
    )
    if inputs[0].dtype == torch.bool:
        pipeline.pop_stage("check_count.exir")
        pipeline.tester.use_portable_ops = True
    pipeline.run()


@common.parametrize("test_data", Expand.test_parameters)
@common.XfailIfNoCorstone320
def test_expand_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_parameters)
@common.SkipIfNoModelConverter
def test_expand_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_parameters)
@common.SkipIfNoModelConverter
def test_expand_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_reject_set)
def test_expand_tosa_INT_not_delegated(test_data: Tuple):
    pipeline = OpNotSupportedPipeline[input_t1](
        Expand(), test_data(), {exir_op: 1}, n_expected_delegates=0, quantize=True
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_u55_reject_set)
def test_expand_u55_INT_not_delegated(test_data: Tuple):
    pipeline = OpNotSupportedPipeline[input_t1](
        Expand(),
        test_data(),
        {exir_op: 1},
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()
