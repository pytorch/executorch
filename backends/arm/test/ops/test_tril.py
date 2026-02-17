# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import pytest
import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]


class Tril(torch.nn.Module):
    aten_op = "torch.ops.aten.tril.default"
    exir_op_tril = "executorch_exir_dialects_edge__ops_aten_tril"
    exir_op_where = "executorch_exir_dialects_edge__ops_aten_where_self"

    def __init__(self, input: torch.Tensor, diagonal: int = 0):
        super().__init__()
        self.input_ = input
        self.diagonal_ = diagonal

    def forward(self, input_: torch.Tensor):
        if self.diagonal_ == 0:
            return torch.tril(input_)
        return torch.tril(input_, self.diagonal_)

    def get_inputs(self) -> input_t:
        return (self.input_,)


TRIL_DECOMP_OP = "torch.ops.aten.where.self"

op_tril_rank2 = Tril(torch.rand(2, 5), diagonal=0)
op_tril_rank3 = Tril(torch.randn(10, 5, 2), diagonal=-1)
op_tril_rank4 = Tril(torch.randn(3, 2, 2, 2), diagonal=1)
op_tril_rank_square = Tril(torch.rand(4, 4), diagonal=2)

test_data: Dict[str, Callable[[], Tril]] = {
    "tril_rank2": lambda: op_tril_rank2,
    "tril_rank3": lambda: op_tril_rank3,
    "tril_rank4": lambda: op_tril_rank4,
    "tril_rank_square": lambda: op_tril_rank_square,
}


@common.parametrize("test_module", test_data)
def test_tril_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module(),
        test_module().get_inputs(),
        Tril.aten_op,
        Tril.exir_op_tril,
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
def test_tril_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        TRIL_DECOMP_OP,
        Tril.exir_op_where,
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
def test_tril_tosa_INT_a16w8(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        TRIL_DECOMP_OP,
        Tril.exir_op_where,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
@pytest.mark.xfail(
    reason=("TRIL INT decomposition is not supported on U55"),
    strict=True,
)
def test_tril_u55_INT_not_supported(test_module):
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {Tril.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
@common.XfailIfNoCorstone320
def test_tril_u85_INT(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        [],
        TRIL_DECOMP_OP,
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
@common.XfailIfNoCorstone320
def test_tril_16a8w_u85_INT16(test_module):
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        TRIL_DECOMP_OP,
        Tril.exir_op_where,
        per_channel_quantization=per_channel_quantization,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
@common.SkipIfNoModelConverter
def test_tril_vgf_no_quant(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        Tril.aten_op,
        Tril.exir_op_tril,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_module", test_data)
@common.SkipIfNoModelConverter
def test_tril_vgf_quant(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        TRIL_DECOMP_OP,
        Tril.exir_op_where,
        quantize=True,
    )
    pipeline.run()
