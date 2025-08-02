# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class AnyDim(torch.nn.Module):
    aten_op = "torch.ops.aten.any.dim"
    exir_op = "executorch_exir_dialects_edge__ops_aten_any_dim"

    def forward(self, x: torch.Tensor, dim: int, keepdim: bool):
        return torch.any(x, dim=dim, keepdim=keepdim)


class AnyDims(torch.nn.Module):
    aten_op = "torch.ops.aten.any.dims"
    exir_op = "executorch_exir_dialects_edge__ops_aten_any_dims"

    def forward(self, x: torch.Tensor, dim: List[int], keepdim: bool):
        return torch.any(x, dim=dim, keepdim=keepdim)


class AnyReduceAll(torch.nn.Module):
    aten_op = "torch.ops.aten.any.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_any_default"

    def forward(self, x: torch.Tensor):
        return torch.any(x)


input_t1 = Tuple[torch.Tensor]  # Input x


test_input: dict[input_t1] = {
    "rank1": lambda: (torch.tensor([True, False, False], dtype=torch.bool), 0, True),
    "rank1_squeeze": lambda: (
        torch.tensor([True, False, False], dtype=torch.bool),
        -1,
        False,
    ),
    "rank2": lambda: (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        0,
        True,
    ),
    "rank2_squeeze": lambda: (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        0,
        False,
    ),
    "rank2_dims": lambda: (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        [0, 1],
        True,
    ),
    "rank2_dims_squeeze": lambda: (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        [-2, 1],
        False,
    ),
    "rank3_dims_squeeze": lambda: (
        torch.randint(0, 2, (6, 8, 10), dtype=torch.bool),
        [1, 2],
        False,
    ),
    "rank4": lambda: (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        1,
        True,
    ),
    "rank4_squeeze": lambda: (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        1,
        False,
    ),
    "rank4_dims": lambda: (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        [0, 2],
        True,
    ),
    "rank4_dims_squeeze": lambda: (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        [1, -1],
        False,
    ),
    "rank1_reduce_all": lambda: (torch.tensor([True, False, False], dtype=torch.bool),),
    "rank2_reduce_all": lambda: (torch.randint(0, 2, (2, 3), dtype=torch.bool),),
    "rank3_reduce_all": lambda: (torch.randint(0, 2, (6, 8, 10), dtype=torch.bool),),
    "rank4_reduce_all": lambda: (torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),),
}


test_data = {
    "any_rank1": lambda: (AnyDim(), test_input["rank1"]),
    "any_rank1_squeeze": lambda: (AnyDim(), test_input["rank1_squeeze"]),
    "any_rank2": lambda: (AnyDim(), test_input["rank2"]),
    "any_rank2_squeeze": lambda: (AnyDim(), test_input["rank2_squeeze"]),
    "any_rank2_dims": lambda: (AnyDims(), test_input["rank2_dims"]),
    "any_rank2_dims_squeeze": lambda: (AnyDims(), test_input["rank2_dims_squeeze"]),
    "any_rank3_dims_squeeze": lambda: (AnyDims(), test_input["rank3_dims_squeeze"]),
    "any_rank4": lambda: (AnyDim(), test_input["rank4"]),
    "any_rank4_squeeze": lambda: (AnyDim(), test_input["rank4_squeeze"]),
    "any_rank4_dims": lambda: (AnyDims(), test_input["rank4_dims"]),
    "any_rank4_dims_squeeze": lambda: (AnyDims(), test_input["rank4_dims_squeeze"]),
    "any_rank1_reduce_all": lambda: (AnyReduceAll(), test_input["rank1_reduce_all"]),
    "any_rank2_reduce_all": lambda: (AnyReduceAll(), test_input["rank2_reduce_all"]),
    "any_rank3_reduce_all": lambda: (AnyReduceAll(), test_input["rank3_reduce_all"]),
    "any_rank4_reduce_all": lambda: (AnyReduceAll(), test_input["rank4_reduce_all"]),
}


@common.parametrize("test_data", test_data)
def test_any_tosa_FP(test_data: input_t1):
    op, test_input = test_data()
    pipeline = TosaPipelineFP[input_t1](
        op,
        test_input(),
        op.aten_op,
        op.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_any_tosa_INT(test_data: input_t1):
    op, test_input = test_data()
    pipeline = TosaPipelineINT[input_t1](
        op,
        test_input(),
        op.aten_op,
        op.exir_op,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_any_u55_INT(test_data: input_t1):
    # Tests that we don't delegate these ops since they are not supported on U55.
    op, test_input = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        op,
        test_input(),
        {op.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.XfailIfNoCorstone320
def test_any_u85_INT(test_data: input_t1):
    op, test_input = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        op,
        test_input(),
        op.aten_op,
        op.exir_op,
        run_on_fvp=True,
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoModelConverter
def test_any_vgf_FP(test_data: input_t1):
    op, data_fn = test_data()
    pipeline = VgfPipeline[input_t1](
        op,
        data_fn(),
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
@common.SkipIfNoModelConverter
def test_any_vgf_INT(test_data: input_t1):
    op, data_fn = test_data()
    pipeline = VgfPipeline[input_t1](
        op,
        data_fn(),
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("quantize")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
