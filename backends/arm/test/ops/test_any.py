# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
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
    "rank1": (torch.tensor([True, False, False], dtype=torch.bool), 0, True),
    "rank1_squeeze": (torch.tensor([True, False, False], dtype=torch.bool), -1, False),
    "rank2": (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        0,
        True,
    ),
    "rank2_squeeze": (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        0,
        False,
    ),
    "rank2_dims": (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        [0, 1],
        True,
    ),
    "rank2_dims_squeeze": (
        torch.randint(0, 2, (2, 3), dtype=torch.bool),
        [-2, 1],
        False,
    ),
    "rank3_dims_squeeze": (
        torch.randint(0, 2, (6, 8, 10), dtype=torch.bool),
        [1, 2],
        False,
    ),
    "rank4": (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        1,
        True,
    ),
    "rank4_squeeze": (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        1,
        False,
    ),
    "rank4_dims": (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        [0, 2],
        True,
    ),
    "rank4_dims_squeeze": (
        torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),
        [1, -1],
        False,
    ),
    "rank1_reduce_all": (torch.tensor([True, False, False], dtype=torch.bool),),
    "rank2_reduce_all": (torch.randint(0, 2, (2, 3), dtype=torch.bool),),
    "rank3_reduce_all": (torch.randint(0, 2, (6, 8, 10), dtype=torch.bool),),
    "rank4_reduce_all": (torch.randint(0, 2, (1, 6, 8, 10), dtype=torch.bool),),
}


test_data = {
    "any_rank1": (AnyDim(), test_input["rank1"]),
    "any_rank1_squeeze": (AnyDim(), test_input["rank1_squeeze"]),
    "any_rank2": (AnyDim(), test_input["rank2"]),
    "any_rank2_squeeze": (AnyDim(), test_input["rank2_squeeze"]),
    "any_rank2_dims": (AnyDims(), test_input["rank2_dims"]),
    "any_rank2_dims_squeeze": (AnyDims(), test_input["rank2_dims_squeeze"]),
    "any_rank3_dims_squeeze": (AnyDims(), test_input["rank3_dims_squeeze"]),
    "any_rank4": (AnyDim(), test_input["rank4"]),
    "any_rank4_squeeze": (AnyDim(), test_input["rank4_squeeze"]),
    "any_rank4_dims": (AnyDims(), test_input["rank4_dims"]),
    "any_rank4_dims_squeeze": (AnyDims(), test_input["rank4_dims_squeeze"]),
    "any_rank1_reduce_all": (AnyReduceAll(), test_input["rank1_reduce_all"]),
    "any_rank2_reduce_all": (AnyReduceAll(), test_input["rank2_reduce_all"]),
    "any_rank3_reduce_all": (AnyReduceAll(), test_input["rank3_reduce_all"]),
    "any_rank4_reduce_all": (AnyReduceAll(), test_input["rank4_reduce_all"]),
}


fvp_xfails = {
    "any_rank1": "MLETORCH-706 Support ScalarType::Bool in EthosUBackend.",
    "any_rank1_squeeze": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank2": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank2_squeeze": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank2_dims": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank2_dims_squeeze": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank3_dims_squeeze": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank4": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank4_squeeze": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank4_dims": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank4_dims_squeeze": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank1_reduce_all": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank2_reduce_all": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank3_reduce_all": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "any_rank4_reduce_all": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
}


@common.parametrize("test_data", test_data)
def test_any_tosa_MI(test_data: input_t1):
    op, test_input = test_data
    pipeline = TosaPipelineMI[input_t1](op, test_input, op.aten_op, op.exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_any_tosa_BI(test_data: input_t1):
    op, test_input = test_data
    pipeline = TosaPipelineBI[input_t1](op, test_input, op.aten_op, op.exir_op)
    pipeline.pop_stage(pipeline.find_pos("quantize") + 1)
    pipeline.pop_stage("quantize")
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_logical_u55_BI(test_data: input_t1):
    # Tests that we don't delegate these ops since they are not supported on U55.
    op, test_input = test_data
    pipeline = OpNotSupportedPipeline[input_t1](
        op, test_input, "TOSA-0.80+BI+u55", {op.exir_op: 1}
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_floor_u85_BI(test_data: input_t1):
    op, test_input = test_data
    pipeline = EthosU85PipelineBI[input_t1](
        op, test_input, op.aten_op, op.exir_op, run_on_fvp=False
    )
    pipeline.pop_stage(pipeline.find_pos("quantize") + 1)
    pipeline.pop_stage("quantize")
    pipeline.run()


@common.parametrize("test_data", test_data, fvp_xfails)
@common.SkipIfNoCorstone320
def test_floor_u85_BI_on_fvp(test_data: input_t1):
    op, test_input = test_data
    pipeline = EthosU85PipelineBI[input_t1](
        op, test_input, op.aten_op, op.exir_op, run_on_fvp=True
    )
    pipeline.pop_stage(pipeline.find_pos("quantize") + 1)
    pipeline.pop_stage("quantize")
    pipeline.run()
