# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)


class And(torch.nn.Module):
    aten_op = "torch.ops.aten.logical_and.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_and_default"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.logical_and(tensor2)


class Xor(torch.nn.Module):
    aten_op = "torch.ops.aten.logical_xor.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_xor_default"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.logical_xor(tensor2)


class Or(torch.nn.Module):
    aten_op = "torch.ops.aten.logical_or.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_logical_or_default"

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        return tensor1.logical_or(tensor2)


input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


test_input: dict[input_t2] = {
    "rank1": (
        torch.tensor([True, True, False, False], dtype=torch.bool),
        torch.tensor([True, False, True, False], dtype=torch.bool),
    ),
    "rand_rank2": (
        torch.randint(0, 2, (10, 10), dtype=torch.bool),
        torch.randint(0, 2, (10, 10), dtype=torch.bool),
    ),
    "rand_rank3": (
        torch.randint(0, 2, (10, 10, 10), dtype=torch.bool),
        torch.randint(0, 2, (10, 10, 10), dtype=torch.bool),
    ),
    "rand_rank4": (
        torch.randint(0, 2, (1, 10, 10, 10), dtype=torch.bool),
        torch.randint(0, 2, (1, 10, 10, 10), dtype=torch.bool),
    ),
}


test_data = {
    "and_rank1": (And(), test_input["rank1"]),
    "and_rand_rank2": (And(), test_input["rand_rank2"]),
    "and_rand_rank3": (And(), test_input["rand_rank3"]),
    "and_rand_rank4": (And(), test_input["rand_rank4"]),
    "xor_rank1": (Xor(), test_input["rank1"]),
    "xor_rand_rank2": (Xor(), test_input["rand_rank2"]),
    "xor_rand_rank3": (Xor(), test_input["rand_rank3"]),
    "xor_rand_rank4": (Xor(), test_input["rand_rank4"]),
    "or_rank1": (Or(), test_input["rank1"]),
    "or_rand_rank2": (Or(), test_input["rand_rank2"]),
    "or_rand_rank3": (Or(), test_input["rand_rank3"]),
    "or_rand_rank4": (Or(), test_input["rand_rank4"]),
}


fvp_xfails = {
    "and_rank1": "MLETORCH-706 Support ScalarType::Bool in EthosUBackend.",
    "and_rand_rank2": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "and_rand_rank3": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "and_rand_rank4": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "xor_rank1": "MLETORCH-706 Support ScalarType::Bool in EthosUBackend.",
    "xor_rand_rank2": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "xor_rand_rank3": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "xor_rand_rank4": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "or_rank1": "MLETORCH-706 Support ScalarType::Bool in EthosUBackend.",
    "or_rand_rank2": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "or_rand_rank3": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
    "or_rand_rank4": "MLETORCH-706: Support ScalarType::Bool in EthosUBackend.",
}


@common.parametrize("test_data", test_data)
def test_logical_tosa_MI(test_data: input_t2):
    op, test_input = test_data
    pipeline = TosaPipelineMI[input_t2](op, test_input, op.aten_op, op.exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_logical_tosa_BI(test_data: input_t2):
    op, test_input = test_data
    pipeline = TosaPipelineBI[input_t2](op, test_input, op.aten_op, op.exir_op)
    pipeline.pop_stage(pipeline.find_pos("quantize") + 1)
    pipeline.pop_stage("quantize")
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_logical_u55_BI(test_data: input_t2):
    # Tests that we don't delegate these ops since they are not supported on U55.
    op, test_input = test_data
    pipeline = OpNotSupportedPipeline[input_t2](
        op, test_input, "TOSA-0.80+BI+u55", {op.exir_op: 1}
    )
    pipeline.run()


@common.parametrize("test_data", test_data)
def test_logical_u85_BI(test_data: input_t2):
    op, test_input = test_data
    pipeline = EthosU85PipelineBI[input_t2](
        op, test_input, op.aten_op, op.exir_op, run_on_fvp=False
    )
    pipeline.pop_stage(pipeline.find_pos("quantize") + 1)
    pipeline.pop_stage("quantize")
    pipeline.run()


@common.parametrize("test_data", test_data, fvp_xfails)
@common.SkipIfNoCorstone320
def test_logical_u85_BI_on_fvp(test_data: input_t2):
    op, test_input = test_data
    pipeline = EthosU85PipelineBI[input_t2](
        op, test_input, op.aten_op, op.exir_op, run_on_fvp=True
    )
    pipeline.pop_stage(pipeline.find_pos("quantize") + 1)
    pipeline.pop_stage("quantize")
    pipeline.run()
