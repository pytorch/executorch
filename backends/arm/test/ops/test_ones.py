# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t = tuple[torch.Tensor]
test_data_t = tuple[int, torch.dtype]


class OnesAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.ones.default"

    def __init__(self, n: int, dtype: torch.dtype):
        super().__init__()
        self.args = (n,)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10x3x3": (lambda: (torch.randn(10, 3, 3),), (3, torch.float32)),
        "10x1": (lambda: (torch.randn(10, 1),), (10, torch.float32)),
        "int32_int32": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (10, torch.int32),
        ),
    }

    test_data_not_delegated: dict[str, test_data_t] = {
        "fp32_int64": (lambda: (torch.randn(10),), (10, torch.int64)),
        "fp32_int32": (lambda: (torch.randn(10),), (10, torch.int32)),
        "int32_int64": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (10, torch.int64),
        ),
    }


@common.parametrize("test_data", OnesAdd.test_data)
def test_ones_tosa_MI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineMI[input_t](
        OnesAdd(*init_data),
        input_data(),
        OnesAdd.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", OnesAdd.test_data)
def test_ones_tosa_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineBI[input_t](
        OnesAdd(*init_data),
        input_data(),
        OnesAdd.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", OnesAdd.test_data)
def test_ones_u55_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU55PipelineBI[input_t](
        OnesAdd(*init_data),
        input_data(),
        OnesAdd.aten_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", OnesAdd.test_data)
def test_ones_u85_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU85PipelineBI[input_t](
        OnesAdd(*init_data),
        input_data(),
        OnesAdd.aten_op,
        use_to_edge_transform_and_lower=True,
    ).dump_artifact("to_edge_transform_and_lower")
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize(
    "test_data",
    OnesAdd.test_data_not_delegated,
    xfails={
        "fp32_int32": "MLETORCG-716: Do not delegate empty networks to vela",
        "fp32_int64": "MLETORCG-716: Do not delegate empty networks to vela",
        "int32_int64": "MLETORCG-716: Do not delegate empty networks to vela",
    },
)
def test_ones_tosa_BI_not_delegated(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = OpNotSupportedPipeline[input_t](
        OnesAdd(*init_data), input_data(), non_delegated_ops={}, quantize=True
    )
    pipeline.run()
