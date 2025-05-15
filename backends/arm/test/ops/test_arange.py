# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t = tuple[torch.Tensor]
test_data_t = tuple[Callable[[], input_t], tuple[float, float, float, torch.dtype]]


class ArangeAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.arange.start_step"
    exir_op: str = "executorch_exir_dialects_edge__ops_aten_arange_start_step"

    def __init__(self, start: float, stop: float, step: float, dtype: torch.dtype):
        super().__init__()
        self.args = (start, stop, step)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arange(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10": (lambda: (torch.randn(10, 1),), (0.0, 10.0, 1.0, torch.float32)),
        "15": (lambda: (torch.randn(10),), (0.0, 15.0, 1.5, torch.float32)),
        "100": (lambda: (torch.randn(10, 1),), (0.0, 10.0, 0.1, torch.float32)),
    }

    test_data_dtypes: dict[str, test_data_t] = {
        "fp32_int32": (lambda: (torch.randn(10),), (0.0, 10.0, 1.0, torch.int32)),
        "fp32_int64": (lambda: (torch.randn(10),), (0.0, 10.0, 1.0, torch.int64)),
        "int32_int32": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (0.0, 10.0, 1.0, torch.int32),
        ),
        "int32_int64": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (0.0, 10.0, 1.0, torch.int64),
        ),
    }


@common.parametrize("test_data", ArangeAdd.test_data)
def test_arange_start_step_tosa_MI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineMI[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
        ArangeAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data_dtypes)
def test_arange_start_step_tosa_MI_dtypes(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineMI[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
        ArangeAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
def test_arange_start_step_tosa_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineBI[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
        ArangeAdd.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
def test_arange_start_step_u55_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU55PipelineBI[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", ArangeAdd.test_data)
def test_arange_start_step_u85_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU85PipelineBI[input_t](
        ArangeAdd(*init_data),
        input_data(),
        ArangeAdd.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


class LinspaceAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.linspace.default"
    exir_op: str = "executorch_exir_dialects_edge__ops_aten_arange_default"

    def __init__(self, start: float, stop: float, step: int, dtype: torch.dtype):
        super().__init__()
        self.args = (start, stop, step)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linspace(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10": (lambda: (torch.randn(10, 1),), (0.0, 10.0, 100, torch.float32)),
        "15": (lambda: (torch.randn(20),), (0.0, 15.0, 20, torch.float32)),
    }


@common.parametrize("test_data", LinspaceAdd.test_data)
def test_linspace_tosa_MI(test_data):
    input_data, init_data = test_data
    pipeline = TosaPipelineMI[input_t](
        LinspaceAdd(*init_data),
        input_data(),
        LinspaceAdd.aten_op,
        LinspaceAdd.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", LinspaceAdd.test_data)
def test_linspace_tosa_BI(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineBI[input_t](
        LinspaceAdd(*init_data),
        input_data(),
        LinspaceAdd.aten_op,
        LinspaceAdd.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
