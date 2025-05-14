# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the squeeze op which squeezes a given dimension with size 1 into a lower ranked tensor.
#


from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t1 = Tuple[torch.Tensor]  # Input x


class SqueezeDim(torch.nn.Module):
    test_parameters = {
        "squeeze3d_dim_neg_2": lambda: (torch.randn(1, 1, 5), -2),
        "squeeze4d_dim_pos_3": lambda: (torch.randn(1, 2, 3, 1), 3),
        "squeeze4d_dim_neg_2": lambda: (torch.randn(1, 5, 1, 5), -2),
    }

    def forward(self, x: torch.Tensor, dim: int):
        return x.squeeze(dim)


class SqueezeDims(torch.nn.Module):
    test_parameters = {
        "squeeze3d_dims_0_1": lambda: (torch.randn(1, 1, 5), (0, 1)),
        "squeeze4d_dims_0_neg_1": lambda: (torch.randn(1, 5, 5, 1), (0, -1)),
        "squeeze4d_dims_0_neg_2": lambda: (torch.randn(1, 5, 1, 5), (0, -2)),
    }

    def forward(self, x: torch.Tensor, dims: tuple[int]):
        return x.squeeze(dims)


class Squeeze(torch.nn.Module):
    test_parameters = {
        "squeeze3d": lambda: (torch.randn(1, 1, 5),),
        "squeeze4d_dims": lambda: (torch.randn(1, 5, 5, 1),),
        "squeeze3d_dims_mix": lambda: (torch.randn(1, 5, 1, 5),),
    }

    def forward(self, x: torch.Tensor):
        return x.squeeze()


@common.parametrize("test_data", Squeeze.test_parameters)
def test_squeeze_dim_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        Squeeze(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.default",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
def test_squeeze_dim_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        Squeeze(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.default",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
@common.XfailIfNoCorstone300
def test_squeeze_dim_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Squeeze(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.default",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
@common.XfailIfNoCorstone320
def test_squeeze_dim_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Squeeze(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.default",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
def test_squeeze_dim_tosa_MI_2(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        SqueezeDim(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dim",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
def test_squeeze_dim_tosa_BI_2(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        SqueezeDim(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dim",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
@common.XfailIfNoCorstone300
def test_squeeze_dim_u55_BI_2(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        SqueezeDim(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dim",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
@common.XfailIfNoCorstone320
def test_squeeze_dim_u85_BI_2(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        SqueezeDim(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dim",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
def test_squeeze_dims_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        SqueezeDims(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dims",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
def test_squeeze_dims_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        SqueezeDims(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dims",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
@common.XfailIfNoCorstone300
def test_squeeze_dims_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        SqueezeDims(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dims",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
@common.XfailIfNoCorstone320
def test_squeeze_dims_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        SqueezeDims(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dims",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()
