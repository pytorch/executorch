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
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]  # Input x


class SqueezeDim(torch.nn.Module):
    test_parameters = {
        "squeeze3d_dim_neg_2": lambda: (torch.randn(1, 1, 5), -2),
        "squeeze4d_dim_pos_3": lambda: (torch.randn(1, 2, 3, 1), 3),
        "squeeze4d_dim_neg_2": lambda: (torch.randn(1, 5, 1, 5), -2),
        "squeeze5d_dim_neg_2": lambda: (torch.randn(1, 1, 5, 1, 5), -2),
    }

    def forward(self, x: torch.Tensor, dim: int):
        return x.squeeze(dim)


class SqueezeDims(torch.nn.Module):
    test_parameters = {
        "squeeze3d_dims_0_1": lambda: (torch.randn(1, 1, 5), (0, 1)),
        "squeeze4d_dims_0_neg_1": lambda: (torch.randn(1, 5, 5, 1), (0, -1)),
        "squeeze4d_dims_0_neg_2": lambda: (torch.randn(1, 5, 1, 5), (0, -2)),
        "squeeze5d_dims_0_neg_2": lambda: (torch.randn(1, 1, 5, 1, 5), (0, -2)),
    }

    def forward(self, x: torch.Tensor, dims: tuple[int]):
        return x.squeeze(dims)


class Squeeze(torch.nn.Module):
    test_parameters = {
        "squeeze3d": lambda: (torch.randn(1, 1, 5),),
        "squeeze4d_dims": lambda: (torch.randn(1, 5, 5, 1),),
        "squeeze3d_dims_mix": lambda: (torch.randn(1, 5, 1, 5),),
        "squeeze4d_dims_mix": lambda: (torch.randn(1, 1, 5, 1, 5),),
    }

    def forward(self, x: torch.Tensor):
        return x.squeeze()


##############
## Squeeze ###
##############


@common.parametrize("test_data", Squeeze.test_parameters)
def test_squeeze_dim_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Squeeze(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.default",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
def test_squeeze_dim_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Squeeze(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.default",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
@common.XfailIfNoCorstone300
def test_squeeze_dim_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Squeeze(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.default",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
@common.XfailIfNoCorstone320
def test_squeeze_dim_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Squeeze(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.default",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
@common.SkipIfNoModelConverter
def test_squeeze_dim_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Squeeze(),
        test_data(),
        "torch.ops.aten.squeeze.default",
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Squeeze.test_parameters)
@common.SkipIfNoModelConverter
def test_squeeze_dim_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Squeeze(),
        test_data(),
        "torch.ops.aten.squeeze.default",
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


#################
## SqueezeDim ###
#################


@common.parametrize("test_data", SqueezeDim.test_parameters)
def test_squeeze_dim_tosa_FP_2(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        SqueezeDim(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dim",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
def test_squeeze_dim_tosa_INT_2(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        SqueezeDim(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dim",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
@common.XfailIfNoCorstone300
def test_squeeze_dim_u55_INT_2(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        SqueezeDim(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dim",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
@common.XfailIfNoCorstone320
def test_squeeze_dim_u85_INT_2(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        SqueezeDim(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dim",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
@common.SkipIfNoModelConverter
def test_squeeze_dim_vgf_FP_2(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SqueezeDim(),
        test_data(),
        "torch.ops.aten.squeeze.dim",
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDim.test_parameters)
@common.SkipIfNoModelConverter
def test_squeeze_dim_vgf_INT_2(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SqueezeDim(),
        test_data(),
        "torch.ops.aten.squeeze.dim",
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


##################
## SqueezeDims ###
##################


@common.parametrize("test_data", SqueezeDims.test_parameters)
def test_squeeze_dims_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        SqueezeDims(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dims",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
def test_squeeze_dims_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        SqueezeDims(),
        test_data(),
        aten_op="torch.ops.aten.squeeze.dims",
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
@common.XfailIfNoCorstone300
def test_squeeze_dims_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        SqueezeDims(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dims",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
@common.XfailIfNoCorstone320
def test_squeeze_dims_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        SqueezeDims(),
        test_data(),
        aten_ops="torch.ops.aten.squeeze.dims",
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
@common.SkipIfNoModelConverter
def test_squeeze_dims_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SqueezeDims(),
        test_data(),
        "torch.ops.aten.squeeze.dims",
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", SqueezeDims.test_parameters)
@common.SkipIfNoModelConverter
def test_squeeze_dims_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        SqueezeDims(),
        test_data(),
        "torch.ops.aten.squeeze.dims",
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
