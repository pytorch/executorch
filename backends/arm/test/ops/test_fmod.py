# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

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

aten_op_tensor = "torch.ops.aten.fmod.Tensor"
aten_op_scalar = "torch.ops.aten.fmod.Scalar"
exir_op_tensor = "executorch_exir_dialects_edge__ops_aten_fmod_Tensor"
exir_op_scalar = "executorch_exir_dialects_edge__ops_aten_fmod_Scalar"


class Fmod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor | float):
        return torch.fmod(x, y)


test_data_scalar = {
    "fmod_tensor_scalar_pos": lambda: (
        Fmod(),
        torch.tensor([[10.0, 25.5], [-33.2, 4.4]]),
        3.0,
    ),
    "fmod_tensor_scalar_neg": lambda: (
        Fmod(),
        torch.tensor([[10.0, -25.5], [-33.2, 4.4]]),
        -5.0,
    ),
    "fmod_tensor_scalar_one": lambda: (Fmod(), torch.randn(2, 3, 4), 1.0),
    "fmod_tensor_scalar_small_float": lambda: (
        Fmod(),
        torch.tensor([0.123, -0.456, 0.789]),
        0.1,
    ),
    "fmod_tensor_scalar_large_float": lambda: (
        Fmod(),
        torch.tensor([1e8, -1e9, 3.5e6]),
        1e6,
    ),
    "fmod_division_by_zero": lambda: (Fmod(), torch.randn(2, 3, 4), 0.0),
}


test_data_tensor = {
    "fmod_zeros": lambda: (
        Fmod(),
        torch.zeros(1, 10, 10, 10),
        torch.ones(1, 10, 10, 10),
    ),
    "fmod_ones": lambda: (Fmod(), torch.ones(1, 10, 10, 10), torch.ones(1, 10, 10, 10)),
    "fmod_rand": lambda: (Fmod(), torch.rand(10, 10) - 0.5, torch.rand(10, 10) + 0.5),
    "fmod_randn_pos": lambda: (
        Fmod(),
        torch.randn(1, 4, 4, 4) + 10,
        torch.randn(1, 4, 4, 4) + 10,
    ),
    "fmod_randn_neg": lambda: (
        Fmod(),
        torch.randn(1, 4, 4, 4) - 10,
        torch.randn(1, 4, 4, 4) + 10,
    ),
    "fmod_broadcast": lambda: (
        Fmod(),
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        torch.tensor([3.0, 7.0]),
    ),
    "fmod_negative_divisor": lambda: (
        Fmod(),
        torch.tensor([[10.0, -20.0], [-30.0, 40.0]]),
        torch.tensor([[-3.0, -5.0], [-7.0, -6.0]]),
    ),
    "fmod_division_by_zero": lambda: (
        Fmod(),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.0, 1.0, 2.0]),
    ),
    "fmod_mixed_signs": lambda: (
        Fmod(),
        torch.tensor([-10.0, 20.0, -30.0]),
        torch.tensor([3.0, -5.0, 7.0]),
    ),
    "fmod_scalar_tensor": lambda: (Fmod(), torch.tensor(10.0), torch.tensor(3.0)),
    "fmod_large_values": lambda: (
        Fmod(),
        torch.tensor([1e19, -1e21]),
        torch.tensor([3.0, 5.0]),
    ),
    "fmod_small_values": lambda: (
        Fmod(),
        torch.tensor([1e-10, -1e-12]),
        torch.tensor([1e-5, 2e-5]),
    ),
}

xfails = {"fmod_division_by_zero": "Invalid inputs not handled"}


@common.parametrize("test_data", test_data_scalar)
def test_fmod_scalar_tosa_FP(test_data: input_t):
    module, data_x, data_y = test_data()
    pipeline = TosaPipelineFP[input_t](
        module, (data_x, data_y), aten_op=aten_op_scalar, exir_op=exir_op_scalar
    )
    pipeline.run()


@common.parametrize("test_data", test_data_tensor)
def test_fmod_tensor_tosa_FP(test_data: input_t):
    module, data_x, data_y = test_data()
    pipeline = TosaPipelineFP[input_t](
        module, (data_x, data_y), aten_op=aten_op_tensor, exir_op=exir_op_tensor
    )
    pipeline.run()


@common.parametrize("test_data", test_data_scalar, xfails=xfails)
def test_fmod_scalar_tosa_INT(test_data: input_t):
    module, data_x, data_y = test_data()
    pipeline = TosaPipelineINT[input_t](
        module,
        (data_x, data_y),
        aten_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_tensor, xfails=xfails)
def test_fmod_tensor_tosa_INT(test_data: input_t):
    module, data_x, data_y = test_data()
    pipeline = TosaPipelineINT[input_t](
        module,
        (data_x, data_y),
        aten_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_scalar)
@common.XfailIfNoCorstone300
def test_fmod_scalar_u55_INT(test_data):
    module, data_x, data_y = test_data()
    pipeline = OpNotSupportedPipeline[input_t](
        module,
        (data_x, data_y),
        {
            exir_op_tensor: 1,
        },
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_tensor)
@common.XfailIfNoCorstone300
def test_fmod_tensor_u55_INT(test_data):
    module, data_x, data_y = test_data()
    pipeline = OpNotSupportedPipeline[input_t](
        module,
        (data_x, data_y),
        {
            exir_op_tensor: 1,
        },
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_scalar, xfails=xfails)
@common.XfailIfNoCorstone320
def test_fmod_scalar_u85_INT(test_data: input_t):
    module, data_x, data_y = test_data()
    pipeline = EthosU85PipelineINT[input_t](module, (data_x, data_y), aten_ops=[])
    pipeline.run()


@common.parametrize("test_data", test_data_tensor, xfails=xfails)
@common.XfailIfNoCorstone320
def test_fmod_tensor_u85_INT(test_data: input_t):
    module, data_x, data_y = test_data()
    pipeline = EthosU85PipelineINT[input_t](module, (data_x, data_y), aten_ops=[])
    pipeline.run()


@common.parametrize("test_data", test_data_scalar)
@common.SkipIfNoModelConverter
def test_fmod_scalar_vgf_FP(test_data: Tuple):
    module, data_x, data_y = test_data()
    pipeline = VgfPipeline[input_t](
        module,
        (data_x, data_y),
        [],
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_scalar, xfails=xfails)
@common.SkipIfNoModelConverter
def test_fmod_scalar_vgf_INT(test_data: Tuple):
    module, data_x, data_y = test_data()
    pipeline = VgfPipeline[input_t](
        module,
        (data_x, data_y),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_tensor)
@common.SkipIfNoModelConverter
def test_fmod_tensor_vgf_FP(test_data: Tuple):
    module, data_x, data_y = test_data()
    pipeline = VgfPipeline[input_t](
        module,
        (data_x, data_y),
        [],
        [],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_tensor, xfails=xfails)
@common.SkipIfNoModelConverter
def test_fmod_tensor_vgf_INT(test_data: Tuple):
    module, data_x, data_y = test_data()
    pipeline = VgfPipeline[input_t](
        module,
        (data_x, data_y),
        [],
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
