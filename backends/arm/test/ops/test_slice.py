# Copyright 2024-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.slice.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_slice_copy"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    "ones_neg_3": lambda: (torch.ones(10), [(3, -3)]),
    "ones_neg_8": lambda: (torch.ones(10), [(-8, 3)]),
    "ones_slice_2": lambda: (torch.ones(10, 10), [(1, 3), (3, 10)]),
    "ones_slice_3": lambda: (torch.ones(10, 10, 10), [(0, 7), (0, 10), (0, 8)]),
    "ones_slice_4": lambda: (
        torch.ones((1, 12, 10, 10)),
        [(0, 1), (0, 5), (3, 5), (4, 10)],
    ),
}


class Slice(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: list[tuple[int, int]]):
        slices = [slice(*i) for i in s]
        return x[slices]


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](Slice(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_INT_nchw(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_INT_nhwc(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_slice_tensor_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_slice_tensor_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test slice operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_slice_tensor_16a8w_u55_INT(test_data: torch.Tensor):
    """Test slice operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_slice_tensor_16a8w_u85_INT(test_data: torch.Tensor):
    """Test slice operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


####################################
## Non-unit step / stride slicing ##
####################################

input_t_step = Tuple[torch.Tensor, int, int, int, int]  # (x, dim, start, end, step)


test_data_step_fp = {
    # x[0:10:2] == x[::2]
    "arange_fp32_1d_step2": lambda: (
        torch.arange(10, dtype=torch.float32),
        0,
        0,
        10,
        2,
    ),
    # x[:, 1:10:4]
    "arange_fp32_2d_step4": lambda: (
        torch.arange(40, dtype=torch.float32).reshape(4, 10),
        1,
        1,
        10,
        4,
    ),
    # x[:, 0:4:2, :]
    "arange_fp32_3d_dim1_step2": lambda: (
        torch.arange(2 * 4 * 17, dtype=torch.float32).reshape(2, 4, 17),
        1,
        0,
        4,
        2,
    ),
    # x[:, :, :, 0:17:4]
    "arange_fp32_4d_dim3_step4": lambda: (
        torch.arange(2 * 3 * 5 * 17, dtype=torch.float32).reshape(2, 3, 5, 17),
        3,
        0,
        17,
        4,
    ),
    # x[:, 0:12:4]
    "bool_2d_step4": lambda: (
        (torch.rand((2, 12)) < 0.5),  # [2,12], dtype=bool
        1,
        0,
        12,
        4,
    ),
}

test_data_step_int = {
    # x[:, 0:9:3]
    "rand_int8_2d_step3": lambda: (
        torch.randint(-8, 8, size=(3, 9), dtype=torch.int8),
        1,
        0,
        9,
        3,
    ),
    # x[:, 0:6:2, :]
    "arange_int32_3d_step2_dim1": lambda: (
        torch.arange(2 * 6 * 4, dtype=torch.int32).reshape(2, 6, 4),
        1,
        0,
        6,
        2,
    ),
    # x[:, :, :, 0:19:4]
    "arange_int8_4d_dim3_step4": lambda: (
        torch.arange(2 * 2 * 4 * 19, dtype=torch.int8).reshape(2, 2, 4, 19),
        3,
        0,
        19,
        4,
    ),
    # x[:, 0:12:4]
    "bool_2d_step4": lambda: (
        (torch.rand((2, 12)) < 0.5),  # [2,12], dtype=bool
        1,
        0,
        12,
        4,
    ),
}


class SliceWithStep(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, dim_: int, start_: int, end_: int, step_: int
    ) -> torch.Tensor:
        # Use aten.slice to generate a slice_copy in Edge for lowering.
        return torch.ops.aten.slice.Tensor(x, dim_, start_, end_, step_)


@common.parametrize("test_data", test_data_step_fp)
def test_slice_tensor_tosa_FP_step(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t_step](
        SliceWithStep(),
        test_data(),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_step_int | test_data_step_fp)
def test_slice_tensor_tosa_INT_step(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t_step](
        SliceWithStep(),
        test_data(),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_step_int | test_data_step_fp,
    xfails={
        "bool_2d_step4": "MLETORCH-1744: bool test fails",
    },
)
@common.XfailIfNoCorstone300
def test_slice_tensor_u55_INT_step(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        SliceWithStep(),
        test_data(),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_step_int | test_data_step_fp)
@common.XfailIfNoCorstone320
def test_slice_tensor_u85_INT_step(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        SliceWithStep(),
        test_data(),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_step_int | test_data_step_fp)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_no_quant_step(test_data: Tuple):
    pipeline = VgfPipeline[input_t_step](
        SliceWithStep(),
        test_data(),
        aten_op=aten_op,
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_step_int | test_data_step_fp)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_quant_step(test_data: Tuple):
    pipeline = VgfPipeline[input_t_step](
        SliceWithStep(),
        test_data(),
        aten_op=aten_op,
        exir_op=exir_op,
        quantize=True,
    )
    pipeline.run()
