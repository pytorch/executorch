# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the view op which changes the size of a Tensor without changing the underlying data.
#

from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    TOSAQuantizer,
)

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize

aten_op = "torch.ops.aten.view.default"

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x,  Input y


class View(torch.nn.Module):

    needs_transpose_tests = {
        "rand_1d_neg": lambda: (torch.rand(100), (1, -1, 5, 2)),
        "rand_4d_neg": lambda: (torch.rand(10, 2, 1, 5), (1, -1, 5, 2)),
        "rand_4d_4d_small": lambda: (torch.rand(1, 2, 1, 9), (3, 1, 3, 2)),
        "rand_4d_4d": lambda: (torch.rand(2, 1, 1, 9), (3, 2, 3, 1)),
        "rand_4d_2d": lambda: (torch.rand(2, 50, 2, 1), (1, 200)),
        "rand_4d_3d": lambda: (torch.rand(2, 5, 2, 3), (1, 15, 4)),
        "rand_4d_1": lambda: (torch.rand(2, 1, 1, 9), (3, 1, 3, 2)),
        "rand_4d_2": lambda: (torch.rand(5, 10, 1, 1), (25, 2, 1, 1)),
        "rand_4d_2_4": lambda: (torch.rand(10, 2), (1, 1, 5, 4)),
        "rand_4d_2_4_big": lambda: (torch.rand(10, 10), (5, 1, 5, 4)),
        "rand_4d_4_4": lambda: (torch.rand(1, 1, 1, 10), (1, 1, 10, 1)),
        "rand_4d_4_4_big": lambda: (torch.rand(1, 1, 5, 10), (1, 1, 50, 1)),
        "rand_4d_4_3": lambda: (torch.rand(5, 10, 1, 1), (1, 25, 2)),
        "rand_4d_4_2": lambda: (torch.rand(2, 50, 1, 1), (1, 100)),
        "rand_4d_2_4_same": lambda: (torch.rand(2, 3, 2, 3), (2, 3, 3, 2)),
        "rand_4d_5d": lambda: (torch.rand(1, 3, 4, 5), (1, 1, 4, 5, -1)),
        "rand_5d_5d": lambda: (torch.rand(1, 1, 4, 5, 6), (1, 1, 4, -1, 6)),
        "rand_5d_3d": lambda: (torch.rand(1, 1, 4, 5, 6), (2, 3, -1)),
        "rand_3d_5d": lambda: (torch.rand(4, 5, 6), (1, 1, 2, -1, 3)),
    }

    rank_product_too_large = {
        "rand_4d_large": lambda: (torch.rand(1, 49, 16, 128), (1, 16, 49, 128)),
    }

    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x: torch.Tensor):
        return x.view(self.new_shape)


@common.parametrize("test_data", View.needs_transpose_tests)
def test_view_tosa_FP(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = TosaPipelineFP[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
def test_view_tosa_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = TosaPipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.XfailIfNoCorstone300
def test_view_u55_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.SkipIfNoModelConverter
def test_view_vgf_no_quant(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = VgfPipeline[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.SkipIfNoModelConverter
def test_view_vgf_quant(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = VgfPipeline[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", View.rank_product_too_large)
@common.XfailIfNoCorstone300
def test_view_u55_INT_not_delegated(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        View(new_shape),
        (test_tensor,),
        {"executorch_exir_dialects_edge__ops_aten_view_copy": 1},
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.XfailIfNoCorstone320
def test_view_u85_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


def get_symmetric_a16w8_view_quantizer(per_channel_quantization=False):
    tosa_version = conftest.get_option("tosa_version")
    tosa_profiles = {
        "1.0": TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
    }

    quantizer = TOSAQuantizer(tosa_profiles[tosa_version])
    quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )

    return Quantize(
        quantizer,
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization
        ),
    )


@common.parametrize("test_data", View.needs_transpose_tests)
def test_view_16a8w_tosa_INT(test_data: Tuple):
    """Test view operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False
    test_tensor, new_shape = test_data()

    pipeline = TosaPipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_view_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.XfailIfNoCorstone300
def test_view_16a8w_u55_INT16(test_data: Tuple):
    """Test view operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False
    test_tensor, new_shape = test_data()

    pipeline = EthosU55PipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_view_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.XfailIfNoCorstone320
def test_view_16a8w_u85_INT16(test_data: Tuple):
    """Test view operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False
    test_tensor, new_shape = test_data()

    pipeline = EthosU85PipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_view_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()
