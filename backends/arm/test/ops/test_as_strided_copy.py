# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.common.as_strided_utils import contiguous_strides

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


aten_op = "torch.ops.aten.as_strided_copy.default"
input_t = Tuple[torch.Tensor]


class AsStridedCopyModule(torch.nn.Module):
    def __init__(
        self,
        size: Tuple[int, ...],
        stride: Tuple[int, ...],
        storage_offset: int = 0,
    ):
        super().__init__()
        self.size = size
        self.stride = stride
        self.storage_offset = storage_offset

    def forward(self, x: torch.Tensor):
        y = torch.ops.aten.as_strided_copy.default(
            x, self.size, self.stride, self.storage_offset
        )
        return y


def _make_case(
    tensor_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    tensor = torch.rand(tensor_shape)
    stride = contiguous_strides(target_shape)
    return tensor, target_shape, stride


delegated_cases = {
    "reshape_2d": lambda: _make_case((4, 6), (3, 8)),
    "flatten": lambda: _make_case((2, 3, 4), (6, 4)),
    "expand_rank": lambda: _make_case((2, 3, 4), (2, 3, 4)),
}

unsupported_cases = {
    "non_contiguous_stride": lambda: (
        torch.rand(3, 3),
        (3, 3),
        (1, 3),  # Not a contiguous stride layout for (3, 3)
    ),
    "non_zero_offset": lambda: (
        torch.rand(4, 4),
        (4, 4),
        contiguous_strides((4, 4)),
        4,
    ),
}


@common.parametrize("test_data", delegated_cases)
def test_as_strided_copy_tosa_FP(test_data):
    tensor, size, stride = test_data()
    module = AsStridedCopyModule(size, stride)
    pipeline = TosaPipelineFP[input_t](
        module,
        (tensor,),
        aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", delegated_cases)
def test_as_strided_copy_tosa_INT(test_data):
    tensor, size, stride = test_data()
    module = AsStridedCopyModule(size, stride)
    pipeline = TosaPipelineINT[input_t](
        module,
        (tensor,),
        aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", delegated_cases)
@common.SkipIfNoModelConverter
def test_as_strided_copy_vgf_no_quant(test_data):
    tensor, size, stride = test_data()
    module = AsStridedCopyModule(size, stride)
    pipeline = VgfPipeline[input_t](
        module,
        (tensor,),
        aten_op,
        tosa_version="TOSA-1.0+FP",
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", delegated_cases)
@common.SkipIfNoModelConverter
def test_as_strided_copy_vgf_quant(test_data):
    tensor, size, stride = test_data()
    module = AsStridedCopyModule(size, stride)
    pipeline = VgfPipeline[input_t](
        module,
        (tensor,),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", unsupported_cases)
def test_as_strided_copy_not_delegated(test_data):
    tensor, size, stride, *rest = test_data()
    storage_offset = rest[0] if rest else 0
    module = AsStridedCopyModule(size, stride, storage_offset=storage_offset)
    pipeline = OpNotSupportedPipeline[input_t](
        module,
        (tensor,),
        {"executorch_exir_dialects_edge__ops_aten_as_strided_copy_default": 1},
        n_expected_delegates=0,
    )
    pipeline.run()
