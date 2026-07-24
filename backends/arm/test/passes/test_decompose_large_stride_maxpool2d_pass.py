# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch
from executorch.backends.arm._passes.decompose_large_stride_maxpool2d_pass import (
    DecomposeLargeStrideMaxPool2dPass,
)
from executorch.backends.arm._passes.remove_getitem_pass import RemoveGetItemPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]


class MaxPool1d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool1d(x, kernel_size=5, stride=5)


class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )


def _run_pass(
    module: torch.nn.Module, inputs: input_t, expected_pool_count: int
) -> None:
    pipeline = PassPipeline[input_t](
        module,
        inputs,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_default": expected_pool_count,
        },
        pass_list=[RemoveGetItemPass, DecomposeLargeStrideMaxPool2dPass],
    )
    with patch(
        "executorch.backends.arm._passes.decompose_large_stride_maxpool2d_pass.get_context_spec",
        return_value=SimpleNamespace(is_U55_subset=True),
    ):
        pipeline.run()


def test_decompose_large_stride_max_pool1d() -> None:
    _run_pass(MaxPool1d(), (torch.randn(1, 3, 17),), 2)


def test_decompose_large_square_stride_max_pool2d() -> None:
    _run_pass(MaxPool2d((5, 5), (5, 5)), (torch.randn(1, 3, 13, 17),), 2)


def test_decompose_large_rectangular_stride_max_pool2d() -> None:
    _run_pass(MaxPool2d((4, 7), (4, 7)), (torch.randn(1, 2, 11, 23),), 2)


def test_keep_overlapping_large_stride_max_pool2d() -> None:
    _run_pass(MaxPool2d((6, 6), (4, 4)), (torch.randn(1, 2, 15, 15),), 1)


def test_keep_supported_scalar_pool_attributes() -> None:
    _run_pass(MaxPool2d(2, 2), (torch.randn(1, 2, 15, 15),), 1)
