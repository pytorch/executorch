# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch
from executorch.backends.arm._passes.decompose_large_stride_maxpool2d_pass import (
    can_decompose_large_stride_maxpool2d,
    DecomposeLargeStrideMaxPool2dForU55Pass,
)
from executorch.backends.arm._passes.remove_getitem_pass import RemoveGetItemPass
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    PassPipeline,
)
from executorch.exir import EdgeCompileConfig, to_edge

input_t = Tuple[torch.Tensor]

_GET_CONTEXT_SPEC_PATCH = (
    "executorch.backends.arm._passes.decompose_large_stride_maxpool2d_pass."
    "get_context_spec"
)


class MaxPool1d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool1d(x, kernel_size=5, stride=5)


class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None,
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


def _transformed_module(
    module: torch.nn.Module, inputs: input_t, is_u55: bool
) -> torch.nn.Module:
    exported_program = torch.export.export(module.eval(), inputs, strict=True)
    edge_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    with patch(
        _GET_CONTEXT_SPEC_PATCH,
        return_value=SimpleNamespace(is_U55_subset=is_u55),
    ):
        transformed = edge_program.transform(
            [RemoveGetItemPass(), DecomposeLargeStrideMaxPool2dForU55Pass()]
        )
    return transformed.exported_program().module()


def _assert_pass_matches_eager(
    module: torch.nn.Module, inputs: input_t, is_u55: bool = True
) -> None:
    torch.testing.assert_close(
        module.eval()(*inputs),
        _transformed_module(module, inputs, is_u55)(*inputs),
    )


def _run_pass(
    module: torch.nn.Module,
    inputs: input_t,
    expected_pool_count: int,
    is_u55: bool = True,
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
        pass_list=[RemoveGetItemPass, DecomposeLargeStrideMaxPool2dForU55Pass],
    )
    with patch(
        _GET_CONTEXT_SPEC_PATCH,
        return_value=SimpleNamespace(is_U55_subset=is_u55),
    ):
        pipeline.run()
    _assert_pass_matches_eager(module, inputs, is_u55)


def test_decompose_large_stride_max_pool1d() -> None:
    _run_pass(MaxPool1d(), (torch.randn(1, 3, 17),), 2)


def test_decompose_large_square_stride_max_pool2d() -> None:
    _run_pass(MaxPool2d((5, 5), (5, 5)), (torch.randn(1, 3, 13, 17),), 2)


def test_decompose_large_rectangular_stride_max_pool2d() -> None:
    _run_pass(MaxPool2d((4, 7), (4, 7)), (torch.randn(1, 2, 11, 23),), 2)


def test_decompose_large_stride_max_pool2d_with_default_stride() -> None:
    _run_pass(MaxPool2d((5, 5), None), (torch.randn(1, 3, 13, 17),), 2)


def test_keep_large_stride_max_pool2d_for_non_u55() -> None:
    _run_pass(
        MaxPool2d((5, 5), (5, 5)),
        (torch.randn(1, 3, 13, 17),),
        1,
        is_u55=False,
    )


def test_keep_overlapping_large_stride_max_pool2d() -> None:
    _run_pass(MaxPool2d((6, 6), (4, 4)), (torch.randn(1, 2, 15, 15),), 1)


def test_keep_supported_scalar_pool_attributes() -> None:
    _run_pass(MaxPool2d(2, 2), (torch.randn(1, 2, 15, 15),), 1)


def test_reject_pool_exceeding_u55_original_dim_limit() -> None:
    assert not can_decompose_large_stride_maxpool2d(
        (5, 5),
        (5, 5),
        (0, 0),
        (1, 1),
        False,
        (1, 1, 65537, 25),
    )


def test_reject_pool_exceeding_u55_intermediate_dim_limit() -> None:
    assert not can_decompose_large_stride_maxpool2d(
        (4, 4),
        (4, 4),
        (0, 0),
        (1, 1),
        False,
        (1, 1, 1028, 1028),
    )


def test_decompose_large_stride_max_pool2d_u55_INT_pipeline() -> None:
    inputs = (torch.randn(1, 3, 13, 17),)
    pipeline = EthosU55PipelineINT[input_t](
        MaxPool2d((5, 5), (5, 5)),
        inputs,
        [],
        [],
        run_on_fvp=False,
    )
    pipeline.pop_stage("check_not.exir")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("to_executorch")
    pipeline.run()
    _assert_pass_matches_eager(MaxPool2d((5, 5), (5, 5)), inputs)
