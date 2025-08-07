# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class SliceSimple(torch.nn.Module):
    def __init__(self, index=1):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class SliceRange(torch.nn.Module):
    def forward(self, x):
        return x[1:3]


class SliceMultiDim2D(torch.nn.Module):
    def forward(self, x):
        return x[2:6, 4:12]


class SliceMultiDim3D(torch.nn.Module):
    def forward(self, x):
        return x[1:4, 2:8, 3:15]


class SliceMultiDim4D(torch.nn.Module):
    def forward(self, x):
        return x[0:2, 1:4, 2:6, 3:12]


class SliceMultiDimMixed(torch.nn.Module):
    def forward(self, x):
        # Mix of single indices and ranges
        return x[1, 2:8, 3:15]


@operator_test
class Slice(OperatorTest):
    @dtype_test
    def test_slice_simple_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            SliceSimple().to(dtype),
            (torch.rand(8, 16, 20).to(dtype),),
            flow,
        )

    def test_slice_range(self, flow: TestFlow) -> None:
        self._test_op(
            SliceRange(),
            (torch.rand(8, 32, 16),),
            flow,
        )

    def test_slice_multi_dimensions(self, flow: TestFlow) -> None:
        # Test 2D multi-dimensional slicing
        self._test_op(
            SliceMultiDim2D(),
            (torch.randn(12, 20),),
            flow,
        )

        # Test 3D multi-dimensional slicing
        self._test_op(
            SliceMultiDim3D(),
            (torch.randn(8, 12, 20),),
            flow,
        )

        # Test 4D multi-dimensional slicing
        self._test_op(
            SliceMultiDim4D(),
            (torch.randn(4, 8, 12, 16),),
            flow,
        )

        # Test mixed slicing (single index + ranges)
        self._test_op(
            SliceMultiDimMixed(),
            (torch.randn(4, 12, 20),),
            flow,
        )

    def test_slice_different_patterns(self, flow: TestFlow) -> None:
        # Test various slicing patterns on larger tensors

        # Pattern 1: Start from beginning
        class SliceFromStart(torch.nn.Module):
            def forward(self, x):
                return x[:4, :8, 2:16]

        self._test_op(
            SliceFromStart(),
            (torch.randn(8, 12, 20),),
            flow,
        )

        # Pattern 2: Slice to end
        class SliceToEnd(torch.nn.Module):
            def forward(self, x):
                return x[2:, 4:, 1:]

        self._test_op(
            SliceToEnd(),
            (torch.randn(8, 12, 16),),
            flow,
        )

        # Pattern 3: Step slicing on multiple dimensions
        class SliceWithStep(torch.nn.Module):
            def forward(self, x):
                return x[::2, 1::2, 2::3]

        self._test_op(
            SliceWithStep(),
            (torch.randn(12, 16, 24),),
            flow,
        )

        # Pattern 4: Negative indices
        class SliceNegative(torch.nn.Module):
            def forward(self, x):
                return x[-6:-2, -12:-4, -16:-2]

        self._test_op(
            SliceNegative(),
            (torch.randn(10, 16, 20),),
            flow,
        )
