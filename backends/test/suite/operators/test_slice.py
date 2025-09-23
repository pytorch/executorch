# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


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


@parameterize_by_dtype
def test_slice_simple_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        SliceSimple().to(dtype),
        (torch.rand(8, 16, 20).to(dtype),),
    )


def test_slice_range(test_runner) -> None:
    test_runner.lower_and_run_model(
        SliceRange(),
        (torch.rand(8, 32, 16),),
    )


def test_slice_multi_dimensions(test_runner) -> None:
    # Test 2D multi-dimensional slicing
    test_runner.lower_and_run_model(
        SliceMultiDim2D(),
        (torch.randn(12, 20),),
    )

    # Test 3D multi-dimensional slicing
    test_runner.lower_and_run_model(
        SliceMultiDim3D(),
        (torch.randn(8, 12, 20),),
    )

    # Test 4D multi-dimensional slicing
    test_runner.lower_and_run_model(
        SliceMultiDim4D(),
        (torch.randn(4, 8, 12, 16),),
    )

    # Test mixed slicing (single index + ranges)
    test_runner.lower_and_run_model(
        SliceMultiDimMixed(),
        (torch.randn(4, 12, 20),),
    )


def test_slice_different_patterns(test_runner) -> None:
    # Test various slicing patterns on larger tensors

    # Pattern 1: Start from beginning
    class SliceFromStart(torch.nn.Module):
        def forward(self, x):
            return x[:4, :8, 2:16]

    test_runner.lower_and_run_model(
        SliceFromStart(),
        (torch.randn(8, 12, 20),),
    )

    # Pattern 2: Slice to end
    class SliceToEnd(torch.nn.Module):
        def forward(self, x):
            return x[2:, 4:, 1:]

    test_runner.lower_and_run_model(
        SliceToEnd(),
        (torch.randn(8, 12, 16),),
    )

    # Pattern 3: Step slicing on multiple dimensions
    class SliceWithStep(torch.nn.Module):
        def forward(self, x):
            return x[::2, 1::2, 2::3]

    test_runner.lower_and_run_model(
        SliceWithStep(),
        (torch.randn(12, 16, 24),),
    )

    # Pattern 4: Negative indices
    class SliceNegative(torch.nn.Module):
        def forward(self, x):
            return x[-6:-2, -12:-4, -16:-2]

    test_runner.lower_and_run_model(
        SliceNegative(),
        (torch.randn(10, 16, 20),),
    )
