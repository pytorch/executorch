# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Optional

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class MedianModel(torch.nn.Module):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.median(x, dim=self.dim, keepdim=self.keepdim)


class MedianValueOnlyModel(torch.nn.Module):
    """Model that returns only the median values (not indices) when dim is specified."""

    def __init__(self, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is not None:
            return torch.median(x, dim=self.dim, keepdim=self.keepdim)[0]
        else:
            return torch.median(x)


@parameterize_by_dtype
def test_median_dtype(test_runner, dtype) -> None:
    # Test with different dtypes (global reduction)
    model = MedianValueOnlyModel().to(dtype)
    test_runner.lower_and_run_model(model, (torch.rand(10, 10).to(dtype),))


def test_median_basic(test_runner) -> None:
    # Basic test with default parameters (global reduction)
    test_runner.lower_and_run_model(MedianValueOnlyModel(), (torch.randn(10, 10),))


def test_median_dim(test_runner) -> None:
    # Test with different dimensions (values only)

    # 2D tensor, dim=0
    test_runner.lower_and_run_model(MedianValueOnlyModel(dim=0), (torch.randn(5, 10),))

    # 2D tensor, dim=1
    test_runner.lower_and_run_model(MedianValueOnlyModel(dim=1), (torch.randn(5, 10),))

    # 3D tensor, dim=0
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=0), (torch.randn(3, 4, 5),)
    )

    # 3D tensor, dim=1
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=1), (torch.randn(3, 4, 5),)
    )

    # 3D tensor, dim=2
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=2), (torch.randn(3, 4, 5),)
    )

    # 4D tensor, dim=1
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=1), (torch.randn(2, 3, 4, 5),)
    )

    # Negative dim (last dimension)
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=-1), (torch.randn(3, 4, 5),)
    )

    # Negative dim (second-to-last dimension)
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=-2), (torch.randn(3, 4, 5),)
    )


def test_median_with_indices(test_runner) -> None:
    # Test with different dimensions (values and indices)

    # 2D tensor, dim=0
    test_runner.lower_and_run_model(MedianModel(dim=0), (torch.randn(5, 10),))

    # 2D tensor, dim=1
    test_runner.lower_and_run_model(MedianModel(dim=1), (torch.randn(5, 10),))

    # 3D tensor, dim=0
    test_runner.lower_and_run_model(MedianModel(dim=0), (torch.randn(3, 4, 5),))

    # 3D tensor, dim=1
    test_runner.lower_and_run_model(MedianModel(dim=1), (torch.randn(3, 4, 5),))

    # 3D tensor, dim=2
    test_runner.lower_and_run_model(MedianModel(dim=2), (torch.randn(3, 4, 5),))

    # 4D tensor, dim=1
    test_runner.lower_and_run_model(MedianModel(dim=1), (torch.randn(2, 3, 4, 5),))

    # Negative dim (last dimension)
    test_runner.lower_and_run_model(MedianModel(dim=-1), (torch.randn(3, 4, 5),))

    # Negative dim (second-to-last dimension)
    test_runner.lower_and_run_model(MedianModel(dim=-2), (torch.randn(3, 4, 5),))


def test_median_keepdim(test_runner) -> None:
    # Test with keepdim=True (values only)

    # 2D tensor, dim=0, keepdim=True
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=0, keepdim=True), (torch.randn(5, 10),)
    )

    # 2D tensor, dim=1, keepdim=True
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=1, keepdim=True), (torch.randn(5, 10),)
    )

    # 3D tensor, dim=1, keepdim=True
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),)
    )

    # 4D tensor, dim=2, keepdim=True
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),)
    )


def test_median_keepdim_with_indices(test_runner) -> None:
    # Test with keepdim=True (values and indices)

    # 2D tensor, dim=0, keepdim=True
    test_runner.lower_and_run_model(
        MedianModel(dim=0, keepdim=True), (torch.randn(5, 10),)
    )

    # 2D tensor, dim=1, keepdim=True
    test_runner.lower_and_run_model(
        MedianModel(dim=1, keepdim=True), (torch.randn(5, 10),)
    )

    # 3D tensor, dim=1, keepdim=True
    test_runner.lower_and_run_model(
        MedianModel(dim=1, keepdim=True), (torch.randn(3, 4, 5),)
    )

    # 4D tensor, dim=2, keepdim=True
    test_runner.lower_and_run_model(
        MedianModel(dim=2, keepdim=True), (torch.randn(2, 3, 4, 5),)
    )


def test_median_shapes(test_runner) -> None:
    # Test with different tensor shapes (global reduction)

    # 1D tensor
    test_runner.lower_and_run_model(MedianValueOnlyModel(), (torch.randn(20),))

    # 2D tensor
    test_runner.lower_and_run_model(MedianValueOnlyModel(), (torch.randn(5, 10),))

    # 3D tensor
    test_runner.lower_and_run_model(MedianValueOnlyModel(), (torch.randn(3, 4, 5),))

    # 4D tensor
    test_runner.lower_and_run_model(MedianValueOnlyModel(), (torch.randn(2, 3, 4, 5),))

    # 5D tensor
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(), (torch.randn(2, 2, 3, 4, 5),)
    )


@unittest.skip("NaN and Inf are not enforced for backends.")
def test_median_edge_cases(test_runner) -> None:
    # Tensor with NaN (NaN should be propagated)
    x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(), (x,), generate_random_test_inputs=False
    )
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=0), (x,), generate_random_test_inputs=False
    )
    test_runner.lower_and_run_model(
        MedianValueOnlyModel(dim=1), (x,), generate_random_test_inputs=False
    )


def test_median_scalar(test_runner) -> None:
    # Test with scalar input (1-element tensor)
    test_runner.lower_and_run_model(MedianValueOnlyModel(), (torch.tensor([5.0]),))
    test_runner.lower_and_run_model(MedianValueOnlyModel(dim=0), (torch.tensor([5.0]),))
