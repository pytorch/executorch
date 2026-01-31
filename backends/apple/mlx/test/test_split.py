# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.apple.mlx.test.test_utils import OpTestCase, register_test
from torch import nn


class SplitModel(nn.Module):
    """Model that splits a tensor into chunks with specified sizes."""

    def __init__(self, sizes, dim=0):
        super().__init__()
        self.sizes = sizes
        self.dim = dim

    def forward(self, x):
        # split_with_sizes_copy returns a tuple of tensors
        chunks = torch.ops.aten.split_with_sizes_copy.default(x, self.sizes, self.dim)
        # Return just the first chunk for testing
        # (getitem will extract individual elements in the graph)
        return chunks[0]


class SplitMultiOutputModel(nn.Module):
    """Model that splits and uses multiple outputs."""

    def __init__(self, sizes, dim=0):
        super().__init__()
        self.sizes = sizes
        self.dim = dim

    def forward(self, x):
        chunks = torch.ops.aten.split_with_sizes_copy.default(x, self.sizes, self.dim)
        # Use first and last chunks
        return chunks[0] + chunks[-1]


@register_test
class SplitTest(OpTestCase):
    name = "split"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(9, 4), sizes=None, dim=0):
        self.shape = shape
        self.sizes = sizes if sizes is not None else [2, 3, 4]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitModel(sizes=self.sizes, dim=self.dim)


@register_test
class SplitDim1Test(OpTestCase):
    name = "split_dim1"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(3, 10), sizes=None, dim=1):
        self.shape = shape
        self.sizes = sizes if sizes is not None else [2, 3, 5]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitModel(sizes=self.sizes, dim=self.dim)


@register_test
class Split3DTest(OpTestCase):
    name = "split_3d"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(2, 12, 4), sizes=None, dim=1):
        self.shape = shape
        self.sizes = sizes if sizes is not None else [3, 4, 5]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitModel(sizes=self.sizes, dim=self.dim)


@register_test
class SplitTwoChunksTest(OpTestCase):
    name = "split_two"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(8, 4), sizes=None, dim=0):
        self.shape = shape
        self.sizes = sizes if sizes is not None else [3, 5]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitModel(sizes=self.sizes, dim=self.dim)


@register_test
class SplitMultiOutputTest(OpTestCase):
    name = "split_multi"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(10, 3), sizes=None, dim=0):
        self.shape = shape
        self.sizes = sizes if sizes is not None else [5, 5]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitMultiOutputModel(sizes=self.sizes, dim=self.dim)


class SplitUniformModel(nn.Module):
    """Model that splits a tensor into chunks of uniform size using torch.split."""

    def __init__(self, split_size, dim=0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        # torch.split with a single int splits into chunks of that size
        # Last chunk may be smaller if it doesn't divide evenly
        chunks = torch.split(x, self.split_size, dim=self.dim)
        return chunks[0]


class SplitUniformMultiOutputModel(nn.Module):
    """Model that splits uniformly and uses multiple outputs."""

    def __init__(self, split_size, dim=0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        chunks = torch.split(x, self.split_size, dim=self.dim)
        # Concatenate first and last chunks along the split dimension
        return torch.cat([chunks[0], chunks[-1]], dim=self.dim)


@register_test
class SplitUniformTest(OpTestCase):
    """Test splitting 10 elements with split_size=3 -> sizes [3, 3, 3, 1]."""

    name = "split_uniform"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(10, 4), split_size=3, dim=0):
        self.shape = shape
        self.split_size = split_size
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitUniformModel(split_size=self.split_size, dim=self.dim)


@register_test
class SplitUniformDim1Test(OpTestCase):
    """Test splitting 7 elements with split_size=4 along dim 1 -> sizes [4, 3]."""

    name = "split_uniform_dim1"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(3, 7), split_size=4, dim=1):
        self.shape = shape
        self.split_size = split_size
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitUniformModel(split_size=self.split_size, dim=self.dim)


@register_test
class SplitUniformMultiOutputTest(OpTestCase):
    """Test uniform split with multiple outputs used."""

    name = "split_uniform_multi"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shape=(11, 5), split_size=3, dim=0):
        self.shape = shape
        self.split_size = split_size
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SplitUniformMultiOutputModel(split_size=self.split_size, dim=self.dim)
