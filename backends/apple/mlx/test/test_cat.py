# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.apple.mlx.test.test_utils import OpTestCase, register_test
from torch import nn


class CatModel(nn.Module):
    """Model that concatenates multiple tensors along a dimension."""

    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2, x3):
        return torch.cat([x1, x2, x3], dim=self.dim)


@register_test
class CatTest(OpTestCase):
    name = "cat"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shapes=None, dim=0):  # Concatenate along dim 0
        self.shapes = shapes if shapes is not None else [(2, 3), (4, 3), (1, 3)]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        tensors = [torch.randn(shape) for shape in self.shapes]
        return tuple(tensors)

    def create_model(self) -> nn.Module:
        return CatModel(dim=self.dim)


@register_test
class CatAlongDim1Test(OpTestCase):
    name = "cat_dim1"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shapes=None, dim=1):
        self.shapes = shapes if shapes is not None else [(3, 2), (3, 4), (3, 1)]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        tensors = [torch.randn(shape) for shape in self.shapes]
        return tuple(tensors)

    def create_model(self) -> nn.Module:
        return CatModel(dim=self.dim)


@register_test
class Cat3DTest(OpTestCase):
    name = "cat_3d"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shapes=None, dim=0):
        self.shapes = (
            shapes if shapes is not None else [(2, 3, 4), (5, 3, 4), (3, 3, 4)]
        )
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        tensors = [torch.randn(shape) for shape in self.shapes]
        return tuple(tensors)

    def create_model(self) -> nn.Module:
        return CatModel(dim=self.dim)


@register_test
class CatTwoTensorsTest(OpTestCase):
    name = "cat_two"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shapes=None, dim=0):
        self.shapes = shapes if shapes is not None else [(3, 4), (2, 4)]
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        tensors = [torch.randn(shape) for shape in self.shapes]
        return tuple(tensors)

    def create_model(self) -> nn.Module:
        # Need a model that only takes 2 inputs
        class Cat2Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x1, x2):
                return torch.cat([x1, x2], dim=self.dim)

        return Cat2Model(dim=self.dim)


@register_test
class CatNegativeDimTest(OpTestCase):
    name = "cat_neg_dim"
    rtol = 1e-5
    atol = 1e-5

    def __init__(self, shapes=None, dim=-2):
        self.shapes = (
            shapes if shapes is not None else [(3, 2, 4), (3, 5, 4), (3, 1, 4)]
        )
        self.dim = dim

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        tensors = [torch.randn(shape) for shape in self.shapes]
        return tuple(tensors)

    def create_model(self) -> nn.Module:
        return CatModel(dim=self.dim)
