#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Consolidated op tests for the MLX delegate.

This file contains all op tests organized by category. Each test class inherits
from OpTestCase and can be run via the run_all_tests.py script.

Usage:
    # Run all tests (with 4 parallel workers, cleanup after)
    python -m executorch.backends.mlx.test.run_all_tests -j4 --clean-after

    # Run specific test
    python -m executorch.backends.mlx.test.run_all_tests add

    # List available tests
    python -m executorch.backends.mlx.test.run_all_tests --list

See README.md in this directory for full documentation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

# Import custom ops for RoPE and KV cache tests
from executorch.backends.mlx import (  # noqa: F401 - registers mlx ops  # noqa: F401 - registers mlx.rope
    custom_ops,
    ops,
)

from .test_utils import OpTestCase, register_test


class BmmModel(nn.Module):
    """Model that performs batch matrix multiplication."""

    def __init__(self, batch_size: int, n: int, m: int, p: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(batch_size, m, p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weight)


@register_test
class BmmTest(OpTestCase):
    """Test case for bmm (batch matrix multiplication)."""

    name = "bmm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 4,
        n: int = 8,
        m: int = 16,
        p: int = 32,
    ):
        self.batch_size = batch_size
        self.n = n
        self.m = m
        self.p = p
        self.name = f"bmm_{batch_size}x{n}x{m}x{p}"

    @classmethod
    def get_test_configs(cls) -> List["BmmTest"]:
        return [
            cls(batch_size=4, n=8, m=16, p=32),
            cls(batch_size=2, n=64, m=64, p=32),
        ]

    def create_model(self) -> nn.Module:
        return BmmModel(self.batch_size, self.n, self.m, self.p)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.n, self.m)
        return (x,)


class AddmmModel(nn.Module):
    """Model that performs addmm: bias + (mat1 @ mat2)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(
                self.bias, x, self.weight.t(), beta=self.beta, alpha=self.alpha
            )
        else:
            return torch.mm(x, self.weight.t())


@register_test
class AddmmTest(OpTestCase):
    """Test case for addmm."""

    name = "addmm"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 2,
        in_features: int = 64,
        out_features: int = 32,
        bias: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

        # Build unique test name
        if not bias:
            name = f"addmm_{in_features}x{out_features}_no_bias"
        elif alpha != 1.0 or beta != 1.0:
            name = f"addmm_{in_features}x{out_features}_a{alpha}_b{beta}"
        else:
            name = f"addmm_{in_features}x{out_features}"
        self.name = name

    @classmethod
    def get_test_configs(cls) -> List["AddmmTest"]:
        return [
            cls(
                batch_size=2, in_features=64, out_features=32
            ),  # with bias, default alpha/beta
            cls(
                batch_size=2, in_features=64, out_features=32, bias=False
            ),  # without bias
            cls(batch_size=4, in_features=128, out_features=64),  # larger size
            cls(
                batch_size=2, in_features=64, out_features=32, alpha=2.0, beta=0.5
            ),  # custom alpha/beta
        ]

    def create_model(self) -> nn.Module:
        return AddmmModel(
            self.in_features,
            self.out_features,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(self.batch_size, self.in_features)
        return (x,)
