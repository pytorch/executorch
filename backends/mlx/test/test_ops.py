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
