#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test for addmm op using the MLX delegate.

addmm(bias, mat1, mat2) computes: bias + (mat1 @ mat2)

This is the decomposed form of nn.Linear in Edge IR:
  linear(x, weight, bias) -> permute(weight) + addmm(bias, x, permuted_weight)

Usage:
    # Run via run_all_tests (recommended):
    python -m executorch.backends.apple.mlx.test.run_all_tests addmm

    # Run directly:
    python -m executorch.backends.apple.mlx.test.test_addmm run
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test, run_op_test_main


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
        # This will decompose to permute + addmm in Edge IR
        # addmm(bias, x, weight.T, beta, alpha) = beta * bias + alpha * (x @ weight.T)
        if self.bias is not None:
            return torch.addmm(
                self.bias, x, self.weight.t(), beta=self.beta, alpha=self.alpha
            )
        else:
            # Without bias, this becomes just matmul
            return torch.mm(x, self.weight.t())


@register_test
class AddmmTest(OpTestCase):
    """Test case for addmm.

    Note: addmm is the decomposed form of linear in Edge IR.
    The handler transposes mat2 and uses LinearNode internally.
    """

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
        """Return all test configurations to run."""
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

    def get_dynamic_shapes(self):
        return None


# Factory for CLI usage
def _create_from_args(args) -> AddmmTest:
    if args is None:
        return AddmmTest()
    return AddmmTest(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features,
    )


def _add_args(parser):
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--in-features", type=int, default=64, help="Input features")
    parser.add_argument("--out-features", type=int, default=32, help="Output features")


if __name__ == "__main__":
    run_op_test_main(_create_from_args, "Test addmm on MLX delegate", _add_args)
