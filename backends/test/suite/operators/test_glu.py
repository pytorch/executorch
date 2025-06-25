# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Callable

import torch

from executorch.backends.test.suite import dtype_test, operator_test, OperatorTest


class Model(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.glu(x, dim=self.dim)


@operator_test
class TestGLU(OperatorTest):
    @dtype_test
    def test_glu_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input must have even number of elements in the specified dimension
        self._test_op(
            Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),), tester_factory
        )

    def test_glu_f32_dim_last(self, tester_factory: Callable) -> None:
        # Default dim is -1 (last dimension)
        self._test_op(Model(), (torch.randn(3, 4, 6),), tester_factory)

    def test_glu_f32_dim_first(self, tester_factory: Callable) -> None:
        # Test with dim=0 (first dimension)
        self._test_op(Model(dim=0), (torch.randn(4, 3, 5),), tester_factory)

    def test_glu_f32_dim_middle(self, tester_factory: Callable) -> None:
        # Test with dim=1 (middle dimension)
        self._test_op(Model(dim=1), (torch.randn(3, 8, 5),), tester_factory)

    def test_glu_f32_boundary_values(self, tester_factory: Callable) -> None:
        # Test with specific values spanning negative and positive ranges
        # Input must have even number of elements in the specified dimension
        x = torch.tensor([[-10.0, -5.0, -1.0, 0.0], [1.0, 5.0, 10.0, -2.0]])
        self._test_op(Model(dim=1), (x,), tester_factory)
