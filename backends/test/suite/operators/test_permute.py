# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class PermuteModel(torch.nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


@operator_test
class Permute(OperatorTest):
    @dtype_test
    def test_permute_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            PermuteModel(dims=[1, 0]),
            (torch.rand(20, 32).to(dtype),),
            flow,
        )

    def test_permute_3d(self, flow: TestFlow) -> None:
        self._test_op(
            PermuteModel(dims=[2, 0, 1]),
            (torch.randn(8, 10, 12),),
            flow,
        )

        self._test_op(
            PermuteModel(dims=[1, 2, 0]),
            (torch.randn(8, 10, 12),),
            flow,
        )

        self._test_op(
            PermuteModel(dims=[0, 2, 1]),
            (torch.randn(8, 10, 12),),
            flow,
        )

    def test_permute_4d(self, flow: TestFlow) -> None:
        self._test_op(
            PermuteModel(dims=[3, 2, 1, 0]),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

        self._test_op(
            PermuteModel(dims=[0, 2, 1, 3]),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

    def test_permute_identity(self, flow: TestFlow) -> None:
        self._test_op(
            PermuteModel(dims=[0, 1]),
            (torch.randn(20, 32),),
            flow,
        )

        self._test_op(
            PermuteModel(dims=[0, 1, 2]),
            (torch.randn(8, 10, 12),),
            flow,
        )

    def test_permute_negative_dims(self, flow: TestFlow) -> None:
        self._test_op(
            PermuteModel(dims=[-1, -3, -2, -4]),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

        self._test_op(
            PermuteModel(dims=[-4, -2, -3, -1]),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

    def test_permute_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            PermuteModel(dims=[0]),
            (torch.randn(512),),
            flow,
        )

        self._test_op(
            PermuteModel(dims=[4, 3, 2, 1, 0]),
            (torch.randn(2, 3, 4, 5, 6),),
            flow,
        )
