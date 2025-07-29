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


class TransposeModel(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


@operator_test
class Transpose(OperatorTest):
    @dtype_test
    def test_transpose_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            TransposeModel(dim0=0, dim1=1),
            (torch.rand(20, 32).to(dtype),),
            flow,
        )

    def test_transpose_basic(self, flow: TestFlow) -> None:
        self._test_op(
            TransposeModel(dim0=0, dim1=1),
            (torch.randn(20, 32),),
            flow,
        )

    def test_transpose_3d(self, flow: TestFlow) -> None:
        self._test_op(
            TransposeModel(dim0=0, dim1=1),
            (torch.randn(8, 10, 12),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=0, dim1=2),
            (torch.randn(8, 10, 12),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=1, dim1=2),
            (torch.randn(8, 10, 12),),
            flow,
        )

    def test_transpose_4d(self, flow: TestFlow) -> None:
        self._test_op(
            TransposeModel(dim0=0, dim1=3),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=1, dim1=2),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

    def test_transpose_identity(self, flow: TestFlow) -> None:
        self._test_op(
            TransposeModel(dim0=0, dim1=0),
            (torch.randn(20, 32),),
            flow,
        )
        self._test_op(
            TransposeModel(dim0=1, dim1=1),
            (torch.randn(20, 32),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=0, dim1=0),
            (torch.randn(8, 10, 12),),
            flow,
        )
        self._test_op(
            TransposeModel(dim0=1, dim1=1),
            (torch.randn(8, 10, 12),),
            flow,
        )
        self._test_op(
            TransposeModel(dim0=2, dim1=2),
            (torch.randn(8, 10, 12),),
            flow,
        )

    def test_transpose_negative_dims(self, flow: TestFlow) -> None:
        self._test_op(
            TransposeModel(dim0=-3, dim1=-1),
            (torch.randn(8, 10, 12),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=-2, dim1=-1),
            (torch.randn(8, 10, 12),),
            flow,
        )

    def test_transpose_different_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            TransposeModel(dim0=0, dim1=1),
            (torch.randn(20, 32),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=0, dim1=2),
            (torch.randn(8, 10, 12),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=1, dim1=3),
            (torch.randn(4, 6, 8, 10),),
            flow,
        )

        self._test_op(
            TransposeModel(dim0=0, dim1=4),
            (torch.randn(2, 3, 4, 5, 6),),
            flow,
        )
