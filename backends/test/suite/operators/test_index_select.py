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


class IndexSelectModel(torch.nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x, indices):
        return torch.index_select(x, self.dim, indices)


@operator_test
class IndexSelect(OperatorTest):
    @dtype_test
    def test_index_select_dtype(self, flow: TestFlow, dtype) -> None:
        indices = torch.tensor([0, 2], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=0),
            ((torch.rand(5, 3) * 100).to(dtype), indices),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_select_dimensions(self, flow: TestFlow) -> None:
        indices = torch.tensor([0, 2], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([0, 1], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=1),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([0, 2], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=2),
            (torch.randn(3, 4, 5), indices),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_select_shapes(self, flow: TestFlow) -> None:
        indices = torch.tensor([0, 1], dtype=torch.int64)

        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5), indices),
            flow,
            generate_random_test_inputs=False,
        )

        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )

        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3, 2), indices),
            flow,
            generate_random_test_inputs=False,
        )

        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3, 2, 4), indices),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_select_indices(self, flow: TestFlow) -> None:
        indices = torch.tensor([2], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([0, 2, 4], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([1, 1, 3, 3], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64)
        self._test_op(
            IndexSelectModel(dim=0),
            (torch.randn(5, 3), indices),
            flow,
            generate_random_test_inputs=False,
        )
