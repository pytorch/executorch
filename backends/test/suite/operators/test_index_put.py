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


class IndexPutModel(torch.nn.Module):
    def __init__(self, accumulate=False):
        super().__init__()
        self.accumulate = accumulate

    def forward(self, x, indices, values):
        # Clone the input to avoid modifying it in-place
        result = x.clone()
        # Apply index_put_ and return the modified tensor
        result.index_put_(indices, values, self.accumulate)
        return result


@operator_test
class IndexPut(OperatorTest):
    @dtype_test
    def test_index_put_dtype(self, flow: TestFlow, dtype) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0]).to(dtype)
        self._test_op(
            IndexPutModel(),
            ((torch.rand(5, 2) * 100).to(dtype), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_basic(self, flow: TestFlow) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_accumulate(self, flow: TestFlow) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutModel(accumulate=False),
            (torch.ones(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutModel(accumulate=True),
            (torch.ones(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_shapes(self, flow: TestFlow) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]), torch.tensor([0, 1]))
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (
            torch.tensor([0, 2]),
            torch.tensor([1, 1]),
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        )
        values = torch.tensor(
            [
                10.0,
            ]
        )
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 3, 2, 4), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_indices(self, flow: TestFlow) -> None:
        indices = (torch.tensor([2]),)
        values = torch.tensor([10.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor([10.0, 20.0, 30.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([1, 1, 3, 3]),)
        values = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self._test_op(
            IndexPutModel(accumulate=True),
            (torch.randn(5), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_edge_cases(self, flow: TestFlow) -> None:
        indices = (torch.tensor([0, 1, 2, 3, 4]),)
        values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 5), indices, values),
            flow,
            generate_random_test_inputs=False,
        )
