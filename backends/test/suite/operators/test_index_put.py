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


class IndexPutInPlaceModel(torch.nn.Module):
    def __init__(self, accumulate=False):
        super().__init__()
        self.accumulate = accumulate

    def forward(self, x, indices, values):
        # Clone the input to avoid modifying it in-place
        result = x.clone()
        # Apply index_put_ and return the modified tensor
        result.index_put_(indices, values, self.accumulate)
        return result


class IndexPutModel(torch.nn.Module):
    def __init__(self, accumulate=False):
        super().__init__()
        self.accumulate = accumulate

    def forward(self, x, indices, values):
        # Use the non-in-place variant which returns a new tensor
        return torch.index_put(x, indices, values, self.accumulate)


@operator_test
class IndexPut(OperatorTest):
    @dtype_test
    def test_index_put_in_place_dtype(self, flow: TestFlow, dtype) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0]).to(dtype)
        self._test_op(
            IndexPutInPlaceModel(),
            ((torch.rand(5, 2) * 100).to(dtype), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

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

    def test_index_put_in_place_accumulate(self, flow: TestFlow) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutInPlaceModel(accumulate=False),
            (torch.ones(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutInPlaceModel(accumulate=True),
            (torch.ones(5, 2), indices, values),
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

    def test_index_put_in_place_shapes(self, flow: TestFlow) -> None:
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(5), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]), torch.tensor([0, 1]))
        values = torch.tensor([10.0, 20.0])
        self._test_op(
            IndexPutInPlaceModel(),
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
            IndexPutInPlaceModel(),
            (torch.randn(5, 3, 2, 4), indices, values),
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

    def test_index_put_in_place_indices(self, flow: TestFlow) -> None:
        indices = (torch.tensor([2]),)
        values = torch.tensor([10.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(5, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor([10.0, 20.0, 30.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(5, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        indices = (torch.tensor([1, 1, 3, 3]),)
        values = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self._test_op(
            IndexPutInPlaceModel(accumulate=True),
            (torch.randn(5), indices, values),
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

    def test_index_put_in_place_broadcasting(self, flow: TestFlow) -> None:
        # Test scalar broadcasting - single value to multiple positions
        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor([42.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(5, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test 1D broadcasting to 2D indexed positions
        indices = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        values = torch.tensor([10.0, 20.0])  # 1D tensor
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(3, 4), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test broadcasting with compatible shapes - 1D to multiple 2D slices
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([5.0, 15.0])  # Will broadcast to (2, 3) shape
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(4, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test 2D values broadcasting to 3D indexed positions
        indices = (torch.tensor([0, 1]),)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(3, 2, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test broadcasting with accumulate=True
        indices = (torch.tensor([1, 1, 1]),)
        values = torch.tensor([5.0])  # Scalar will be added 3 times to same position
        self._test_op(
            IndexPutInPlaceModel(accumulate=True),
            (torch.ones(4, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_broadcasting(self, flow: TestFlow) -> None:
        # Test scalar broadcasting - single value to multiple positions
        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor([42.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(5, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test 1D broadcasting to 2D indexed positions
        indices = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        values = torch.tensor([10.0, 20.0])  # 1D tensor
        self._test_op(
            IndexPutModel(),
            (torch.randn(3, 4), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test broadcasting with compatible shapes - 1D to multiple 2D slices
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([5.0, 15.0])  # Will broadcast to (2, 3) shape
        self._test_op(
            IndexPutModel(),
            (torch.randn(4, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test 2D values broadcasting to 3D indexed positions
        indices = (torch.tensor([0, 1]),)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor
        self._test_op(
            IndexPutModel(),
            (torch.randn(3, 2, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test broadcasting with accumulate=True
        indices = (torch.tensor([1, 1, 1]),)
        values = torch.tensor([5.0])  # Scalar will be added 3 times to same position
        self._test_op(
            IndexPutModel(accumulate=True),
            (torch.ones(4, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_in_place_two_indices(self, flow: TestFlow) -> None:
        # Test basic two-index tensor indexing
        indices = (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2]))
        values = torch.tensor([10.0, 20.0, 30.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(4, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with different lengths (broadcasting)
        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
        values = torch.tensor([15.0, 25.0])
        self._test_op(
            IndexPutInPlaceModel(),
            (torch.randn(3, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with repeated positions and accumulate=True
        indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
        values = torch.tensor([5.0, 10.0, 15.0])
        self._test_op(
            IndexPutInPlaceModel(accumulate=True),
            (torch.zeros(3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with repeated positions and accumulate=False
        indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
        values = torch.tensor([5.0, 10.0, 15.0])
        self._test_op(
            IndexPutInPlaceModel(accumulate=False),
            (torch.zeros(3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with index broadcast.
        indices = (torch.tensor([1]), torch.tensor([0, 0, 1]))
        values = torch.tensor([5.0, 10.0, 15.0])
        self._test_op(
            IndexPutInPlaceModel(accumulate=False),
            (torch.zeros(3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

    def test_index_put_two_indices(self, flow: TestFlow) -> None:
        # Test basic two-index tensor indexing
        indices = (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2]))
        values = torch.tensor([10.0, 20.0, 30.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(4, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with different lengths (broadcasting)
        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
        values = torch.tensor([15.0, 25.0])
        self._test_op(
            IndexPutModel(),
            (torch.randn(3, 3), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with repeated positions and accumulate=True
        indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
        values = torch.tensor([5.0, 10.0, 15.0])
        self._test_op(
            IndexPutModel(accumulate=True),
            (torch.zeros(3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with repeated positions and accumulate=False
        indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
        values = torch.tensor([5.0, 10.0, 15.0])
        self._test_op(
            IndexPutModel(accumulate=False),
            (torch.zeros(3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )

        # Test two-index with index broadcast.
        indices = (torch.tensor([1]), torch.tensor([0, 0, 1]))
        values = torch.tensor([5.0, 10.0, 15.0])
        self._test_op(
            IndexPutModel(accumulate=False),
            (torch.zeros(3, 2), indices, values),
            flow,
            generate_random_test_inputs=False,
        )
