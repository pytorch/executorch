# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


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


@parameterize_by_dtype
def test_index_put_in_place_dtype(test_runner, dtype) -> None:
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0]).to(dtype)
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        ((torch.rand(5, 2) * 100).to(dtype), indices, values),
        generate_random_test_inputs=False,
    )


@parameterize_by_dtype
def test_index_put_dtype(test_runner, dtype) -> None:
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0]).to(dtype)
    test_runner.lower_and_run_model(
        IndexPutModel(),
        ((torch.rand(5, 2) * 100).to(dtype), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_in_place_accumulate(test_runner) -> None:
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=False),
        (torch.ones(5, 2), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=True),
        (torch.ones(5, 2), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_accumulate(test_runner) -> None:
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=False),
        (torch.ones(5, 2), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=True),
        (torch.ones(5, 2), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_in_place_shapes(test_runner) -> None:
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5, 2), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2]), torch.tensor([1, 1]), torch.tensor([0, 1]))
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5, 3, 2), indices, values),
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
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5, 3, 2, 4), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_shapes(test_runner) -> None:
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5, 2), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2]), torch.tensor([1, 1]), torch.tensor([0, 1]))
    values = torch.tensor([10.0, 20.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5, 3, 2), indices, values),
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
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5, 3, 2, 4), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_in_place_indices(test_runner) -> None:
    indices = (torch.tensor([2]),)
    values = torch.tensor([10.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5, 2), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2, 4]),)
    values = torch.tensor([10.0, 20.0, 30.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5, 3), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([1, 1, 3, 3]),)
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=True),
        (torch.randn(5), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_indices(test_runner) -> None:
    indices = (torch.tensor([2]),)
    values = torch.tensor([10.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5, 2), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([0, 2, 4]),)
    values = torch.tensor([10.0, 20.0, 30.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5, 3), indices, values),
        generate_random_test_inputs=False,
    )

    indices = (torch.tensor([1, 1, 3, 3]),)
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=True),
        (torch.randn(5), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_in_place_broadcasting(test_runner) -> None:
    # Test scalar broadcasting - single value to multiple positions
    indices = (torch.tensor([0, 2, 4]),)
    values = torch.tensor([42.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(5, 3), indices, values),
        generate_random_test_inputs=False,
    )

    # Test 1D broadcasting to 2D indexed positions
    indices = (torch.tensor([0, 1]), torch.tensor([1, 2]))
    values = torch.tensor([10.0, 20.0])  # 1D tensor
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(3, 4), indices, values),
        generate_random_test_inputs=False,
    )

    # Test broadcasting with compatible shapes - 1D to multiple 2D slices
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([5.0, 15.0])  # Will broadcast to (2, 3) shape
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(4, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test 2D values broadcasting to 3D indexed positions
    indices = (torch.tensor([0, 1]),)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(3, 2, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test broadcasting with accumulate=True
    indices = (torch.tensor([1, 1, 1]),)
    values = torch.tensor([5.0])  # Scalar will be added 3 times to same position
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=True),
        (torch.ones(4, 2), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_broadcasting(test_runner) -> None:
    # Test scalar broadcasting - single value to multiple positions
    indices = (torch.tensor([0, 2, 4]),)
    values = torch.tensor([42.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(5, 3), indices, values),
        generate_random_test_inputs=False,
    )

    # Test 1D broadcasting to 2D indexed positions
    indices = (torch.tensor([0, 1]), torch.tensor([1, 2]))
    values = torch.tensor([10.0, 20.0])  # 1D tensor
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(3, 4), indices, values),
        generate_random_test_inputs=False,
    )

    # Test broadcasting with compatible shapes - 1D to multiple 2D slices
    indices = (torch.tensor([0, 2]),)
    values = torch.tensor([5.0, 15.0])  # Will broadcast to (2, 3) shape
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(4, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test 2D values broadcasting to 3D indexed positions
    indices = (torch.tensor([0, 1]),)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D tensor
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(3, 2, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test broadcasting with accumulate=True
    indices = (torch.tensor([1, 1, 1]),)
    values = torch.tensor([5.0])  # Scalar will be added 3 times to same position
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=True),
        (torch.ones(4, 2), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_in_place_two_indices(test_runner) -> None:
    # Test basic two-index tensor indexing
    indices = (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2]))
    values = torch.tensor([10.0, 20.0, 30.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(4, 3), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with different lengths (broadcasting)
    indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
    values = torch.tensor([15.0, 25.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(),
        (torch.randn(3, 3), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with repeated positions and accumulate=True
    indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
    values = torch.tensor([5.0, 10.0, 15.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=True),
        (torch.zeros(3, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with repeated positions and accumulate=False
    indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
    values = torch.tensor([5.0, 10.0, 15.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=False),
        (torch.zeros(3, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with index broadcast.
    indices = (torch.tensor([1]), torch.tensor([0, 0, 1]))
    values = torch.tensor([5.0, 10.0, 15.0])
    test_runner.lower_and_run_model(
        IndexPutInPlaceModel(accumulate=False),
        (torch.zeros(3, 2), indices, values),
        generate_random_test_inputs=False,
    )


def test_index_put_two_indices(test_runner) -> None:
    # Test basic two-index tensor indexing
    indices = (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2]))
    values = torch.tensor([10.0, 20.0, 30.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(4, 3), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with different lengths (broadcasting)
    indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
    values = torch.tensor([15.0, 25.0])
    test_runner.lower_and_run_model(
        IndexPutModel(),
        (torch.randn(3, 3), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with repeated positions and accumulate=True
    indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
    values = torch.tensor([5.0, 10.0, 15.0])
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=True),
        (torch.zeros(3, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with repeated positions and accumulate=False
    indices = (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1]))
    values = torch.tensor([5.0, 10.0, 15.0])
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=False),
        (torch.zeros(3, 2), indices, values),
        generate_random_test_inputs=False,
    )

    # Test two-index with index broadcast.
    indices = (torch.tensor([1]), torch.tensor([0, 0, 1]))
    values = torch.tensor([5.0, 10.0, 15.0])
    test_runner.lower_and_run_model(
        IndexPutModel(accumulate=False),
        (torch.zeros(3, 2), indices, values),
        generate_random_test_inputs=False,
    )
