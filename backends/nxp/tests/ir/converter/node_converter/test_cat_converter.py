# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import (
    ModelInputSpec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    Cat,
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


def _normalized_dim(dim, rank):
    return dim if dim >= 0 else dim + rank


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class CatModule(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs: torch.Tensor):
        return torch.cat(list(inputs), self.dim)


class CatMaxPoolModule(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_pool_2d = torch.nn.MaxPool2d(kernel_size=1)

    def forward(self, *inputs: torch.Tensor):
        x = torch.cat(list(inputs), self.dim)
        x = self.max_pool_2d(x)
        return x


class TestCat:

    def test__qat(self, mocker, use_qat):
        input_shape = (2, 3, 5)
        num_inputs = 2

        input_shapes = [ModelInputSpec(input_shape)] * num_inputs
        model = CatModule(1)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Cat: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_shapes, graph_verifier, use_qat=use_qat)

    @pytest.mark.parametrize("dim", list(range(-3, 3)), ids=lambda dim: f"dim={dim}")
    @pytest.mark.parametrize("num_inputs", [2, 5], ids=lambda n: f"n={n}")
    def test__same_shapes(self, mocker, dim, num_inputs):
        input_shape = (2, 3, 5)
        input_shapes = [ModelInputSpec(input_shape)] * num_inputs

        model = CatModule(dim)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Cat: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_shapes, graph_verifier)

    @pytest.mark.parametrize("dim", [0, -3, 2, -1], ids=lambda dim: f"dim={dim}")
    @pytest.mark.parametrize("num_inputs", [2, 5], ids=lambda n: f"n={n}")
    def test__same_shapes__channels_first(self, mocker, dim, num_inputs):
        input_shape = (2, 3, 4, 5)
        input_shapes = [ModelInputSpec(input_shape)] * num_inputs

        model = CatMaxPoolModule(dim)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Cat: 1, MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(model, input_shapes, graph_verifier)

    @pytest.mark.parametrize("dim", [0, -1], ids=lambda dim: f"dim={dim}")
    @pytest.mark.parametrize("rank", [2, 3, 4], ids=lambda rank: f"rank={rank}")
    @pytest.mark.parametrize("num_inputs", [2, 3], ids=lambda n: f"n={n}")
    def test__different_shapes(self, mocker, dim, rank, num_inputs):
        # The input shapes can only differ in the `dim` dimension. So we can just assign a different one for each input.
        # e.g. [(2, 3, 4), (3, 3, 4), (4, 3, 4), (5, 3, 4), (6, 3, 4)]
        base_shape = [i + 2 for i in range(rank)]
        input_shapes = [list(base_shape) for _ in range(num_inputs)]
        for i, input_shape in enumerate(input_shapes):
            input_shape[dim] = i + 2
        input_shapes = list(map(tuple, input_shapes))

        model = CatModule(dim)
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Cat: 1}, expected_non_delegated_ops={}
        )

        lower_run_compare(model, input_shapes, graph_verifier)

    @pytest.mark.parametrize("dim", [1, -1], ids=lambda dim: f"dim={dim}")
    @pytest.mark.parametrize("num_inputs", [2, 5], ids=lambda n: f"n={n}")
    def test__different_shapes__channels_first(self, mocker, dim, num_inputs):
        # The input shapes can only differ in the `dim` dimension. So we can just assign a different one for each input.
        # e.g. [(1, 3, 4, 5), (2, 3, 4, 5)]
        base_shape = (2, 3, 4, 5)
        input_shapes = [list(base_shape) for _ in range(num_inputs)]
        for i, input_shape in enumerate(input_shapes):
            input_shape[dim] = i + 2
        input_shapes = list(map(tuple, input_shapes))

        model = CatMaxPoolModule(dim)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Cat: 1, MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(model, input_shapes, graph_verifier)

    def test__single_input__alone_in_partition__not_delegated(self):
        # The operator is a noop, and there is no other op in the model. The Neutron Converter would produce an empty
        #  graph, so the `cat` is not delegated.
        input_shape = [ModelInputSpec((2, 3, 5))]
        model = CatModule(1)

        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        # Make sure the `cat` was NOT delegated.
        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [Cat])

    def test__single_input__not_alone_in_partition__delegated(self, mocker):
        # The operator is a noop, but there is another op in the model, so they are both delegated.
        input_shape = [ModelInputSpec((2, 3, 4, 5))]

        model = CatMaxPoolModule(1)
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Cat: 1, MaxPool2DWithIndices: 1, GetItem: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(model, input_shape, graph_verifier)
