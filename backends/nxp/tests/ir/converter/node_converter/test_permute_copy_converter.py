# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

# noinspection PyUnusedImports
import pytest
import torch
from _pytest.mark import ParameterSet

from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import (
    ExecutorchDelegateCall,
    GetItem,
    MaxPool2DWithIndices,
    PermuteCopy,
)
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


class PermuteModule(torch.nn.Module):
    def __init__(self, perm: tuple[int, ...]):
        super().__init__()
        self.perm = perm

    def forward(self, x):
        return torch.permute(x, self.perm)


class MaxPoolPermuteModule(torch.nn.Module):
    def __init__(self, perm: tuple[int, ...]):
        super().__init__()
        self.perm = perm
        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=1
        )  # No-op, but it enforces the channels first format.

    def forward(self, x):
        x = self.max_pool2d(x)
        return torch.permute(x, self.perm)


class PermuteMaxPoolModule(torch.nn.Module):
    def __init__(self, perm: tuple[int, ...]):
        super().__init__()
        self.perm = perm
        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=1
        )  # No-op, but it enforces the channels first format.

    def forward(self, x):
        x = torch.permute(x, self.perm)
        return self.max_pool2d(x)


class PermuteMaxPoolPermuteModule(torch.nn.Module):
    def __init__(self, perm1: tuple[int, ...], perm2: tuple[int, ...]):
        super().__init__()
        self.perm1 = perm1
        self.perm2 = perm2
        self.max_pool2d = torch.nn.MaxPool2d(
            kernel_size=1
        )  # No-op, but it enforces the channels first format.

    def forward(self, x):
        x = torch.permute(x, self.perm1)
        x = self.max_pool2d(x)
        x = torch.permute(x, self.perm2)
        return x


class TestPermuteCopy:
    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self, model, input_shape, mocker, expected_delegated_ops=None, use_qat=False
    ):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops or {PermuteCopy: 1},
            expected_non_delegated_ops={},
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            use_qat=use_qat,
        )

    # noinspection PyMethodMayBeStatic
    def assert_not_delegated(self, model, input_shape):
        delegated_ep = to_quantized_edge_program(model, input_shape).exported_program()

        assert not graph_contains_any_of_ops(
            delegated_ep.graph, [ExecutorchDelegateCall]
        )
        assert graph_contains_any_of_ops(delegated_ep.graph, [PermuteCopy])

    @staticmethod
    def _all_permutations_for_rank(rank: int) -> list[tuple[int, ...]]:
        return [tuple(perm) for perm in itertools.permutations(range(rank))]

    @staticmethod
    def _special_4d_permutations() -> list[ParameterSet]:
        # noinspection PyTypeChecker
        return [
            pytest.param((0, 1, 2, 3), id="identity"),
            pytest.param((0, 2, 3, 1), id="to channels last"),
            pytest.param((0, 3, 1, 2), id="to channels first"),
            pytest.param((3, 2, 1, 0), id="reverse"),
        ]

    def test__qat(self, mocker, use_qat):
        input_shape = (2, 3, 5, 7)
        permutation = (0, 2, 3, 1)  # NCHW -> NHWC
        model = PermuteModule(permutation)
        self.assert_delegated(model, input_shape, mocker, use_qat=use_qat)

    @pytest.mark.parametrize(
        "permutation",
        _all_permutations_for_rank(3),
        ids=lambda perm: f"permutation = {perm}",
    )
    def test__all_permutations__3d(self, mocker, permutation: tuple[int]):
        # Avoid dimensions of size 1 and multiples of `num_macs` for a thorough test.
        input_shape = (2, 3, 5)
        model = PermuteModule(permutation)
        if permutation == (0, 1, 2):
            # Identity permutation is a no-op on Neutron. As it's the only node in the testing model, it's delegation
            #  would result in an empty graph, which is not allowed. Therefore, it's not delegated.
            self.assert_not_delegated(model, input_shape)
        else:
            self.assert_delegated(model, input_shape, mocker)

    @pytest.mark.parametrize(
        "permutation",
        _all_permutations_for_rank(4),
        ids=lambda perm: f"permutation = {perm}",
    )
    def test__all_permutations__4d(self, mocker, permutation: tuple[int]):
        # Avoid dimensions of size 1 and multiples of `num_macs` for a thorough test.
        input_shape = (2, 3, 5, 7)
        model = PermuteModule(permutation)
        if permutation == (0, 1, 2, 3):
            # Identity permutation is a no-op on Neutron. As it's the only node in the testing model, it's delegation
            #  would result in an empty graph, which is not allowed. Therefore, it's not delegated.
            self.assert_not_delegated(model, input_shape)
        else:
            self.assert_delegated(model, input_shape, mocker)

    @pytest.mark.parametrize("permutation", _special_4d_permutations())
    def test__all_permutations__4d__channels_first_input(
        self, mocker, permutation: tuple[int]
    ):
        # Avoid dimensions of size 1 and multiples of `num_macs` for a thorough test.
        input_shape = (2, 3, 5, 7)
        model = MaxPoolPermuteModule(permutation)
        expected_delegated_ops = {MaxPool2DWithIndices: 1, GetItem: 1, PermuteCopy: 1}
        self.assert_delegated(
            model, input_shape, mocker, expected_delegated_ops=expected_delegated_ops
        )

    @pytest.mark.parametrize("permutation", _special_4d_permutations())
    def test__all_permutations__4d__channels_first_output(
        self, mocker, permutation: tuple[int]
    ):
        # Avoid dimensions of size 1 and multiples of `num_macs` for a thorough test.
        input_shape = (2, 3, 5, 7)
        model = PermuteMaxPoolModule(permutation)
        expected_delegated_ops = {MaxPool2DWithIndices: 1, GetItem: 1, PermuteCopy: 1}
        self.assert_delegated(
            model, input_shape, mocker, expected_delegated_ops=expected_delegated_ops
        )

    @pytest.mark.parametrize("perm1", _special_4d_permutations())
    @pytest.mark.parametrize("perm2", _special_4d_permutations())
    def test__all_permutations__4d__channels_first_io(
        self, mocker, perm1: tuple[int], perm2: tuple[int]
    ):
        # Avoid dimensions of size 1 and multiples of `num_macs` for a thorough test.
        input_shape = (2, 3, 5, 7)
        model = PermuteMaxPoolPermuteModule(perm1, perm2)
        expected_delegated_ops = {MaxPool2DWithIndices: 1, GetItem: 1, PermuteCopy: 2}
        self.assert_delegated(
            model, input_shape, mocker, expected_delegated_ops=expected_delegated_ops
        )

    @pytest.mark.parametrize(
        "permutation",
        [
            pytest.param((0, 1, 2, 3, 4), id="identity"),
            pytest.param((0, 2, 3, 4, 1), id="to channels last"),
            pytest.param((0, 4, 1, 2, 3), id="to channels first"),
            pytest.param((4, 3, 2, 1, 0), id="reverse"),
            pytest.param((4, 2, 3, 0, 1), id="perm = (4, 2, 3, 0, 1)"),
        ],
    )
    def test__5d(self, mocker, permutation):
        # Avoid dimensions of size 1 and multiples of `num_macs` for a thorough test.
        input_shape = (2, 3, 5, 3, 5)
        model = PermuteModule(permutation)
        if permutation == (0, 1, 2, 3, 4):
            # Identity permutation is a no-op on Neutron. As it's the only node in the testing model, it's delegation
            #  would result in an empty graph, which is not allowed. Therefore, it's not delegated.
            self.assert_not_delegated(model, input_shape)
        else:
            self.assert_delegated(model, input_shape, mocker)
