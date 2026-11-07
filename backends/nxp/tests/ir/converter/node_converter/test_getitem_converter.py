# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from backends.nxp.tests.model_output_comparator import AllCloseOutputComparator
from executorch.backends.nxp.backend.ops_aliases import (
    AddTensor,
    GetItem,
    MaxPool2DWithIndices,
    PermuteCopy,
)

from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class MaxPool2dModule(torch.nn.Module):
    """MaxPool2d whose output (via getitem) is the model output."""

    def __init__(self):
        super().__init__()
        # stride=1 and padding=1 keep spatial dimensions unchanged, which
        # simplifies building downstream modules that operate on the same shape.
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.max_pool2d(x)


class MaxPoolAddModule(torch.nn.Module):
    """MaxPool2d output (via getitem) consumed by a single add operation."""

    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x, y):
        x = self.max_pool2d(x)
        return torch.add(x, y)


class MaxPoolPermuteModule(torch.nn.Module):
    """MaxPool2d output (via getitem) consumed by a single permute operation."""

    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.max_pool2d(x)
        return torch.permute(x, (0, 2, 3, 1))


class MaxPoolAddPermuteModule(torch.nn.Module):
    """MaxPool2d output (via getitem) consumed by both an add and a permute.

    This models a scenario where two different downstream compute nodes share
    the same getitem as their input, exercising getitem with multiple consumers.
    """

    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x, y):
        pooled = self.max_pool2d(x)
        added = torch.add(pooled, y)
        permuted = torch.permute(pooled, (0, 2, 3, 1))
        # Combine both branches into a single tensor to avoid a tuple output,
        # which would introduce extra getitem nodes for output unpacking.
        return torch.add(added, torch.permute(permuted, (0, 3, 1, 2)))


class TestGetItem:
    """Tests verifying that the GetItem converter always delegates the node.

    GetItemConverter._is_supported_in_IR unconditionally returns True, so every
    getitem node that appears in the graph (always paired with MaxPool2DWithIndices
    in practice) must end up inside the delegated partition.

    The test scenarios cover the different downstream contexts for the getitem
    output: no consumer (model output), a compute op (add), a layout op
    (permute_copy), and a mix of both.
    """

    # noinspection PyMethodMayBeStatic
    def assert_delegated(
        self, model, input_shape, mocker, request, expected_delegated_ops, use_qat=False
    ):
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops=expected_delegated_ops,
            expected_non_delegated_ops={},
        )
        # Some tests produce slightly inaccurate outputs, which however cannot be caused by the `getitem`. For the
        #  purposes of this testing module, the important part is that the lowering succeeds and the model runs without
        #  crashing. A small error is tolerable.
        output_comparator = AllCloseOutputComparator(atol=0.12)
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            output_comparator=output_comparator,
            use_qat=use_qat,
        )

    def test__basic__getitem_is_model_output(self, mocker, request, use_qat):
        # The getitem node is the direct output of the model -- no downstream op.
        # Parametrized over PTQ and QAT via the use_qat fixture.
        input_shape = (1, 4, 6, 6)
        model = MaxPool2dModule()
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1},
            use_qat=use_qat,
        )

    def test__getitem_followed_by_add(self, mocker, request):
        # The getitem output is consumed by an AddTensor node.
        input_shape = (1, 4, 6, 6)
        model = MaxPoolAddModule()
        self.assert_delegated(
            model,
            [input_shape, input_shape],
            mocker,
            request,
            expected_delegated_ops={MaxPool2DWithIndices: 1, GetItem: 1, AddTensor: 1},
        )

    def test__getitem_followed_by_permute_copy(self, mocker, request):
        # The getitem output is consumed by a PermuteCopy node.
        input_shape = (1, 4, 6, 6)
        model = MaxPoolPermuteModule()
        self.assert_delegated(
            model,
            input_shape,
            mocker,
            request,
            expected_delegated_ops={
                MaxPool2DWithIndices: 1,
                GetItem: 1,
                PermuteCopy: 1,
            },
        )

    def test__getitem_followed_by_add_and_permute_copy(self, mocker, request):
        # The getitem output has two distinct consumers: an AddTensor and a
        # PermuteCopy. This verifies that getitem is correctly delegated even
        # when it feeds into multiple downstream nodes simultaneously.
        input_shape = (1, 4, 6, 6)
        model = MaxPoolAddPermuteModule()
        self.assert_delegated(
            model,
            [input_shape, input_shape],
            mocker,
            request,
            expected_delegated_ops={
                MaxPool2DWithIndices: 1,
                GetItem: 1,
                AddTensor: 2,
                PermuteCopy: 2,
            },
        )
