# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch
from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import (
    lower_run_compare,
    RandomDatasetCreator,
)
from executorch.backends.nxp.tests.ops_aliases import Abs
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class ConvBlocksWithAbsModule(torch.nn.Module):
    def __init__(self, conv_in_channels: int = 3):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=3,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=conv_in_channels,
                out_channels=10,
                kernel_size=(2, 2),
                stride=(2, 2),
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.block1(x).abs()
        return self.block2(x)


class AbsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.abs()


class TestAbs:
    @staticmethod
    def _get_dataset_creator():
        # to test `abs` reliably, we need to include negative values
        low = -255.0
        high = 255.0

        dataset = RandomDatasetCreator(low=low, high=high)
        return dataset

    def test__basic_nsys_inference(self, mocker, request):
        input_shape = (2, 3, 6, 7)
        model = AbsModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Abs: 1}, expected_non_delegated_ops={}
        )

        dataset_creator = self._get_dataset_creator()
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
        )

    def test__basic_nsys_inference__big(self, mocker, request):
        # some operators have delegation requirement that size must be < 4096
        input_shape = (4097, 1)
        model = AbsModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Abs: 1}, expected_non_delegated_ops={}
        )

        dataset_creator = self._get_dataset_creator()
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            request,
            dataset_creator,
        )
