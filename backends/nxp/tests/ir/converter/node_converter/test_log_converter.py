# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# noinspection PyUnusedImports
import pytest
import torch

from executorch.backends.nxp.tests.graph_verifier import DetailedGraphVerifier
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from executorch.backends.nxp.tests.ops_aliases import Clamp, Convolution, Log, Relu
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.backends.nxp.tests.dataset_creator import LinearRampDatasetCreator


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class LogModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)


class ConvBlocksWithLogModule(torch.nn.Module):
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
        x = self.block1(x)
        x = x.clamp_min(1e-6).log()
        return self.block2(x)


class TestLog:
    def test__basic_nsys_inference(self, mocker):
        # Use 256 elements so that, after quantization to uint8, the input can
        # cover the full discrete range [0, 255].
        # The dataset is generated as a linear float ramp and later quantized,
        # which effectively exercises all uint8 values.
        input_shape = (256,)
        model = LogModule()
        graph_verifier = DetailedGraphVerifier(
            mocker, expected_delegated_ops={Log: 1}, expected_non_delegated_ops={}
        )
        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
            dataset_creator=LinearRampDatasetCreator(),
        )

    def test__basic_nsys_inference__with_conv(self, mocker):
        input_shape = (2, 3, 6, 7)
        in_channels = input_shape[1]
        model = ConvBlocksWithLogModule(conv_in_channels=in_channels)

        # `clamp` and one `relu` ends up in the same delegated partition as `log`
        graph_verifier = DetailedGraphVerifier(
            mocker,
            expected_delegated_ops={Log: 1, Relu: 1, Clamp: 1},
            expected_non_delegated_ops={Relu: 1, Convolution: 2},
        )

        lower_run_compare(
            model,
            input_shape,
            graph_verifier,
        )
