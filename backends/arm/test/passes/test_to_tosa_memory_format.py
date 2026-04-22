# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor]  # Input x


class NoNHWC(torch.nn.Module):
    """Test-module with no ops requiring NHWC memory format."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 2, 2, 2),)


class ParallelClusters(torch.nn.Module):
    """Test-module with multiple parallel clusters of nodes requiring different
    memory formats.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=1,
            bias=True,
        )
        self.maxpool = torch.nn.MaxPool2d(1, 1)
        self.avgpool = torch.nn.AvgPool2d(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        x4 = x * x
        return x1 + x2 + x3 + x4

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 2, 2, 2),)


class SerialClusters(torch.nn.Module):
    """Test-module with multiple serial clusters of nodes requiring different
    memory formats.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=1,
            bias=True,
        )
        self.fc = torch.nn.Linear(
            in_features=2,
            out_features=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x * x
        x = self.conv(x)
        x = x.view((2, 1, 2, 4))
        x = x * 2
        x = x.view((2, 2, 2, 2))
        x = self.conv(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(2, 2, 2, 2),)


class Reshapes(torch.nn.Module):
    """Test-module with different configurations of views."""

    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.maxpool(x)
        x = x.view((2, 2, 4, 16, 1))
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((256))
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((16, 16))
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((16, 4, 4))
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((2, 4, 4, 8))
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((8, 1, 2, 4, 4))
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = self.maxpool(x)

        return x

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4, 4, 4),)


modules = {
    "no_nhwc": NoNHWC(),
    "parallel_clusters": ParallelClusters(),
    "serial_clusters": SerialClusters(),
    "reshapes": Reshapes(),
}


@common.parametrize("module", modules)
def test_tosa_memory_format_functional(module) -> None:
    """Run the full TOSA pipeline to ensure functional correctness."""
    module_nn = cast(torch.nn.Module, module)
    pipeline = TosaPipelineINT[input_t](module_nn, module.get_inputs(), [])
    pipeline.run()
