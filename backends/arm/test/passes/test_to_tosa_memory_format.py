# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import ToTosaMemoryFormatPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass

input_t = Tuple[torch.Tensor]  # Input x


class NoNHWC(torch.nn.Module):
    """
    Test-module with no ops requiring NHWC mermory format.
    """

    ops_after_pass = {"executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2}
    ops_not_after_pass = []

    def forward(self, x):
        x = x + x
        return x

    def get_inputs(self):
        return (torch.rand(1, 2, 2, 2),)


class ParallelClusters(torch.nn.Module):
    """
    Test-module with multiple parallel clusters of nodes requiring different memory formats.
    """

    ops_after_pass = {"executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 2}
    ops_not_after_pass = []

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

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        x4 = x * x
        return x1 + x2 + x3 + x4

    def get_inputs(self):
        return (torch.rand(1, 2, 2, 2),)


class SerialClusters(torch.nn.Module):
    """
    Test-module with multiple serial clusters of nodes requring different memory formats.
    """

    ops_before_pass = {}
    ops_after_pass = {"executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 4}
    ops_not_after_pass = []

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

    def forward(self, x):
        x = self.conv(x)
        x = x * x
        x = self.conv(x)
        x = x.view((2, 1, 2, 4))
        x = x * 2
        x = x.view((2, 2, 2, 2))
        x = self.conv(x)
        return x

    def get_inputs(self):
        return (torch.rand(2, 2, 2, 2),)


class Reshapes(torch.nn.Module):
    """
    Test-module with different configurations of views requiring different memory formats.
    """

    ops_before_pass = {}
    ops_after_pass = {
        "executorch_exir_dialects_backend__ops_tosa_TRANSPOSE_default": 16
    }
    ops_not_after_pass = []

    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(1, 1)  # Use maxpool to force NHWC format

    def forward(self, x):

        x = self.maxpool(x)
        x = x.view((2, 2, 4, 16, 1))  # N-C-HW-invariant intact, no transposes needed
        x = x * 2  # Add op to avoid views merging
        x = x.view((4, 4, 4, 4))
        x = x / 2  # Add op to avoid views merging
        x = self.maxpool(x)

        x = x.view((256))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((16, 16))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((16, 4, 4))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((2, 4, 4, 8))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = x / 2
        x = self.maxpool(x)

        x = x.view((8, 1, 2, 4, 4))  # Break N-C-HW invariant
        x = x * 2
        x = x.view((4, 4, 4, 4))
        x = self.maxpool(x)

        return x

    def get_inputs(self):
        return (torch.rand(4, 4, 4, 4),)


modules = {
    "no_nhwc": NoNHWC(),
    "parallel_clusters": ParallelClusters(),
    "serial_clusters": SerialClusters(),
    "reshapes": Reshapes(),
}


@common.parametrize("module", modules)
def test_to_tosa_memory_format_tosa_INT(module):
    # We cannot check op counts after a specific pass with the full pipeline
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        pass_list=[RemoveGetItemPass],
        passes_with_exported_program=[ToTosaMemoryFormatPass],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # Eager execution is not possible after introducing tosa.TRANSPOSE
    pipeline.run()


@common.parametrize("module", modules)
def test_to_tosa_memory_format_tosa_INT_functional(module):
    # Also run the actual pass pipeline to ensure functional correctness.
    pipeline = TosaPipelineINT[input_t](module, module.get_inputs(), [])
    pipeline.run()
