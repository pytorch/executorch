# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineINT,
)


input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input y


class ChannelsLastInput(torch.nn.Module):
    """
    Test rejection of a partition which has a channels last input.
    """

    inputs: input_t1 = (
        torch.randn(1, 2, 2, 2).to(memory_format=torch.channels_last),
        torch.randn(1, 2, 2, 2),
    )

    def forward(self, x, y):
        x = x * y
        x = x.to(dtype=torch.int32, memory_format=torch.channels_last)
        x = x / 2
        return x, y


class ChannelsLastOutput(torch.nn.Module):
    """
    Test rejection of a partition which has a channels last output
    """

    inputs: input_t1 = (
        torch.randn(
            1,
            2,
            2,
            2,
        ),
        torch.randn(1, 2, 2, 2),
    )

    def forward(self, x, y):
        x = x * y
        x = x.clone(memory_format=torch.channels_last)
        x = x / 2
        return x, y


class ChannelsLastInsidePartition(torch.nn.Module):
    """
    Test a non rejection of a fully partitioned module which changes memory inside the partition.
    The TOSA backend ignores this memory format change, and since the input and output
    has the expected channels_last memory format, the partition should be accepted.
    """

    inputs: input_t1 = (
        torch.randn(
            1,
            2,
            2,
            2,
        ),
        torch.randn(1, 2, 2, 2),
    )

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)

    def forward(self, x, y):
        x = x * y
        x = x.to(memory_format=torch.channels_last)
        x = self.conv(x)
        x = x.clone(memory_format=torch.contiguous_format)
        return x, y


def test_dim_order_ok():
    pipeline = TosaPipelineINT[input_t1](
        ChannelsLastInsidePartition(), ChannelsLastInsidePartition.inputs, []
    )
    pipeline.run()


def test_channels_last_input():
    pipeline = OpNotSupportedPipeline[input_t1](
        ChannelsLastInput(),
        ChannelsLastInput.inputs,
        non_delegated_ops={},
        n_expected_delegates=0,
    )
    pipeline.run()


def test_channels_last_output():
    pipeline = OpNotSupportedPipeline[input_t1](
        ChannelsLastOutput(),
        ChannelsLastOutput.inputs,
        non_delegated_ops={},
        n_expected_delegates=0,
    )
    pipeline.run()
