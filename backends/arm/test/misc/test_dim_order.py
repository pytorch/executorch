# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)


input_t1 = Tuple[torch.Tensor]  # Input x


class ChannelsLastInput(torch.nn.Module):
    """
    Test a complex case with (channels last, channels first) input,
    and  (channels first, channels last) output.
    """

    inputs: input_t1 = (
        torch.arange(1, 25, dtype=torch.float32)
        .reshape((1, 2, 3, 4))
        .to(memory_format=torch.channels_last),
        torch.arange(1, 25, dtype=torch.float32).reshape((1, 2, 3, 4)),
    )

    def forward(self, x, y):
        x = x * x
        return y, x


class ChannelsFirstOutput(torch.nn.Module):
    """
    Test coverting to channels_first inside the delegate.
    """

    inputs: input_t1 = (
        torch.arange(1, 25, dtype=torch.float32)
        .reshape((1, 2, 3, 4))
        .to(memory_format=torch.channels_last),
    )

    def forward(self, x):
        x = x.clone(memory_format=torch.contiguous_format) * x
        return x


class ChannelsLastOutput(torch.nn.Module):
    """
    Test changing of dim_order inside the delegate.
    """

    inputs: input_t1 = (torch.arange(1, 9, dtype=torch.float32).reshape((1, 2, 2, 2)),)

    def forward(self, x):
        x = x * x
        x = x.clone(memory_format=torch.channels_last)
        return x


class ChannelsLastInsidePartition(torch.nn.Module):
    """
    Test dim_order changes inside the partiton, but no dim_order changes at input/output.
    """

    inputs: input_t1 = (torch.randn((1, 2, 3, 3)),)

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3, 3))

    def forward(self, x):
        return (
            self.conv2d(x.clone(memory_format=torch.channels_last)).clone(
                memory_format=torch.contiguous_format
            )
            * 1
        )


test_modules = {
    "channels_last_input": ChannelsLastInput,
    "channels_first_output": ChannelsFirstOutput,
    "channels_last_output": ChannelsLastOutput,
    "channels_last_inside_partition": ChannelsLastInsidePartition,
}


@common.parametrize("module", test_modules)
def test_dim_order_tosa_FP(module):
    pipeline = TosaPipelineFP[input_t1](module(), module.inputs, [])
    pipeline.run()


@common.parametrize("module", test_modules)
def test_dim_order_tosa_INT(module):
    pipeline = TosaPipelineINT[input_t1](
        module(), module.inputs, [], symmetric_io_quantization=True
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("module", test_modules)
def test_dim_order_u55_INT(module):
    pipeline = EthosU55PipelineINT[input_t1](module(), module.inputs, [])
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("module", test_modules)
def test_dim_order_u85_INT(module):
    pipeline = EthosU85PipelineINT[input_t1](module(), module.inputs, [])
    pipeline.run()
