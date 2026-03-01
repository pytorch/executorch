# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting simple models to flatbuffer

import logging

from typing import cast, Sequence

import torch

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

from executorch.backends.cadence.aot.export_example import export_and_run_model


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":
    (
        shape,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        dilation,
        depthwise,
        bias,
        channel_last,
    ) = [(1, 2, 4), 2, 8, 1, 1, 0, 1, False, False, False]

    class QuantizedConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels if depthwise else 1,
                bias=bias,
            )

        def forward(self, x: torch.Tensor):
            return self.conv1d(x)

    model = QuantizedConv()
    model.eval()

    example_inputs = (torch.randn(cast(Sequence[int], shape)),)

    export_and_run_model(model, example_inputs)

    model = QuantizedConv()
    model.eval()

    example_inputs = (torch.randn(cast(Sequence[int], shape)),)

    export_and_run_model(model, example_inputs, opt_level=3)
