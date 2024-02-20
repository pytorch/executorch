# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from .meta_registrations import *  # noqa

import torch
from executorch.examples.xtensa.aot.export_example import export_xtensa_model


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
    ) = [(1, 8, 33), 8, 16, 3, 2, 4, 3, False, False, False]

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
            if channel_last:
                x = x.permute([0, 2, 1])
            x = self.conv1d(x)
            if channel_last:
                x = x.permute([0, 2, 1])
            return x

    model = QuantizedConv(shape)
    model.eval()

    example_inputs = (torch.randn(shape),)

    export_xtensa_model(model, example_inputs)
