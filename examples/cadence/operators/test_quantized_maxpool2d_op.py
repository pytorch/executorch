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
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    ) = [(1, 64, 112, 112), 3, 2, 1, 1, False]

    class QuantizedMaxPool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.MaxPool2d = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=False,
            )

        def forward(self, x: torch.Tensor):
            return self.MaxPool2d(x)

    model = QuantizedMaxPool()
    model.eval()

    example_inputs = (torch.randn(cast(Sequence[int], shape)),)

    export_and_run_model(model, example_inputs)
