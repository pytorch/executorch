# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from ..aot.meta_registrations import *  # noqa

import torch

from ..aot.export_example import export_xtensa_model


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":
    in_features = 32
    out_features = 16
    bias = True
    shape = [64, in_features]

    class QuantizedLinear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool):
            super().__init__()
            self.output_linear = torch.nn.Linear(in_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor):
            output_linear_out = self.output_linear(x)
            return output_linear_out

    model = QuantizedLinear(in_features, out_features, bias)
    model.eval()

    example_inputs = (torch.ones(shape),)

    export_xtensa_model(model, example_inputs)
