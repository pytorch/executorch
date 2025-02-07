# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch
import torchvision

from executorch.backends.cadence.aot.export_example import export_model


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":

    model = torchvision.models.vit_b_16()
    example_inputs = (torch.randn(1, 3, 224, 224),)

    export_model(model, example_inputs)
