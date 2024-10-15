# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch
from encodec import EncodecModel

from executorch.backends.cadence.aot.export_example import export_model

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    logging.info("Encodec requires pip install -U encodec")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()

    wav = torch.randn(1, 1, 144000)

    example_inputs = (wav,)

    export_model(model, example_inputs)


if __name__ == "__main__":
    main()
