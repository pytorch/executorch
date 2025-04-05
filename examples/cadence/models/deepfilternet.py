# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch
from df.enhance import init_df

from executorch.backends.cadence.aot.export_example import export_model

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    logging.info("DeepFilterNet requires pip install -q -U deepfilternet")
    model, _, _= init_df()
    model.eval()

    spec = torch.randn(1, 1, 1061, 481, 2)
    spec_feat = torch.randn(1, 1, 1061, 96, 2)
    erb_feat = torch.randn(1, 1, 1061, 32)

    example_inputs = (
        spec,
        erb_feat,
        spec_feat,
    )

    export_model(model, example_inputs)


if __name__ == "__main__":
    main()
