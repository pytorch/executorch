# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import torch

from executorch.backends.cadence.aot.export_example import export_model
# from torchaudio.models.wav2vec2.model import wav2vec2_model, Wav2Vec2Model
from df.enhance import init_df

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    # The wrapper is needed to avoid issues with the optional second arguments

    model,_ ,_= init_df() 
    model.eval()
    # test input
    # spec = torch.load("examples/cadence/models/spec.pt")
    spec = torch.rand([1, 1, 1061, 481, 2])
    # spec_feat = torch.load("examples/cadence/models/spec_feat.pt")
    spec_feat = torch.rand([1, 1, 1061, 96, 2])
    # erb_feat = torch.load("examples/cadence/models/erb_feat.pt")
    erb_feat = torch.rand([1, 1, 1061, 32])
    
    check = model(spec,erb_feat,spec_feat)
    print("check")

    example_inputs = (spec,spec_feat,erb_feat,)

    export_model(model, example_inputs)


if __name__ == "__main__":
    main()
