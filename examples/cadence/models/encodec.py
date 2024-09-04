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
# from torchaudio.models.wav2vec2.model import wav2vec2_model, Wav2Vec2Model

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    class wrapper_e(torch.nn.Module):
        def __init__(self, model):
            super(wrapper_e, self).__init__()
            self.model = model
        def forward(self, inputs):
            encoded_frames = self.model.encode(inputs)
            return encoded_frames[0][0]

    # model = wrapper_e(model)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = wrapper_e(model)
    model.eval()

    # test input

    # wav = torch.load("examples/cadence/models/wav.pt")
    wav = torch.rand([1, 1, 144000])
    print("check encodec")
    # print(model(wav))
    example_inputs = (wav,)
    print("check encodec 1")
    export_model(model, example_inputs)
    print("check encodec 2")


if __name__ == "__main__":
    main()
