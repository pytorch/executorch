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
from torchaudio.models.wav2vec2.model import wav2vec2_model, Wav2Vec2Model

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    # The wrapper is needed to avoid issues with the optional second arguments
    # of Wav2Vec2Models.
    class Wav2Vec2ModelWrapper(torch.nn.Module):
        def __init__(self, model: Wav2Vec2Model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out, _ = self.model(x)
            return out

    _model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,
        encoder_layer_drop=0.1,
        aux_num_out=None,
    )
    _model.eval()

    model = Wav2Vec2ModelWrapper(_model)
    model.eval()

    # test input
    audio_len = 1680
    example_inputs = (torch.rand(1, audio_len),)

    export_model(model, example_inputs)


if __name__ == "__main__":
    main()
