# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from ..model_base import EagerModelBase
from torchaudio import models

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


class Wav2LetterModel(EagerModelBase):
    def __init__(self):
        self.batch_size = 10
        self.input_frames = 700
        self.vocab_size = 4096

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading wav2letter model")
        wav2letter = models.Wav2Letter(num_classes=self.vocab_size)
        logging.info("loaded wav2letter model")
        return wav2letter

    def get_example_inputs(self):
        input_shape = (self.batch_size, 1, self.input_frames)
        return (torch.randn(input_shape),)
