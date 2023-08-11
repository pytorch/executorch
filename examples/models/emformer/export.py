# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchaudio.models import Emformer


FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


__all__ = ["Emformer"]


class EmformerModel:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading emformer model")
        emformer = Emformer(32, 8, 128, 20, 4, right_context_length=1)
        logging.info("loaded emformer model")
        return emformer

    @staticmethod
    def get_example_inputs():
        # TODO: Try with different shapes.
        input = torch.rand(1, 4, 32)  # batch, num_frames, feature_dim
        lengths = torch.randint(1, 20, (12,))  # batch
        return (input, lengths)
