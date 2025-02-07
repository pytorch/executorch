# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torch.nn.quantizable.modules import rnn

from ..model_base import EagerModelBase


class LSTMModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading LSTM model")
        lstm = rnn.LSTM(10, 20, 2)
        logging.info("Loaded LSTM model")
        return lstm

    def get_example_inputs(self):
        input_tensor = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)
        return (input_tensor, (h0, c0))
