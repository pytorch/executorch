# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)
from examples.models.llama2.model import ModelArgs, Transformer

class LLAMA2Model:
    def __init__(self):
        cur_path = os.path.dirname(__file__)
        checkpoint = os.path.join(cur_path, 'tinyllamas/stories260K/stories260K.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        gptconf = ModelArgs(**checkpoint_dict['model_args'])
        self.model_ = Transformer(gptconf)

    # @staticmethod
    def get_eager_model(self):
        return self.model_

    @staticmethod
    def get_example_inputs():
        return (torch.tensor([[1]]),)
