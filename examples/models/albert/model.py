# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from transformers import AlbertModel, AutoTokenizer  # @manual

from ..model_base import EagerModelBase


class AlbertModelExample(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading ALBERT model")
        # pyre-ignore
        model = AlbertModel.from_pretrained("albert-base-v2", return_dict=False)
        model.eval()
        logging.info("Loaded ALBERT model")
        return model

    def get_example_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        return (tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"],)
