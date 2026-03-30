# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import AutoTokenizer, DistilBertModel  # @manual

from ..model_base import EagerModelBase


class DistilBertModelExample(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading DistilBERT model")
        model = DistilBertModel.from_pretrained(
            "distilbert/distilbert-base-uncased", return_dict=False
        )
        logging.info("Loaded DistilBERT model")
        return model

    def get_example_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        return (tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"],)
