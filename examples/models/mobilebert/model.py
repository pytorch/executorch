# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from transformers import AutoTokenizer, MobileBertModel  # @manual

from ..model_base import EagerModelBase


class MobileBertModelExample(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("loading mobilebert model")
        # pyre-ignore
        model = MobileBertModel.from_pretrained(
            "google/mobilebert-uncased", return_dict=False
        )
        logging.info("loaded mobilebert model")
        return model

    def get_example_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        return (tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"],)
