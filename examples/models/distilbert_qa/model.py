# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

from ..model_base import EagerModelBase


class DistilBertQAWrapper(torch.nn.Module):
    """Wrapper for HuggingFace DistilBERT QA model to make it torch.export compatible"""

    def __init__(self, model_name="distilbert-base-cased-distilled-squad"):
        super().__init__()
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        # Get question answering outputs
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Return start and end logits for answer span
        return outputs.start_logits, outputs.end_logits


class DistilBertQAModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading DistilBERT QA model from HuggingFace")
        model = DistilBertQAWrapper("distilbert-base-cased-distilled-squad")
        model.eval()
        logging.info("Loaded DistilBERT QA model")
        return model

    def get_example_inputs(self):
        # Example inputs for DistilBERT QA
        # Combined question and context: batch_size=1, max_length=512
        input_ids = torch.randint(0, 28996, (1, 512))  # DistilBERT vocab size
        attention_mask = torch.ones(1, 512)

        return (input_ids, attention_mask)
