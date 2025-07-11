# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from ..model_base import EagerModelBase


class RobertaSentimentWrapper(torch.nn.Module):
    """Wrapper for HuggingFace RoBERTa sentiment model to make it torch.export compatible"""

    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        # Sentiment classification
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Return classification logits
        return outputs.logits


class RobertaSentimentModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading RoBERTa sentiment model from HuggingFace")
        model = RobertaSentimentWrapper(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        model.eval()
        logging.info("Loaded RoBERTa sentiment model")
        return model

    def get_example_inputs(self):
        # Example inputs for RoBERTa sentiment
        # Text: batch_size=1, max_length=512
        input_ids = torch.randint(0, 50265, (1, 512))  # RoBERTa vocab size
        attention_mask = torch.ones(1, 512)

        return (input_ids, attention_mask)
