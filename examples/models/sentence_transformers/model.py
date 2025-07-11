# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from sentence_transformers import SentenceTransformer as HFSentenceTransformer

from ..model_base import EagerModelBase


class SentenceTransformersWrapper(torch.nn.Module):
    """Wrapper for Sentence Transformers model to make it torch.export compatible"""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()

        self.model = HFSentenceTransformer(model_name, device="cpu")
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        # Get sentence embeddings
        with torch.no_grad():
            # Use the underlying transformer model directly
            features = {"input_ids": input_ids, "attention_mask": attention_mask}
            embeddings = self.model[0](features)  # Get transformer outputs
            embeddings = self.model[1](embeddings)  # Apply pooling

        return embeddings["sentence_embedding"]


class SentenceTransformersModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading Sentence Transformers model from HuggingFace")
        model = SentenceTransformersWrapper("sentence-transformers/all-MiniLM-L6-v2")
        model.eval()
        logging.info("Loaded Sentence Transformers model")
        return model

    def get_example_inputs(self):
        # Example inputs for Sentence Transformers
        # Text: batch_size=1, max_length=128
        input_ids = torch.randint(0, 30522, (1, 128))  # BERT vocab size
        attention_mask = torch.ones(1, 128)

        return (input_ids, attention_mask)
