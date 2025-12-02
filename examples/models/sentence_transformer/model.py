# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from transformers import AutoModel, AutoTokenizer  # @manual

try:
    from ..model_base import EagerModelBase
except ImportError:
    # If running as a script, we don't need EagerModelBase
    EagerModelBase = object


class SentenceTransformerModel(torch.nn.Module):
    """
    Wrapper for sentence-transformers models that includes mean pooling.
    This creates a model suitable for generating sentence embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings, taking attention mask into account.

        Args:
            token_embeddings: Token-level embeddings from the model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Sentence embeddings [batch_size, hidden_size]
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that generates sentence embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Sentence embeddings [batch_size, hidden_size]
        """
        # Get model outputs
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state
        token_embeddings = model_output.last_hidden_state

        # Apply mean pooling
        sentence_embeddings = self.mean_pooling(token_embeddings, attention_mask)

        return sentence_embeddings


class SentenceTransformerModelExample(EagerModelBase):
    """
    Example implementation for exporting sentence-transformers models to ExecuTorch.
    Supports models like all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def get_eager_model(self) -> torch.nn.Module:
        logging.info(f"Loading sentence transformer model: {self.model_name}")
        model = SentenceTransformerModel(self.model_name)
        model.eval()
        logging.info(f"Loaded sentence transformer model: {self.model_name}")
        return model

    def get_example_inputs(self):
        """
        Returns example inputs for model tracing.
        Uses a sample sentence to generate input_ids and attention_mask.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        text = "This is an example sentence for generating embeddings."
        encoded = tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        return (encoded["input_ids"], encoded["attention_mask"])
