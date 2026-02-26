# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JetBlock Model for ExecutorTorch.

This module provides a JetBlockModel class that wraps the JetBlock attention
mechanism for use with ExecutorTorch's export and runtime infrastructure.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from executorch.examples.models.jet_nemotron.jet_block import (
    JetBlock,
    JetBlockConfig,
)
from executorch.examples.models.model_base import EagerModelBase


class JetBlockStack(nn.Module):
    """
    A simple stack of JetBlock layers for demonstration.

    This can be used as a building block in larger transformer models,
    or as a standalone module for experimenting with the JetBlock style.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_layers: int = 1,
        config: Optional[JetBlockConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if config is None:
            config = JetBlockConfig()

        self.layers = nn.ModuleList(
            [
                JetBlock(
                    hidden_size=hidden_size,
                    config=config,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the JetBlock stack.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        for layer in self.layers:
            residual = hidden_states
            hidden_states, _, _ = layer(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        return hidden_states


class JetBlockModel(EagerModelBase):
    """
    JetBlock model wrapper for ExecutorTorch.

    This class provides the interface expected by ExecutorTorch's model
    infrastructure, including methods for getting the eager model and
    example inputs for export.

    Example usage:
        >>> model = JetBlockModel()
        >>> eager_model = model.get_eager_model()
        >>> inputs = model.get_example_inputs()
        >>> output = eager_model(*inputs)
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_layers: int = 1,
        num_heads: int = 6,
        head_dim: int = 256,
        batch_size: int = 1,
        seq_len: int = 128,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize JetBlockModel.

        Args:
            hidden_size: Hidden dimension size (default: 1536)
            num_layers: Number of JetBlock layers (default: 1)
            num_heads: Number of attention heads (default: 6)
            head_dim: Dimension per head (default: 256)
            batch_size: Batch size for example inputs (default: 1)
            seq_len: Sequence length for example inputs (default: 128)
            dtype: Data type for the model (default: torch.float32)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dtype = dtype

        self.config = JetBlockConfig(
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def get_eager_model(self) -> nn.Module:
        """
        Get the eager PyTorch model.

        Returns:
            An instance of JetBlockStack configured with the model parameters.
        """
        model = JetBlockStack(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            config=self.config,
        )
        model.eval()
        return model.to(self.dtype)

    def get_example_inputs(self) -> Tuple[torch.Tensor]:
        """
        Get example inputs for tracing and export.

        Returns:
            A tuple containing example input tensors.
        """
        hidden_states = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_size,
            dtype=self.dtype,
        )
        return (hidden_states,)

    def get_dynamic_shapes(self):
        """
        Get dynamic shapes for export.

        Returns:
            Dynamic shape specifications for the input tensors.
        """
        # Sequence length can be dynamic
        return {
            "hidden_states": {
                0: torch.export.Dim("batch", min=1, max=32),
                1: torch.export.Dim("seq_len", min=1, max=8192),
            }
        }
