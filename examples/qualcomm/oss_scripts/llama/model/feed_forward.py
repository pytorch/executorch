# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
from executorch.examples.models.llama.model_args import ModelArgs
from transformers.activations import GELUActivation


class FeedForwardBase(torch.nn.Module, ABC):
    """Abstract base class for feed forward layers with unified interface."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed forward layer.
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        pass


FeedForward_REGISTRY: Dict[str, Type[FeedForwardBase]] = {}


def register_feed_forward(name: str):
    """Decorator to register norm classes"""

    def decorator(cls: Type[FeedForwardBase]):
        FeedForward_REGISTRY[name] = cls
        return cls

    return decorator


@register_feed_forward("CodeGenModel")
class CodegenFeedForward(FeedForwardBase):
    """FeedForward with fc_in and fc_out"""

    def __init__(self, args: ModelArgs):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()

        assert args.hidden_dim is not None
        self.dim = args.dim
        self.hidden_dim: int = args.hidden_dim

        self.fc_in = torch.nn.Linear(self.dim, self.hidden_dim)
        self.fc_out = torch.nn.Linear(self.hidden_dim, self.dim)
        # HF uses NewGelu, however, Gelu is a fused op in QNN and can run faster
        self.act = GELUActivation(use_gelu_python=False)

    def prepare_feedfoward_conv(self):
        intermediate_size = 4 * self.dim
        self.fc_in_conv = torch.nn.Conv2d(self.dim, intermediate_size, 1, bias=True)
        self.fc_out_conv = torch.nn.Conv2d(self.hidden_dim, self.dim, 1, bias=True)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedfoward_conv

        self.fc_in_conv.weight.data.copy_(self.fc_in.weight[:, :, None, None])
        self.fc_out_conv.weight.data.copy_(self.fc_out.weight[:, :, None, None])

        self.fc_in_conv.bias.data.copy_(self.fc_in.bias)
        self.fc_out_conv.bias.data.copy_(self.fc_out.bias)

        del self.fc_in
        del self.fc_out

    def forward_feedfoward_conv(self, x):
        bsz, _, _ = x.size()

        x = torch.reshape(x, (bsz, -1, 1, self.dim))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.fc_in_conv(x)
        x = self.act(x)
        x = self.fc_out_conv(x)
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.dim))
        return x

    def forward(self, x):
        hidden_states = self.fc_in(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states
