# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
import torch.nn.functional as F


class Norm(torch.nn.Module, ABC):
    """Abstract base class for normalization layers with unified interface."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for normalization layer.
        Args:
            x: Input tensor
        Returns:
            Normalized tensor
        """
        pass


NORM_REGISTRY: Dict[str, Type[Norm]] = {}


def register_norm(name: str):
    """Decorator to register norm classes"""

    def decorator(cls: Type[Norm]):
        NORM_REGISTRY[name.lower()] = cls
        return cls

    return decorator


# Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo/modular_olmo.py
@register_norm("olmo")
class OlmoLayerNorm(Norm):
    """LayerNorm but with no learnable weight or bias."""

    def __init__(self, hidden_size: int, eps=1e-5) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(
            hidden_states.to(dtype=torch.float32),
            self.normalized_shape,
            None,
            None,
            eps=self.eps,
        ).to(orig_dtype)
