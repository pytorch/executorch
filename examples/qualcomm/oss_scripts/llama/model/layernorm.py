# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Dict, Type

import torch


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
        NORM_REGISTRY[name] = cls
        return cls

    return decorator


@register_norm("layernorm")
class LayerNorm(torch.nn.LayerNorm, Norm):
    def __init__(self, hidden_size: int, eps=1e-5):
        super().__init__(hidden_size, eps=eps)


@register_norm("gemma3")
@register_norm("rmsnorm")
class RMSNorm(torch.nn.RMSNorm, Norm):
    def __init__(self, hidden_size: int, eps=1e-5):
        super().__init__(hidden_size, eps=eps)
