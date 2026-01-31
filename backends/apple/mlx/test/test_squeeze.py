"""
Test cases for Squeeze operation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class SqueezeModel(nn.Module):
    """Model that squeezes a tensor at specified dimensions."""

    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.dims) == 0:
            # Squeeze all dimensions of size 1
            return torch.squeeze(x)
        else:
            # Squeeze specific dimensions
            return torch.squeeze(x, dim=self.dims)


@register_test
class SqueezeTest(OpTestCase):
    """Test case for squeeze op."""

    name = "squeeze"
    rtol = 1e-5
    atol = 1e-5

    def __init__(
        self, shape: Tuple[int, ...] = (1, 3, 1, 4), dims: Tuple[int, ...] = (0, 2)
    ):
        self.shape = shape
        self.dims = dims
        shape_str = "x".join(str(s) for s in shape)
        dims_str = "_".join(str(d) for d in dims) if dims else "all"
        self.name = f"squeeze_{shape_str}_dims{dims_str}"

    @classmethod
    def get_test_configs(cls) -> List["SqueezeTest"]:
        """Return all test configurations to run."""
        return [
            cls(shape=(1, 3, 1, 4), dims=(0, 2)),  # Squeeze specific dims
            cls(shape=(1, 5, 1, 1), dims=(0,)),  # Squeeze first dim only
            cls(shape=(3, 1, 4), dims=(1,)),  # Squeeze middle dim
            cls(shape=(1, 1, 8), dims=(0, 1)),  # Squeeze first two dims
            cls(shape=(2, 1, 3, 1), dims=(1, 3)),  # Squeeze non-consecutive dims
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Create tensor with the specified shape
        x = torch.randn(self.shape)
        return (x,)

    def create_model(self) -> nn.Module:
        return SqueezeModel(self.dims)
