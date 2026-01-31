"""
Test cases for Conv2D operation.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .test_utils import OpTestCase, register_test


class Conv2DModel(nn.Module):
    """Model that performs 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_test
class Conv2DTest(OpTestCase):
    """Test case for conv2d op."""

    name = "conv2d"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        input_size: Tuple[int, int] = (32, 32),
        batch_size: int = 1,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.batch_size = batch_size
        self.bias = bias

        # Generate descriptive name
        parts = [
            "conv2d",
            f"in{in_channels}",
            f"out{out_channels}",
            f"k{kernel_size}",
        ]
        if stride != 1:
            parts.append(f"s{stride}")
        if padding != 0:
            parts.append(f"p{padding}")
        parts.append(f"{input_size[0]}x{input_size[1]}")
        if batch_size != 1:
            parts.append(f"b{batch_size}")
        if not bias:
            parts.append("nobias")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["Conv2DTest"]:
        """Return all test configurations to run."""
        return [
            # Basic 3x3 conv
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(32, 32),
            ),
            # Stride 2
            cls(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                input_size=(64, 64),
            ),
            # 1x1 conv (common in ResNet)
            cls(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                input_size=(16, 16),
            ),
            # 5x5 conv
            cls(
                in_channels=3,
                out_channels=8,
                kernel_size=5,
                padding=2,
                input_size=(28, 28),
            ),
            # Batch size > 1
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(32, 32),
                batch_size=4,
            ),
            # No bias
            cls(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                input_size=(32, 32),
                bias=False,
            ),
        ]

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size, self.in_channels, self.input_size[0], self.input_size[1]
        )
        return (x,)

    def create_model(self) -> nn.Module:
        return Conv2DModel(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
        )
