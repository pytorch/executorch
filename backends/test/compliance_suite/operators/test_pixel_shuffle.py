# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class PixelShuffleModel(torch.nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        return self.pixel_shuffle(x)

@operator_test
class TestPixelShuffle(OperatorTest):
    @dtype_test
    def test_pixel_shuffle_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, C * upscale_factor^2, H, W)
        # Output shape: (batch_size, C, H * upscale_factor, W * upscale_factor)
        model = PixelShuffleModel(upscale_factor=2).to(dtype)
        self._test_op(model, (torch.rand(2, 12, 4, 4).to(dtype),), tester_factory)
        
    def test_pixel_shuffle_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (upscale_factor=2)
        # Input: (2, 12, 4, 4) -> Output: (2, 3, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 4, 4),), tester_factory)
        
    def test_pixel_shuffle_upscale_factors(self, tester_factory: Callable) -> None:
        # Test with different upscale factors
        
        # Upscale factor = 2
        # Input: (2, 12, 4, 4) -> Output: (2, 3, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 4, 4),), tester_factory)
        
        # Upscale factor = 3
        # Input: (2, 27, 4, 4) -> Output: (2, 3, 12, 12)
        self._test_op(PixelShuffleModel(upscale_factor=3), (torch.randn(2, 27, 4, 4),), tester_factory)
        
        # Upscale factor = 4
        # Input: (2, 48, 4, 4) -> Output: (2, 3, 16, 16)
        self._test_op(PixelShuffleModel(upscale_factor=4), (torch.randn(2, 48, 4, 4),), tester_factory)
        
    def test_pixel_shuffle_batch_sizes(self, tester_factory: Callable) -> None:
        # Test with different batch sizes
        
        # Batch size = 1
        # Input: (1, 12, 4, 4) -> Output: (1, 3, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(1, 12, 4, 4),), tester_factory)
        
        # Batch size = 4
        # Input: (4, 12, 4, 4) -> Output: (4, 3, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(4, 12, 4, 4),), tester_factory)
        
        # Batch size = 8
        # Input: (8, 12, 4, 4) -> Output: (8, 3, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(8, 12, 4, 4),), tester_factory)
        
    def test_pixel_shuffle_channels(self, tester_factory: Callable) -> None:
        # Test with different numbers of output channels
        
        # Output channels = 1 (grayscale)
        # Input: (2, 4, 4, 4) -> Output: (2, 1, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 4, 4, 4),), tester_factory)
        
        # Output channels = 3 (RGB)
        # Input: (2, 12, 4, 4) -> Output: (2, 3, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 4, 4),), tester_factory)
        
        # Output channels = 4 (RGBA)
        # Input: (2, 16, 4, 4) -> Output: (2, 4, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 16, 4, 4),), tester_factory)
        
        # Output channels = 16 (multi-channel)
        # Input: (2, 64, 4, 4) -> Output: (2, 16, 8, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 64, 4, 4),), tester_factory)
        
    def test_pixel_shuffle_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        
        # Small input
        # Input: (2, 12, 2, 2) -> Output: (2, 3, 4, 4)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 2, 2),), tester_factory)
        
        # Medium input
        # Input: (2, 12, 8, 8) -> Output: (2, 3, 16, 16)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 8, 8),), tester_factory)
        
        # Large input
        # Input: (2, 12, 16, 16) -> Output: (2, 3, 32, 32)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 16, 16),), tester_factory)
        
    def test_pixel_shuffle_non_square(self, tester_factory: Callable) -> None:
        # Test with non-square input
        
        # Input: (2, 12, 4, 8) -> Output: (2, 3, 8, 16)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 4, 8),), tester_factory)
        
        # Input: (2, 12, 8, 4) -> Output: (2, 3, 16, 8)
        self._test_op(PixelShuffleModel(upscale_factor=2), (torch.randn(2, 12, 8, 4),), tester_factory)
