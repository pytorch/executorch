# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class PixelUnshuffleModel(torch.nn.Module):
    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor)
        
    def forward(self, x):
        return self.pixel_unshuffle(x)

@operator_test
class TestPixelUnshuffle(OperatorTest):
    @dtype_test
    def test_pixel_unshuffle_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, C, H * downscale_factor, W * downscale_factor)
        # Output shape: (batch_size, C * downscale_factor^2, H, W)
        model = PixelUnshuffleModel(downscale_factor=2).to(dtype)
        self._test_op(model, (torch.rand(2, 3, 8, 8).to(dtype),), tester_factory)
        
    def test_pixel_unshuffle_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters (downscale_factor=2)
        # Input: (2, 3, 8, 8) -> Output: (2, 12, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_pixel_unshuffle_downscale_factors(self, tester_factory: Callable) -> None:
        # Test with different downscale factors
        
        # Downscale factor = 2
        # Input: (2, 3, 8, 8) -> Output: (2, 12, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 8, 8),), tester_factory)
        
        # Downscale factor = 3
        # Input: (2, 3, 12, 12) -> Output: (2, 27, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=3), (torch.randn(2, 3, 12, 12),), tester_factory)
        
        # Downscale factor = 4
        # Input: (2, 3, 16, 16) -> Output: (2, 48, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=4), (torch.randn(2, 3, 16, 16),), tester_factory)
        
    def test_pixel_unshuffle_batch_sizes(self, tester_factory: Callable) -> None:
        # Test with different batch sizes
        
        # Batch size = 1
        # Input: (1, 3, 8, 8) -> Output: (1, 12, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(1, 3, 8, 8),), tester_factory)
        
        # Batch size = 4
        # Input: (4, 3, 8, 8) -> Output: (4, 12, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(4, 3, 8, 8),), tester_factory)
        
        # Batch size = 8
        # Input: (8, 3, 8, 8) -> Output: (8, 12, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(8, 3, 8, 8),), tester_factory)
        
    def test_pixel_unshuffle_channels(self, tester_factory: Callable) -> None:
        # Test with different numbers of input channels
        
        # Input channels = 1 (grayscale)
        # Input: (2, 1, 8, 8) -> Output: (2, 4, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 1, 8, 8),), tester_factory)
        
        # Input channels = 3 (RGB)
        # Input: (2, 3, 8, 8) -> Output: (2, 12, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 8, 8),), tester_factory)
        
        # Input channels = 4 (RGBA)
        # Input: (2, 4, 8, 8) -> Output: (2, 16, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 4, 8, 8),), tester_factory)
        
        # Input channels = 16 (multi-channel)
        # Input: (2, 16, 8, 8) -> Output: (2, 64, 4, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 16, 8, 8),), tester_factory)
        
    def test_pixel_unshuffle_input_sizes(self, tester_factory: Callable) -> None:
        # Test with different input sizes
        
        # Small input
        # Input: (2, 3, 4, 4) -> Output: (2, 12, 2, 2)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 4, 4),), tester_factory)
        
        # Medium input
        # Input: (2, 3, 16, 16) -> Output: (2, 12, 8, 8)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 16, 16),), tester_factory)
        
        # Large input
        # Input: (2, 3, 32, 32) -> Output: (2, 12, 16, 16)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 32, 32),), tester_factory)
        
    def test_pixel_unshuffle_non_square(self, tester_factory: Callable) -> None:
        # Test with non-square input
        
        # Input: (2, 3, 8, 16) -> Output: (2, 12, 4, 8)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 8, 16),), tester_factory)
        
        # Input: (2, 3, 16, 8) -> Output: (2, 12, 8, 4)
        self._test_op(PixelUnshuffleModel(downscale_factor=2), (torch.randn(2, 3, 16, 8),), tester_factory)
        
    def test_pixel_unshuffle_odd_dimensions(self, tester_factory: Callable) -> None:
        # Test with dimensions that are not divisible by downscale_factor
        # This should raise an error, but we'll test it to ensure proper error handling
        
        # Input: (2, 3, 7, 8) -> Should raise error (7 is not divisible by 2)
        with self.assertRaises(RuntimeError):
            model = PixelUnshuffleModel(downscale_factor=2)
            model(torch.randn(2, 3, 7, 8))
            
        # Input: (2, 3, 8, 7) -> Should raise error (7 is not divisible by 2)
        with self.assertRaises(RuntimeError):
            model = PixelUnshuffleModel(downscale_factor=2)
            model(torch.randn(2, 3, 8, 7))
