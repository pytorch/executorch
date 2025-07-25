# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Callable, Optional, Tuple, Union

import torch

from executorch.backends.test.compliance_suite import (
    dtype_test,
    operator_test,
    OperatorTest,
)

class ModelWithSize(torch.nn.Module):
    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        align_corners: Optional[bool] = None,
    ):
        super().__init__()
        self.size = size
        self.align_corners = align_corners
        
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=self.align_corners)

class ModelWithScale(torch.nn.Module):
    def __init__(
        self,
        scale_factor: Union[float, Tuple[float, float]] = 2.0,
        align_corners: Optional[bool] = None,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners)

@operator_test
class TestUpsampleBilinear2d(OperatorTest):
    @dtype_test
    def test_upsample_bilinear2d_dtype(self, dtype, tester_factory: Callable) -> None:
        # Input shape: (batch_size, channels, height, width)
        model = ModelWithSize(size=(10, 10), align_corners=False).to(dtype)
        self._test_op(model, (torch.rand(2, 3, 5, 5).to(dtype),), tester_factory)
        
    def test_upsample_bilinear2d_basic(self, tester_factory: Callable) -> None:
        # Basic test with default parameters
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(10, 10), align_corners=True), (torch.randn(2, 3, 5, 5),), tester_factory)
        
    def test_upsample_bilinear2d_sizes(self, tester_factory: Callable) -> None:
        # Test with different input and output sizes
        
        # Small input, larger output
        self._test_op(ModelWithSize(size=(8, 8), align_corners=False), (torch.randn(1, 2, 4, 4),), tester_factory)
        self._test_op(ModelWithSize(size=(8, 8), align_corners=True), (torch.randn(1, 2, 4, 4),), tester_factory)
        
        # Larger input, even larger output
        self._test_op(ModelWithSize(size=(16, 16), align_corners=False), (torch.randn(1, 2, 8, 8),), tester_factory)
        self._test_op(ModelWithSize(size=(16, 16), align_corners=True), (torch.randn(1, 2, 8, 8),), tester_factory)
        
        # Different height and width
        self._test_op(ModelWithSize(size=(16, 8), align_corners=False), (torch.randn(1, 2, 8, 4),), tester_factory)
        self._test_op(ModelWithSize(size=(16, 8), align_corners=True), (torch.randn(1, 2, 8, 4),), tester_factory)
        
        # Asymmetric upsampling
        self._test_op(ModelWithSize(size=(20, 10), align_corners=False), (torch.randn(1, 2, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(20, 10), align_corners=True), (torch.randn(1, 2, 5, 5),), tester_factory)
        
    def test_upsample_bilinear2d_scale_factors(self, tester_factory: Callable) -> None:
        # Test with different scale factors
        
        # Scale by 2
        self._test_op(ModelWithScale(scale_factor=2.0, align_corners=False), (torch.randn(1, 2, 5, 5),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=2.0, align_corners=True), (torch.randn(1, 2, 5, 5),), tester_factory)
        
        # Scale by 3
        self._test_op(ModelWithScale(scale_factor=3.0, align_corners=False), (torch.randn(1, 2, 5, 5),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=3.0, align_corners=True), (torch.randn(1, 2, 5, 5),), tester_factory)
        
        # Scale by 1.5
        self._test_op(ModelWithScale(scale_factor=1.5, align_corners=False), (torch.randn(1, 2, 6, 6),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=1.5, align_corners=True), (torch.randn(1, 2, 6, 6),), tester_factory)
        
        # Different scales for height and width
        self._test_op(ModelWithScale(scale_factor=(2.0, 1.5), align_corners=False), (torch.randn(1, 2, 5, 6),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=(2.0, 1.5), align_corners=True), (torch.randn(1, 2, 5, 6),), tester_factory)
        
    def test_upsample_bilinear2d_batch_sizes(self, tester_factory: Callable) -> None:
        # Test with different batch sizes
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(1, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(4, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(8, 3, 5, 5),), tester_factory)
        
    def test_upsample_bilinear2d_channels(self, tester_factory: Callable) -> None:
        # Test with different numbers of channels
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(2, 1, 5, 5),), tester_factory)  # Grayscale
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(2, 3, 5, 5),), tester_factory)  # RGB
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(2, 4, 5, 5),), tester_factory)  # RGBA
        self._test_op(ModelWithSize(size=(10, 10), align_corners=False), (torch.randn(2, 16, 5, 5),), tester_factory)  # Multi-channel
        
    def test_upsample_bilinear2d_same_size(self, tester_factory: Callable) -> None:
        # Test with output size same as input size (should be identity)
        self._test_op(ModelWithSize(size=(5, 5), align_corners=False), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(5, 5), align_corners=True), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=1.0, align_corners=False), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=1.0, align_corners=True), (torch.randn(2, 3, 5, 5),), tester_factory)
        
    def test_upsample_bilinear2d_downsampling(self, tester_factory: Callable) -> None:
        # Test downsampling
        self._test_op(ModelWithSize(size=(4, 4), align_corners=False), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(ModelWithSize(size=(4, 4), align_corners=True), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=0.5, align_corners=False), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=0.5, align_corners=True), (torch.randn(2, 3, 8, 8),), tester_factory)
        
        # Test with non-integer downsampling factor
        self._test_op(ModelWithScale(scale_factor=0.75, align_corners=False), (torch.randn(2, 3, 8, 8),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=0.75, align_corners=True), (torch.randn(2, 3, 8, 8),), tester_factory)
        
    def test_upsample_bilinear2d_large_scale(self, tester_factory: Callable) -> None:
        # Test with large scale factor
        self._test_op(ModelWithScale(scale_factor=4.0, align_corners=False), (torch.randn(1, 2, 4, 4),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=4.0, align_corners=True), (torch.randn(1, 2, 4, 4),), tester_factory)
        
    def test_upsample_bilinear2d_non_square(self, tester_factory: Callable) -> None:
        # Test with non-square input
        self._test_op(ModelWithSize(size=(10, 20), align_corners=False), (torch.randn(2, 3, 5, 10),), tester_factory)
        self._test_op(ModelWithSize(size=(10, 20), align_corners=True), (torch.randn(2, 3, 5, 10),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=2.0, align_corners=False), (torch.randn(2, 3, 5, 10),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=2.0, align_corners=True), (torch.randn(2, 3, 5, 10),), tester_factory)
        
    def test_upsample_bilinear2d_odd_sizes(self, tester_factory: Callable) -> None:
        # Test with odd input and output sizes (where interpolation behavior might be more noticeable)
        self._test_op(ModelWithSize(size=(9, 9), align_corners=False), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(9, 9), align_corners=True), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithSize(size=(7, 7), align_corners=False), (torch.randn(2, 3, 3, 3),), tester_factory)
        self._test_op(ModelWithSize(size=(7, 7), align_corners=True), (torch.randn(2, 3, 3, 3),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=1.5, align_corners=False), (torch.randn(2, 3, 5, 5),), tester_factory)
        self._test_op(ModelWithScale(scale_factor=1.5, align_corners=True), (torch.randn(2, 3, 5, 5),), tester_factory)
