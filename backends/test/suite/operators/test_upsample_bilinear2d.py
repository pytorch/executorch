# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple, Union

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


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
        return torch.nn.functional.interpolate(
            x, size=self.size, mode="bilinear", align_corners=self.align_corners
        )


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
        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=self.align_corners,
        )


@parameterize_by_dtype
def test_upsample_bilinear2d_dtype(test_runner, dtype) -> None:
    # Input shape: (batch_size, channels, height, width)
    model = ModelWithSize(size=(10, 10), align_corners=False).to(dtype)
    test_runner.lower_and_run_model(model, (torch.rand(2, 3, 5, 5).to(dtype),))


def test_upsample_bilinear2d_sizes(test_runner) -> None:
    # Test with different input and output sizes

    # Small input, larger output
    test_runner.lower_and_run_model(
        ModelWithSize(size=(8, 8), align_corners=False),
        (torch.randn(1, 2, 4, 4),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(8, 8), align_corners=True),
        (torch.randn(1, 2, 4, 4),),
    )

    # Larger input, even larger output
    test_runner.lower_and_run_model(
        ModelWithSize(size=(16, 16), align_corners=False),
        (torch.randn(1, 2, 8, 8),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(16, 16), align_corners=True),
        (torch.randn(1, 2, 8, 8),),
    )

    # Different height and width
    test_runner.lower_and_run_model(
        ModelWithSize(size=(16, 8), align_corners=False),
        (torch.randn(1, 2, 8, 4),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(16, 8), align_corners=True),
        (torch.randn(1, 2, 8, 4),),
    )

    # Asymmetric upsampling
    test_runner.lower_and_run_model(
        ModelWithSize(size=(20, 10), align_corners=False),
        (torch.randn(1, 2, 5, 5),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(20, 10), align_corners=True),
        (torch.randn(1, 2, 5, 5),),
    )


def test_upsample_bilinear2d_scale_factors(test_runner) -> None:
    # Test with different scale factors

    # Scale by 2
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=2.0, align_corners=False),
        (torch.randn(1, 2, 5, 5),),
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=2.0, align_corners=True),
        (torch.randn(1, 2, 5, 5),),
    )

    # Scale by 3
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=3.0, align_corners=False),
        (torch.randn(1, 2, 5, 5),),
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=3.0, align_corners=True),
        (torch.randn(1, 2, 5, 5),),
    )

    # Scale by 1.5
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=1.5, align_corners=False),
        (torch.randn(1, 2, 6, 6),),
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=1.5, align_corners=True),
        (torch.randn(1, 2, 6, 6),),
    )

    # Different scales for height and width
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=(2.0, 1.5), align_corners=False),
        (torch.randn(1, 2, 5, 6),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=(2.0, 1.5), align_corners=True),
        (torch.randn(1, 2, 5, 6),),
        generate_random_test_inputs=False,
    )


def test_upsample_bilinear2d_batch_sizes(test_runner) -> None:
    # Test with different batch sizes
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(1, 3, 5, 5),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(4, 3, 5, 5),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(8, 3, 5, 5),),
    )


def test_upsample_bilinear2d_channels(test_runner) -> None:
    # Test with different numbers of channels
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(2, 1, 5, 5),),
    )  # Grayscale
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(2, 3, 5, 5),),
    )  # RGB
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(2, 4, 5, 5),),
    )  # RGBA
    test_runner.lower_and_run_model(
        ModelWithSize(size=(10, 10), align_corners=False),
        (torch.randn(2, 16, 5, 5),),
    )  # Multi-channel


def test_upsample_bilinear2d_same_size(test_runner) -> None:
    # Test with output size same as input size (should be identity)
    test_runner.lower_and_run_model(
        ModelWithSize(size=(5, 5), align_corners=False),
        (torch.randn(2, 3, 5, 5),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(5, 5), align_corners=True),
        (torch.randn(2, 3, 5, 5),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=1.0, align_corners=False),
        (torch.randn(2, 3, 5, 5),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=1.0, align_corners=True),
        (torch.randn(2, 3, 5, 5),),
        generate_random_test_inputs=False,
    )


def test_upsample_bilinear2d_downsampling(test_runner) -> None:
    # Test downsampling
    test_runner.lower_and_run_model(
        ModelWithSize(size=(4, 4), align_corners=False),
        (torch.randn(2, 3, 8, 8),),
    )
    test_runner.lower_and_run_model(
        ModelWithSize(size=(4, 4), align_corners=True),
        (torch.randn(2, 3, 8, 8),),
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=0.5, align_corners=False),
        (torch.randn(2, 3, 8, 8),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=0.5, align_corners=True),
        (torch.randn(2, 3, 8, 8),),
        generate_random_test_inputs=False,
    )

    # Test with non-integer downsampling factor
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=0.75, align_corners=False),
        (torch.randn(2, 3, 8, 8),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        ModelWithScale(scale_factor=0.75, align_corners=True),
        (torch.randn(2, 3, 8, 8),),
        generate_random_test_inputs=False,
    )
