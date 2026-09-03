# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export


class ConvModule(torch.nn.Module):
    def __init__(self, channels: int, groups: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestChannelsLastDepthwiseConv(unittest.TestCase):
    """A channels-last weight with a size-1 input-channel dim must stay canonical.

    Such a weight -- any depthwise convolution, and any single-channel one --
    has two dimensions tied at stride 1, so ordering by stride alone reports
    ``(0, 2, 1, 3)``. That is neither contiguous nor channels-last, and the
    portable convolution kernel rejects it:

        Expected tensor to have default or channels last dim order, but got
        dim_order(0): 0 dim_order(1): 2 dim_order(2): 1 dim_order(3): 3

    See https://github.com/pytorch/executorch/issues/22520.
    """

    def _assert_runtime_matches_eager(
        self, module: torch.nn.Module, sample_input: torch.Tensor
    ) -> None:
        program = to_edge_transform_and_lower(
            export(module, (sample_input,), strict=True),
            partitioner=[],
        ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

        runtime_output = _load_for_executorch_from_buffer(program.buffer).run_method(
            "forward", (sample_input,)
        )[0]
        with torch.no_grad():
            expected = module(sample_input)
        torch.testing.assert_close(runtime_output, expected, atol=1e-5, rtol=1e-5)

    def test_channels_last_depthwise_conv_runs(self) -> None:
        module = (
            ConvModule(channels=8, groups=8, kernel_size=3)
            .eval()
            .to(memory_format=torch.channels_last)
        )
        sample_input = torch.randn(1, 8, 8, 8).to(memory_format=torch.channels_last)
        self._assert_runtime_matches_eager(module, sample_input)

    def test_channels_last_single_channel_conv_runs(self) -> None:
        # Not depthwise, but the weight is still (C_out, 1, kH, kW).
        module = (
            ConvModule(channels=1, groups=1, kernel_size=3)
            .eval()
            .to(memory_format=torch.channels_last)
        )
        sample_input = torch.randn(1, 1, 8, 8).to(memory_format=torch.channels_last)
        self._assert_runtime_matches_eager(module, sample_input)

    def test_conv_layout_matrix_runs(self) -> None:
        for channels, depthwise, kernel_size, channels_last in itertools.product(
            (1, 4, 8), (False, True), (1, 3), (False, True)
        ):
            groups = channels if depthwise else 1
            with self.subTest(
                channels=channels,
                groups=groups,
                kernel_size=kernel_size,
                channels_last=channels_last,
            ):
                module = ConvModule(channels, groups, kernel_size).eval()
                sample_input = torch.randn(1, channels, 8, 8)
                if channels_last:
                    module = module.to(memory_format=torch.channels_last)
                    sample_input = sample_input.to(memory_format=torch.channels_last)
                self._assert_runtime_matches_eager(module, sample_input)


if __name__ == "__main__":
    unittest.main()
