# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from .custom_ops_defs import conv_with_clamp_op  # noqa


class TestCustomOps(unittest.TestCase):
    def test_conv_with_clamp(self):
        class ConvWithClamp(torch.nn.Module):
            def __init__(
                self,
                weight,
                bias,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
                output_min,
                output_max,
            ):
                super().__init__()
                self.weight = weight
                self.bias = bias
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.transposed = transposed
                self.output_padding = output_padding
                self.groups = groups
                self.output_min = output_min
                self.output_max = output_max

            def forward(self, x):
                return torch.ops.et_vk.conv_with_clamp(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.transposed,
                    self.output_padding,
                    self.groups,
                    self.output_min,
                    self.output_max,
                )

        model = ConvWithClamp(
            weight=torch.randn(64, 64, 3, 3),
            bias=torch.randn(64),
            stride=[1],
            padding=[0],
            dilation=[1],
            transposed=False,
            output_padding=[0],
            groups=1,
            output_min=0,
            output_max=float("inf"),
        )
        x = torch.randn(2, 64, 10, 10)
        custom_out = model(x)

        expected_out = torch.clamp(
            torch.convolution(
                x,
                model.weight,
                model.bias,
                model.stride,
                model.padding,
                model.dilation,
                model.transposed,
                model.output_padding,
                model.groups,
            ),
            min=model.output_min,
            max=model.output_max,
        )

        self.assertEqual(
            custom_out.shape,
            expected_out.shape,
            "custom op `conv_with_clamp` output shape matches expected",
        )
        self.assertTrue(torch.allclose(custom_out, expected_out))

    def test_grid_priors(self):
        class GridPriors(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, stride, offset):
                return torch.ops.et_vk.grid_priors(x, stride, offset)

        model = GridPriors()
        sample_input = (torch.rand(2, 5, 2, 3), 4, 0.5)
        custom_out = model(*sample_input)

        def calculate_expected_output(x, stride, offset):
            height, width = x.shape[-2:]
            shift_x = (torch.arange(0, width) + offset) * stride
            shift_y = (torch.arange(0, height) + offset) * stride
            shift_xx, shift_yy = torch.meshgrid(shift_y, shift_x)
            shift_xx = shift_xx.reshape(-1)
            shift_yy = shift_yy.reshape(-1)
            shifts = torch.stack((shift_yy, shift_xx), dim=-1)
            return shifts

        expected_out = calculate_expected_output(*sample_input)

        self.assertEqual(
            custom_out.shape,
            expected_out.shape,
            "custom op `grid_priors` output shape matches expected",
        )
        self.assertTrue(torch.allclose(custom_out, expected_out))
