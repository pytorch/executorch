# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import (
    randomize_bn,
    TestXNNPACK,
)

from executorch.backends.xnnpack.test.test_xnnpack_utils_classes import (
    OpSequencesAddConv2d,
)


class TestXNNPACKFloatingPoint(TestXNNPACK):
    # FIXME (T148779166)
    @unittest.expectedFailure
    def test_xnnpack_backend_static_resize_bilinear_2d(self):
        class StaticResizeBilinear2DModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                a = torch.nn.functional.interpolate(
                    x,
                    size=(x.shape[2] * 2, x.shape[3] * 3),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                a = torch.nn.functional.interpolate(
                    a,
                    scale_factor=3.0,
                    mode="bilinear",
                    align_corners=True,
                    antialias=False,
                )
                return a

        example_inputs = (torch.randn(2, 3, 4, 5),)
        # FIXME (T152380622)
        self.assertTrue(False)
        self.lower_and_test_with_partitioner(
            StaticResizeBilinear2DModule(), example_inputs
        )

    # TODO(T171468483)
    @unittest.expectedFailure
    def test_xnnpack_backend_max_dim_vals(self):
        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                max_vals, _ = torch.max(x, dim=3, keepdim=True)
                return max_vals

        model_inputs = (torch.randn(16, 3, 12, 12),)
        max_dim_module = MaxModule()

        self.lower_and_test_with_partitioner(max_dim_module, model_inputs)

    def test_xnnpack_backend_max_dim(self):
        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                x = torch.add(x, x)
                max_values_1, max_indices_1 = torch.max(x, dim=2, keepdim=True)
                max_values_2, max_indices_2 = torch.max(x, dim=3, keepdim=True)
                return (max_values_1, max_indices_1, max_values_2, max_indices_2)

        model_inputs = (torch.randn(16, 3, 12, 12),)
        max_dim_module = MaxModule()

        self.lower_and_test_with_partitioner(max_dim_module, model_inputs)

    def test_xnnpack_backend_hardswish(self):
        # model_inputs = (torch.randn(1, 3, 3),)

        class HardswishModule(torch.nn.Module):
            def __init__(self):
                super(HardswishModule, self).__init__()
                self.hardswish_out_of_place = torch.nn.Hardswish()
                self.hardswish_in_place = torch.nn.Hardswish(inplace=True)
                self.hardswish_functional = torch.nn.functional.hardswish

            def forward(self, x):
                a = self.hardswish_out_of_place(x)
                a = self.hardswish_in_place(a)
                a = self.hardswish_functional(a)
                return a

        # TODO(T158969708)
        # self.lower_and_test_with_partitioner(HardswishModule(), model_inputs)

    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_backend_leaky_relu(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class LeakyReLUModule(torch.nn.Module):
            def __init__(self):
                super(LeakyReLUModule, self).__init__()
                self.leaky_relu_out_of_place = torch.nn.LeakyReLU(negative_slope=0.2)
                self.leaky_relu_in_place = torch.nn.LeakyReLU(
                    negative_slope=0.08, inplace=True
                )
                self.leaky_relu_functional_default = torch.nn.functional.leaky_relu

            def forward(self, x):
                a = self.leaky_relu_out_of_place(x)
                a = self.leaky_relu_in_place(a)
                a = self.leaky_relu_functional_default(a)
                return a

        self.lower_and_test_with_partitioner(LeakyReLUModule(), model_inputs)

    def test_xnnpack_channels_last_tagged_reshape_pass_output(self):
        op_sequences = OpSequencesAddConv2d(2, 2)
        op_sequences.eval()

        example_inputs = (torch.ones(1, 1, 6, 6),)

        self.lower_and_test_with_partitioner(op_sequences, example_inputs)

    # TODO(T171810227)
    @unittest.expectedFailure
    def test_xnnpack_backend_elu(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.elu = torch.nn.ELU()

            def forward(self, x):
                return self.elu(x)

        self.lower_and_test_with_partitioner(ELUModule(), model_inputs)

    # TODO(T158653285)
    @unittest.expectedFailure
    def test_xnnpack_backend_prelu(self):
        num_channels = 5
        model_inputs = (torch.randn(1, num_channels, 3, 2),)

        class PReLUModule(torch.nn.Module):
            def __init__(self):
                super(PReLUModule, self).__init__()
                self.prelu = torch.nn.PReLU()
                self.prelu_non_default = torch.nn.PReLU(
                    num_parameters=num_channels, init=0.2
                )

            def forward(self, x):
                a = self.prelu(x)
                a = self.prelu_non_default(a)
                return a

        self.lower_and_test_with_partitioner(PReLUModule(), model_inputs)

        # Should fail to be partitioned since constraint (input dim) is violated
        self.assertRaises(
            Exception,
            self.lower_and_test_with_partitioner,
            torch.nn.PReLU(),
            (torch.randn(1, 2),),
        )
