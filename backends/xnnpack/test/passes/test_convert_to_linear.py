# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestConvertToLinear(unittest.TestCase):
    PassStage = RunPasses([ConvertToLinearPass])

    def setUp(self):
        torch._dynamo.reset()

    def test_fp32_convert_to_linear(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        bias_vals = [True, True, False]

        for i, _ in enumerate(in_sizes):
            torch._dynamo.reset()
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            linear = torch.nn.Linear(input_size, output_size, bias=bias_vals[i])
            inputs = (torch.randn(in_size, input_size),)

            (
                Tester(linear, inputs)
                .export()
                .to_edge()
                .run_passes(self.PassStage)
                .check_count(
                    {"executorch_exir_dialects_edge__ops_aten_linear_default": 1}
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_view_after_two_linears_with_dynamic_shapes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(16, 16)
                self.fc2 = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.fc2(self.fc1(x)).view(-1, 8)

        dynamic_shapes = ({0: torch.export.Dim("batch", min=1, max=32)},)
        (
            Tester(Model(), (torch.randn(4, 16),), dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count({"executorch_exir_dialects_edge__ops_aten_linear_default": 2})
            # The view does not change the shape, so it goes.
            .check_not(["executorch_exir_dialects_edge__ops_aten_view_copy_default"])
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_to_another_shape_with_equal_bounds_is_kept(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x).view(x.shape[1], x.shape[0], 8)

        x = torch.randn(2, 5, 16)
        dynamic_shapes = (
            {
                0: torch.export.Dim("a", min=2, max=8),
                1: torch.export.Dim("b", min=2, max=8),
            },
        )
        (
            Tester(Model(), (x,), dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
                    # [b, a, 8] has the bounds of the linear output [a, b, 8], but
                    # not its shape, so the view stays.
                    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
                }
            )
            .run_method_and_compare_outputs(inputs=(x,))
        )

    def test_fp32_addmm_with_alpha_or_beta(self):
        class TransposedWeightAddmm(torch.nn.Module):
            def __init__(self, alpha, beta):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 8))
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.alpha, self.beta = alpha, beta

            def forward(self, x):
                return torch.addmm(
                    self.bias, x, self.weight.t(), alpha=self.alpha, beta=self.beta
                )

        # linear has no alpha or beta, so only the default addmm may become one.
        for alpha, beta, op in [
            (1, 1, "linear"),
            (2, 1, "addmm"),
            (1, 3, "addmm"),
            (1, 0, "addmm"),
        ]:
            with self.subTest(alpha=alpha, beta=beta):
                (
                    Tester(TransposedWeightAddmm(alpha, beta), (torch.randn(2, 8),))
                    .export()
                    .to_edge()
                    .run_passes(self.PassStage)
                    .check_count(
                        {f"executorch_exir_dialects_edge__ops_aten_{op}_default": 1}
                    )
                    .run_method_and_compare_outputs()
                )
