# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
from executorch.backends.transforms.fuse_cascaded_transpose_or_permute_ops import (
    FuseCascadedTransposeOrPermuteOps,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.passes.constant_prop_pass import constant_prop_pass


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

    def test_fp32_folded_weight_transpose_is_not_converted_to_linear(self):
        for in_size, out_size, bias in [(4, 8, True), (6, 6, True), (6, 6, False)]:
            torch._dynamo.reset()
            linear = torch.nn.Linear(in_size, out_size, bias=bias)
            tester = Tester(linear, (torch.randn(4, in_size),)).export().to_edge()
            # Folds the weight transpose, so the [add]mm reads a weight that
            # linear would need transposed back.
            constant_prop_pass(tester.get_artifact().exported_program())
            mm = "addmm" if bias else "mm"
            (
                tester.run_passes(self.PassStage)
                .check_count(
                    {f"executorch_exir_dialects_edge__ops_aten_{mm}_default": 1}
                )
                .check_not(["executorch_exir_dialects_edge__ops_aten_linear_default"])
                .partition()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not([f"executorch_exir_dialects_edge__ops_aten_{mm}_default"])
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

    def test_fp32_only_folded_linear_is_not_converted(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(6, 6)

            def forward(self, x, weight):
                # The weight is an input, so its transpose is not folded.
                return torch.nn.functional.linear(self.linear(x), weight)

        inputs = (torch.randn(4, 6), torch.randn(6, 6))
        tester = Tester(Model(), inputs).export().to_edge()
        constant_prop_pass(tester.get_artifact().exported_program())
        (
            tester.run_passes(self.PassStage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten_addmm_default": 1,
                    "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
                }
            )
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_transpose_upstream_of_partition_is_not_converted(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.randn(6, 6))

            def forward(self, x):
                # The transposes around the add are outside the linear.
                return torch.nn.functional.linear(x, (self.a.t() + 1).t())

        x = torch.randn(4, 6)
        tester = Tester(Model(), (x,)).export().to_edge()
        # Cancels the permute pair at the weight, so the mm reads the add
        # directly and the only permute left is upstream of the partition.
        FuseCascadedTransposeOrPermuteOps()(
            tester.get_artifact().exported_program().graph_module
        )
        (
            tester.check_count(
                {"executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1}
            )
            .run_passes(self.PassStage)
            .check_count({"executorch_exir_dialects_edge__ops_aten_mm_default": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_linear_default"])
            .run_method_and_compare_outputs(inputs=(x,))
        )

    def test_fp32_weight_with_non_node_arg_is_not_converted(self):
        class Model(torch.nn.Module):
            def forward(self, x, w1, w2):
                return torch.nn.functional.linear(x, torch.cat([w1, w2]).t())

        inputs = (torch.randn(4, 6), torch.randn(3, 6), torch.randn(3, 6))
        tester = Tester(Model(), inputs).export().to_edge()
        # Cancels the permute pair at the weight, so the mm reads the cat, whose
        # first arg is a list.
        FuseCascadedTransposeOrPermuteOps()(
            tester.get_artifact().exported_program().graph_module
        )
        (
            tester.run_passes(self.PassStage)
            .check_count({"executorch_exir_dialects_edge__ops_aten_mm_default": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_linear_default"])
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_bmm_in_linear_partition(self):
        class Model(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(8, 6, bias=bias)

            def forward(self, x):
                # A non-contiguous input makes the linear decompose to bmm.
                return self.linear(x.transpose(1, 2))

        x = torch.randn(3, 8, 4)
        for bias, fold in itertools.product([False, True], [False, True]):
            torch._dynamo.reset()
            tester = Tester(Model(bias), (x,)).export().to_edge()
            if fold:
                constant_prop_pass(tester.get_artifact().exported_program())
            tester.run_passes(self.PassStage)
            # The bias add of a bmm is not found, so it stays a bmm.
            if fold or bias:
                tester.check_count(
                    {"executorch_exir_dialects_edge__ops_aten_bmm_default": 1}
                ).check_not(["executorch_exir_dialects_edge__ops_aten_linear_default"])
            else:
                tester.check_count(
                    {"executorch_exir_dialects_edge__ops_aten_linear_default": 1}
                ).check_not(["executorch_exir_dialects_edge__ops_aten_bmm_default"])
            tester.run_method_and_compare_outputs(inputs=(x,))

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
