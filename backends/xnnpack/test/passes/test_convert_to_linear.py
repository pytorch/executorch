# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


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

    def test_multimethod_program_without_a_forward_method(self):
        # A multimethod export has no "forward". Each method's pass has to be
        # bound to that method's own exported program.
        class Linear(torch.nn.Module):
            def __init__(self, out_features: int):
                super().__init__()
                self.fc = torch.nn.Linear(8, out_features)

            def forward(self, x):
                return self.fc(x)

        inputs = (torch.randn(2, 8),)
        programs = {
            name: torch.export.export(Linear(out).eval(), inputs, strict=True)
            for name, out in (("base_forward", 4), ("lora_forward", 6))
        }
        # aten.linear is not a core ATen op, so the verifier has to be told to
        # accept what this pass deliberately produces.
        config = exir.EdgeCompileConfig(
            _core_aten_ops_exception_list=[torch.ops.aten.linear.default]
        )
        edge = exir.to_edge(programs, compile_config=config)

        edge = edge.transform(
            {
                name: [ConvertToLinearPass(edge.exported_program(name))]
                for name in edge.methods
            },
            compile_config=config,
        )

        for name in ("base_forward", "lora_forward"):
            graph = edge.exported_program(name).graph
            targets = [n.target for n in graph.nodes if n.op == "call_function"]
            self.assertIn(exir_ops.edge.aten.linear.default, targets)
            self.assertNotIn(exir_ops.edge.aten.addmm.default, targets)

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
