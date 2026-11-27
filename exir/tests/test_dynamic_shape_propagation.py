# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from unittest import TestCase

import torch

from executorch import exir
from executorch.exir import to_edge
from executorch.exir.passes import (
    DebugPass,
    ExportPass,
    HintBasedSymShapeEvalPass,
    SpecPropPass,
)
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.tests.models import Repeat, TensorItem
from torch.export import export


class TestDynamicShapeProp(TestCase):
    def test_repeat(self):
        eager_model = Repeat()
        inputs = eager_model.get_random_inputs()
        inputs = inputs[0], inputs[1]

        prog = to_edge(
            export(
                eager_model,
                inputs,
                dynamic_shapes=eager_model.get_dynamic_shape(),
                strict=True,
            ),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )

        new_prog = prog.transform([SpecPropPass(), HintBasedSymShapeEvalPass()])

        gm = new_prog.exported_program().graph_module

        DebugPass(show_spec=True)(gm)
        *_, return_node = gm.graph.nodes
        speclist = return_node.meta["spec"]
        self.assertEqual(len(speclist), 2)
        first_spec, second_spec = speclist

        self.assertTrue(first_spec.is_upper_bound_tensor)
        self.assertTrue(second_spec.is_upper_bound_tensor)
        self.assertEqual(first_spec.shape, [4, 5])


class TestUnbackedSymInt(TestCase):
    def test_unbacked_symint(self):
        eager_model = TensorItem()
        inputs = eager_model.get_random_inputs()
        inputs = inputs[0], inputs[1]

        prog = to_edge(
            export(eager_model, inputs, dynamic_shapes=None, strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        new_prog = prog.transform([SpecPropPass(), HintBasedSymShapeEvalPass()])
        gm = new_prog.exported_program().graph_module

        DebugPass(show_spec=True)(gm)
        *_, return_node = gm.graph.nodes
        speclist = return_node.meta["spec"]
        self.assertEqual(len(speclist), 1)
        self.assertTrue(speclist[0].is_upper_bound_tensor)
        self.assertEqual(
            speclist[0].shape, [100, 100]
        )  # upper bound of TensorItem model


class TestSymIntViewArgs(TestCase):
    class Conv1dToConv2d(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Use view to make sure edge view handle symint shapes correctly.
            # input = input.view(input.size(0), input.size(1), input.size(2), 1)  # (N, C, H, W)
            # weight = torch.randn(1, 16, 3, 1)  # (out_channels, in_channels, kH, kW)
            # return torch.nn.functional.conv2d(input, weight)

            return torch.nn.functional.conv1d(
                input, torch.randn(1, 16, 3)
            )  # (out_channels, in_channels, kW)

        def get_random_inputs(self) -> tuple[torch.Tensor]:
            return (torch.randn(1, 16, 50),)  # (batch_size, channels, width)

        def get_dynamic_shape(self) -> tuple[dict[int, torch.export.Dim]]:
            dim = torch.export.Dim("width", min=10, max=100)
            return ({2: dim},)

    def test_symint_viewargs(self):
        eager_model = TestSymIntViewArgs.Conv1dToConv2d()
        inputs = eager_model.get_random_inputs()

        class TestViewCopyPass(ExportPass):
            def call_operator(self, op, args, kwargs, meta):
                from executorch.exir.dialects._ops import ops as exir_ops

                if op != exir_ops.edge.aten.convolution.default:
                    return super().call_operator(op, args, kwargs, meta)

                x = args[0]
                x = super().call_operator(
                    exir_ops.edge.aten.view_copy.default,
                    (x, self.call_size_operator_all(x, meta) + [1]),
                    {},
                    meta,
                )

                w = args[1]
                w = super().call_operator(
                    exir_ops.edge.aten.view_copy.default,
                    (w, self.call_size_operator_all(w, meta) + [1]),
                    {},
                    meta,
                )

                new_args = (
                    x,
                    w,
                    args[2],
                    args[3] + [1],  # stride
                    args[4] + [0],  # padding
                    args[5] + [1],  # dilation
                    args[6],
                    args[7] + [0],
                    args[8],
                )
                x = super().call_operator(
                    exir_ops.edge.aten.convolution.default, new_args, kwargs, meta
                )
                x = super().call_operator(
                    exir_ops.edge.aten.view_copy.default,
                    (x, self.call_size_operator_all(x, meta)[:-1]),
                    {},
                    meta,
                )

                return x

        prog = to_edge(
            export(
                eager_model,
                inputs,
                dynamic_shapes=eager_model.get_dynamic_shape(),
                strict=True,
            ),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        new_prog = prog.transform(
            [SpecPropPass(), ConstraintBasedSymShapeEvalPass(), TestViewCopyPass()]
        )
        gm = new_prog.exported_program().graph_module
        DebugPass(show_spec=True)(gm)
        *_, return_node = gm.graph.nodes
        speclist = return_node.meta["spec"]

        self.assertEqual(len(speclist), 1)
        out_spec = speclist[0]
        self.assertTrue(out_spec.is_upper_bound_tensor)
        self.assertEqual(out_spec.shape, [1, 1, 98])


class TestDimConstraints(TestCase):
    """
    ConstraintBasedSymShapeEvalPass keeps the constraints of dynamic inputs
    besides their upper bounds, and they reach the input's ExtraTensorInfo.
    """

    class _AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    def _input_tensors(self, example, dynamic_shapes):
        program = to_edge(
            export(self._AddOne(), (example,), dynamic_shapes=dynamic_shapes)
        ).to_executorch()
        plan = program.executorch_program.execution_plan[0]
        values = plan.values
        inputs = [values[i].val for i in plan.inputs]
        others = [
            v.val
            for i, v in enumerate(values)
            if i not in plan.inputs and isinstance(v.val, exir.schema.Tensor)
        ]
        return inputs, others

    def test_lower_bound_and_divisor(self):
        h = torch.export.Dim("h", min=3, max=8)
        w = torch.export.Dim("w", min=5, max=10)
        (x,), others = self._input_tensors(torch.randn(16, 6), {"x": {0: 4 * h, 1: w}})
        self.assertEqual(x.sizes, [32, 10])
        self.assertEqual(
            x.extra_tensor_info.dim_constraints,
            [
                exir.schema.DimConstraint(min=12, divisor=4, remainder=0),
                exir.schema.DimConstraint(min=5, divisor=1, remainder=0),
            ],
        )
        # Only the method's inputs carry the constraints.
        for tensor in others:
            if tensor.extra_tensor_info is not None:
                self.assertIsNone(tensor.extra_tensor_info.dim_constraints)

    def test_offset_gives_remainder(self):
        # torch.export raises a minimum of 1 to 2, so the minimum that is kept
        # is the one 2 * k + 1 has with k unbounded below, not 2 * 2 + 1.
        k = torch.export.Dim("k", min=1, max=10)
        (x,), _ = self._input_tensors(torch.randn(7), {"x": {0: 2 * k + 1}})
        self.assertEqual(
            x.extra_tensor_info.dim_constraints,
            [exir.schema.DimConstraint(min=1, divisor=2, remainder=1)],
        )

    def test_default_minimum_is_not_serialized(self):
        # A Dim with the default or a small minimum has no constraint besides
        # its upper bound, so nothing is added to the program, and sizes under
        # the minimum torch.export traced with (such as a batch of 1) keep
        # working.
        for dim in (
            torch.export.Dim("b", max=8),
            torch.export.Dim("b", min=2, max=8),
        ):
            (x,), _ = self._input_tensors(torch.randn(4, 3), {"x": {0: dim}})
            self.assertEqual(x.sizes, [8, 3])
            self.assertIsNone(x.extra_tensor_info)
