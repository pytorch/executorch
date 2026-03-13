# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from torch._higher_order_ops.map import map as torch_map
from torch._higher_order_ops.scan import scan


class TestHigherOrderOps(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_cond(self):
        """
        Test that torch.cond with add/sub branches can be lowered to XNNPACK.

        The model returns x + y if x[0] > 0, else x - y.
        Verifies that add and sub ops are delegated to XNNPACK (not present
        as undelegated operators in the executorch program).
        """

        class CondModel(torch.nn.Module):
            def true_fn(self, x, y):
                return x + y

            def false_fn(self, x, y):
                return x - y

            def forward(self, x, y):
                return torch.cond(x[0] > 0, self.true_fn, self.false_fn, [x, y])

        model = CondModel()
        inputs = (torch.randn(4), torch.randn(4))

        tester = (
            Tester(model, inputs).export().to_edge_transform_and_lower().to_executorch()
        )

        # Get the executorch program
        program = tester.get_artifact()._emitter_output.program

        # Check that add and sub are not in the operators list (they should be delegated)
        operator_names = [
            op.name for plan in program.execution_plan for op in plan.operators
        ]

        self.assertNotIn(
            "aten::add",
            operator_names,
            "add op should be delegated",
        )
        self.assertNotIn(
            "aten::sub",
            operator_names,
            "sub op should be delegated",
        )

        # Verify there are XNNPACK delegates
        delegates = [d for plan in program.execution_plan for d in plan.delegates]
        xnnpack_delegates = [d for d in delegates if d.id == "XnnpackBackend"]
        self.assertEqual(
            len(xnnpack_delegates),
            2,
            "Expected 2 XNNPACK delegates (one for each branch)",
        )

        # Verify execution produces correct results
        tester.serialize().run_method_and_compare_outputs()

    def test_cond_with_linear(self):
        """
        Test that torch.cond with a linear module in one branch can be lowered.

        The model returns linear(x) if x[0] > 0, else x * 2.
        This test verifies that lowering and execution succeed.
        """

        class CondLinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def true_fn(self, x):
                return self.linear(x)

            def false_fn(self, x):
                return x * 2

            def forward(self, x):
                return torch.cond(x[0] > 0, self.true_fn, self.false_fn, [x])

        model = CondLinearModel()
        inputs = (torch.randn(4),)

        tester = Tester(model, inputs).export().to_edge_transform_and_lower()

        (tester.to_executorch().serialize().run_method_and_compare_outputs())

    def test_map(self):
        """
        Test that torch.map with add operation can be lowered to XNNPACK.

        Maps a function that adds y to each element of xs.
        Verifies that add ops are delegated to XNNPACK.
        """

        class MapModel(torch.nn.Module):
            def forward(self, xs, y):
                def f(x, y):
                    return x + y

                return torch_map(f, xs, y)

        model = MapModel()
        inputs = (torch.randn(3, 4), torch.randn(4))

        tester = (
            Tester(model, inputs).export().to_edge_transform_and_lower().to_executorch()
        )

        # Get the executorch program (before serialize)
        program = tester.get_artifact()._emitter_output.program

        # Check that add is not in the operators list (it should be delegated)
        operator_names = [
            op.name for plan in program.execution_plan for op in plan.operators
        ]

        self.assertNotIn(
            "aten::add",
            operator_names,
            "add op should be delegated",
        )

        # Verify execution produces correct results
        tester.serialize().run_method_and_compare_outputs()

    def test_scan(self):
        """
        Test that torch.scan (cumulative sum) can be lowered to XNNPACK.

        Performs a cumulative sum over the input tensor.
        Verifies that add ops inside scan are delegated to XNNPACK.
        """

        class ScanModel(torch.nn.Module):
            def forward(self, xs):
                def combine_fn(carry, x):
                    new_carry = carry + x
                    return new_carry, new_carry + 0

                init = torch.zeros_like(xs[0])
                return scan(combine_fn, init, xs)

        model = ScanModel()
        inputs = (torch.randn(5, 4),)

        tester = (
            Tester(model, inputs).export().to_edge_transform_and_lower().to_executorch()
        )

        # Get the executorch program (before serialize)
        program = tester.get_artifact()._emitter_output.program

        # Check that add is not in the operators list (it should be delegated)
        operator_names = [
            op.name for plan in program.execution_plan for op in plan.operators
        ]

        self.assertNotIn(
            "aten::add",
            operator_names,
            "add op should be delegated",
        )

        # Verify execution produces correct results
        tester.serialize().run_method_and_compare_outputs()


if __name__ == "__main__":
    unittest.main()
