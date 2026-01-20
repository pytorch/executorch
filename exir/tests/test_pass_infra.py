# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import ScalarToTensorPass
from executorch.exir.passes.pass_registry import PassRegistry
from torch.export import export
from torch.fx.passes.infra.pass_base import PassBase


class TestPassInfra(unittest.TestCase):
    def test_fail_passbase(self) -> None:
        """
        Tests if we catch errors when we do not inherit PassBase correctly
        """

        # Catches error if we do not implement call()
        class TestPass3(PassBase):
            def __init__(self):
                pass

        with self.assertRaises(TypeError):
            # pyre-ignore
            TestPass3()

    def test_pass_registry_func(self) -> None:
        """
        Test if we register a callable correctly
        """

        # Registering w/o specifying pass_name
        @PassRegistry.register()
        def test_pass1(graph_module: torch.fx.GraphModule) -> None:
            pass

        self.assertEqual(len(PassRegistry.get("test_pass1")), 1)

        # Registering with a specified pass_name
        @PassRegistry.register(pass_name="test_pass1_1")
        def test_pass11(graph_module: torch.fx.GraphModule) -> None:
            pass

        self.assertEqual(len(PassRegistry.get("test_pass1_1")), 1)

    def test_pass_registry_passbase(self) -> None:
        """
        Test if we register a PassBase subclass correctly
        """

        class TestPass2(PassBase):
            def __init__(self) -> None:
                pass

            def call(self, graph_module: torch.fx.GraphModule) -> None:
                pass

        PassRegistry.register("test_pass2")(TestPass2())

        self.assertEqual(len(PassRegistry.get("test_pass2")), 1)

    def test_pass_registry_list(self) -> None:
        def test_pass1(graph_module: torch.fx.GraphModule) -> None:
            pass

        class TestPass2(PassBase):
            def __init__(self) -> None:
                pass

            def call(self, graph_module: torch.fx.GraphModule) -> None:
                pass

        # Register a list of passes
        PassRegistry.register_list(
            pass_name="test_pass3", pass_list=[test_pass1, TestPass2()]
        )
        self.assertEqual(len(PassRegistry.get("test_pass3")), 2)

    def test_pass_manager(self) -> None:
        """
        Tests that the pass manager runs the passes correctly.
        """

        def replace_add_with_mul(gm: torch.fx.GraphModule) -> None:
            for node in gm.graph.nodes:
                if node.op == "call_function" and "aten.add.Tensor" in str(node.target):
                    node.target = torch.mul

        def replace_mul_with_div(gm: torch.fx.GraphModule) -> None:
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target == torch.mul:
                    node.target = torch.div

        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.add(x, x)
                z = torch.add(y, x)
                return z

        f = to_edge(export(F(), (torch.randn(10),))).exported_program().graph_module
        pm = PassManager(passes=[replace_add_with_mul, replace_mul_with_div])
        self.assertEqual(len(pm.passes), 2)
        pm(f)

        # Check that all call_function nodes are divs
        for node in f.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, torch.div)

    def test_pass_manager_invalid_passes(self) -> None:
        """
        Tests that the pass manager detects invalid passes
        """

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        def introduce_call_method(gm: torch.fx.GraphModule) -> None:
            node = list(gm.graph.nodes)[-2]
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method("torch.ops.relu", (torch.randn(2),))
                node.replace_all_uses_with(new_node)

        def introduce_call_module(gm: torch.fx.GraphModule) -> None:
            node = list(gm.graph.nodes)[-2]
            gm.add_submodule("foo", Foo())

            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_module("foo", (torch.randn(2),))
                node.replace_all_uses_with(new_node)

        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.add(x, x)
                z = torch.add(y, x)
                return z

        traced_f1 = (
            to_edge(export(F(), (torch.randn(10),))).exported_program().graph_module
        )
        pm1 = PassManager(
            passes=[introduce_call_method], run_checks_after_each_pass=True
        )

        with self.assertRaisesRegex(Exception, "call_method"):
            pm1(traced_f1)

    def test_pass_metadata(self) -> None:
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        gm = export(F(), (torch.ones(10), torch.ones(10))).graph_module

        pass_result = ScalarToTensorPass()(gm)
        self.assertIsNotNone(pass_result)
        new_gm = pass_result.graph_module

        for node in new_gm.graph.nodes:
            if node.target != "output":
                self.assertIn("val", node.meta)
