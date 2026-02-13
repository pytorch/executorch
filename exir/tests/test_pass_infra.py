# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import executorch.exir as exir
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.pass_manager import ExportedProgramPassManager, PassManager
from executorch.exir.passes import ScalarToTensorPass
from executorch.exir.passes.pass_registry import PassRegistry
from executorch.exir.program import to_edge
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, InputSpec, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult


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

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            z = torch.add(y, x)
            return z

        f = (
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )
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

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            z = torch.add(y, x)
            return z

        traced_f1 = (
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )
        pm1 = PassManager(
            passes=[introduce_call_method], run_checks_after_each_pass=True
        )

        with self.assertRaisesRegex(Exception, "call_method"):
            pm1(traced_f1)

    def test_pass_metadata(self) -> None:
        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        sample_inputs = (torch.randn(1, 3), torch.randn(1, 3))
        gm = exir.capture(
            f, sample_inputs, exir.CaptureConfig()
        ).exported_program.graph_module

        pass_result = ScalarToTensorPass()(gm)
        self.assertIsNotNone(pass_result)
        new_gm = pass_result.graph_module

        for node in new_gm.graph.nodes:
            if node.target != "output":
                self.assertIn("val", node.meta)


class TestExportedProgramPassManager(unittest.TestCase):
    def test_raises_spec_violation_error(self) -> None:
        """
        Ensures that ExportedProgramPassManager raises a SpecViolationError after running
        a pass which places a non-Edge operator in the graph.
        """
        def replace_add_with_torch_aten_mul(gm: torch.fx.GraphModule) -> PassResult:
            modified = False
            for node in gm.graph.find_nodes(op="call_function", target=exir_ops.edge.aten.add.Tensor):
                node.target = torch.ops.aten.mul.Tensor
                modified = True
            return PassResult(gm, modified)

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            z = torch.add(y, x)
            return z

        exported_program = (
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program
        )

        pm = ExportedProgramPassManager(passes=[replace_add_with_torch_aten_mul])
        with self.assertRaisesRegex(torch._export.verifier.SpecViolationError, r"Operator torch._ops.aten.mul.Tensor is not an Edge operator."):
            pm(exported_program)

    def test_runs_graph_module_passes_on_exported_program(self) -> None:
        """
        Tests that ExportedProgramPassManager runs GraphModule passes
        on an ExportedProgram and the graph is correctly modified.
        """

        def replace_add_with_mul(gm: torch.fx.GraphModule) -> PassResult:
            modified = False
            for node in gm.graph.find_nodes(op="call_function", target=exir_ops.edge.aten.add.Tensor):
                node.target = exir_ops.edge.aten.mul.Tensor
                modified = True
            return PassResult(gm, modified)

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            z = torch.add(y, x)
            return z

        exported_program = (
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program
        )

        pm = ExportedProgramPassManager(passes=[replace_add_with_mul])
        result = pm(exported_program)

        # Verify return type
        self.assertIsInstance(result, ExportedProgramPassResult)
        self.assertTrue(result.modified)

        # Check that all add ops were replaced with mul
        for node in result.exported_program.graph_module.graph.nodes:
            if node.op == "call_function":
                self.assertNotIn("add", str(node.target).lower())

    def test_updates_constants_on_exported_program(self) -> None:
        """
        Tests that ExportedProgramPassManager can update constants
        in the ExportedProgram using an ExportedProgram-aware pass.
        """

        class DoubleConstantsPass(ExportedProgramPassBase):
            """Pass that doubles all constant tensor values in the ExportedProgram."""

            def call(
                self, ep: ExportedProgram
            ) -> ExportedProgramPassResult:
                modified = False
                for key, const in ep.constants.items():
                    if isinstance(const, torch.Tensor):
                        ep.constants[key] = const * 2
                        modified = True
                return ExportedProgramPassResult(ep, modified)

        class ModuleWithConstant(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.ones(3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.weight

        module = ModuleWithConstant()
        exported_program = to_edge(
            torch.export.export(module, (torch.randn(3),))
        ).exported_program()

        # Verify there are constants in the ExportedProgram
        self.assertGreater(
            len(exported_program.constants), 0, "Expected constants in ExportedProgram"
        )

        # Store original constant values
        original_values = {
            key: const.clone()
            for key, const in exported_program.constants.items()
            if isinstance(const, torch.Tensor)
        }

        pm = ExportedProgramPassManager(passes=[DoubleConstantsPass()])
        result = pm(exported_program)

        self.assertIsInstance(result, ExportedProgramPassResult)
        self.assertTrue(result.modified)

        # Verify constants were doubled
        for key, original_const in original_values.items():
            new_const = result.exported_program.constants[key]
            self.assertTrue(
                torch.allclose(new_const, original_const * 2),
                f"Constant {key} was not doubled correctly",
            )

    def test_adds_constant_to_exported_program(self) -> None:
        """
        Tests that ExportedProgramPassManager can add a new constant
        to the ExportedProgram, including updating the graph and input specs.
        """

        class AddConstantPass(ExportedProgramPassBase):
            """Pass that adds a new constant tensor to the ExportedProgram."""

            def call(
                self, ep: ExportedProgram
            ) -> ExportedProgramPassResult:
                graph = ep.graph_module.graph
                sig = ep.graph_signature

                # Find the first user input to insert before it
                placeholders = graph.find_nodes(op="placeholder")
                assert len(placeholders) == 1
                user_input_node = placeholders[0]

                # Create a new constant tensor
                new_constant_name = "_test_added_constant"
                new_constant_tensor = torch.tensor([1.0, 2.0, 3.0])

                # Add placeholder node for the new constant
                with graph.inserting_before(user_input_node):
                    new_placeholder = graph.placeholder(new_constant_name)
                    # Set up meta for the new placeholder
                    new_placeholder.meta["val"] = new_constant_tensor

                # Add the constant to the constants dict
                ep.constants[new_constant_name] = new_constant_tensor

                # Update input specs to include the new constant
                new_input_spec = InputSpec(
                    kind=InputKind.CONSTANT_TENSOR,
                    arg=TensorArgument(name=new_placeholder.name),
                    target=new_constant_name,
                    persistent=False,
                )
                sig.input_specs = (new_input_spec, sig.input_specs[0])

                return ExportedProgramPassResult(ep, modified=True)

        class IdentityModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        exported_program = to_edge(
            torch.export.export(IdentityModule(), (torch.randn(3),))
        ).exported_program()
        assert len(exported_program.constants) == 0
        assert len(exported_program.graph_signature.input_specs) == 1

        pm = ExportedProgramPassManager(passes=[AddConstantPass()])
        result = pm(exported_program)

        self.assertIsInstance(result, ExportedProgramPassResult)
        self.assertTrue(result.modified)

        # Verify the new constant was added to constants dict
        self.assertEqual(len(result.exported_program.constants), 1)
        self.assertIn("_test_added_constant", result.exported_program.constants)
        self.assertTrue(
            torch.allclose(
                result.exported_program.constants["_test_added_constant"],
                torch.tensor([1.0, 2.0, 3.0]),
            )
        )

        # Verify input_specs was updated
        self.assertEqual(
            len(result.exported_program.graph_signature.input_specs),
            2,
        )

        # Verify the new placeholder exists in the graph
        placeholder_names = [
            node.target
            for node in result.exported_program.graph_module.graph.find_nodes(
                op="placeholder"
            )
        ]
        self.assertTrue(len(placeholder_names) == 2)

        # Verify the new input spec has the correct kind
        new_spec = None
        for spec in result.exported_program.graph_signature.input_specs:
            if spec.target == "_test_added_constant":
                new_spec = spec
                break
        self.assertIsNotNone(new_spec)
        self.assertEqual(new_spec.kind, InputKind.CONSTANT_TENSOR)

    def test_invalid_pass_creates_call_method(self) -> None:
        """
        Tests that ExportedProgramPassManager detects invalid passes
        that introduce call_method nodes.
        """

        def introduce_call_method(gm: torch.fx.GraphModule) -> PassResult:
            node = list(gm.graph.nodes)[-2]
            with gm.graph.inserting_after(node):
                gm.graph.call_method("torch.ops.relu", (torch.randn(2),))
            return PassResult(gm, True)

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            return y

        exported_program = (
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program
        )

        pm = ExportedProgramPassManager(
            passes=[introduce_call_method], run_checks_after_each_pass=True
        )

        with self.assertRaisesRegex(Exception, "call_method"):
            pm(exported_program)
