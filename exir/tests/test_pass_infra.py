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
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import ScalarToTensorPass
from executorch.exir.passes.pass_registry import PassRegistry
from executorch.exir.program import to_edge
from executorch.exir.edge_program_manager_pass_base import (
    EdgeProgramManagerPassBase,
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    EdgeProgramManagerPassResult,
    ExportedProgramToEdgeProgramManagerPassWrapper,
    MethodFilteredEdgeProgramManagerPass,
)
from torch.export import ExportedProgram, export
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
            for node in gm.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.add.Tensor
            ):
                node.target = torch.mul

        def replace_mul_with_div(gm: torch.fx.GraphModule) -> None:
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target == torch.mul:
                    node.target = torch.div

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.add(x, x)
                z = torch.add(y, x)
                return z

        f = (
            to_edge(export(AddModule(), (torch.randn(10),), strict=True))
            .exported_program()
            .graph_module
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

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.add(x, x)
                z = torch.add(y, x)
                return z

        traced_f1 = (
            to_edge(export(AddModule(), (torch.randn(10),), strict=True))
            .exported_program()
            .graph_module
        )
        pm1 = PassManager(
            passes=[introduce_call_method], run_checks_after_each_pass=True
        )

        with self.assertRaisesRegex(Exception, "call_method"):
            pm1(traced_f1)

    def test_pass_metadata(self) -> None:
        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        sample_inputs = (torch.randn(1, 3), torch.randn(1, 3))
        gm = export(AddModule(), sample_inputs, strict=True).module()

        pass_result = ScalarToTensorPass()(gm)
        self.assertIsNotNone(pass_result)
        new_gm = pass_result.graph_module

        for node in new_gm.graph.nodes:
            if node.target != "output":
                self.assertIn("val", node.meta)


class TestExportedProgramPassManager(unittest.TestCase):
    """Tests for EdgeProgramManager.transform() pass infrastructure.

    These tests validate that the pass manager correctly operates on EdgeProgramManagers,
    preserving the original test objectives from when it operated on ExportedPrograms directly.
    """

    def test_raises_spec_violation_error(self) -> None:
        """
        Ensures that transform() raises a SpecViolationError after running
        a pass which places a non-Edge operator in the graph.
        """

        def replace_add_with_torch_aten_mul(gm: torch.fx.GraphModule) -> PassResult:
            modified = False
            for node in gm.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.add.Tensor
            ):
                node.target = torch.ops.aten.mul.Tensor
                modified = True
            return PassResult(gm, modified)

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            z = torch.add(y, x)
            return z

        epm = to_edge(
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program
        )

        with self.assertRaisesRegex(
            torch._export.verifier.SpecViolationError,
            "Operator torch._ops.aten.mul.Tensor is not an Edge operator.",
        ):
            epm.transform([replace_add_with_torch_aten_mul])

    def test_runs_graph_module_passes_on_exported_program(self) -> None:
        """
        Tests that transform() runs GraphModule passes
        on an EdgeProgramManager and the graph is correctly modified.
        """

        def replace_add_with_mul(gm: torch.fx.GraphModule) -> PassResult:
            modified = False
            for node in gm.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.add.Tensor
            ):
                node.target = exir_ops.edge.aten.mul.Tensor
                modified = True
            return PassResult(gm, modified)

        def f(x: torch.Tensor) -> torch.Tensor:
            y = torch.add(x, x)
            z = torch.add(y, x)
            return z

        epm = to_edge(
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program
        )

        result_epm = epm.transform([replace_add_with_mul])

        # Check that all add ops were replaced with mul
        add_nodes = result_epm.exported_program().graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.add.Tensor
        )
        self.assertEqual(len(add_nodes), 0)

    def test_updates_constants_on_exported_program(self) -> None:
        """
        Tests that transform() can update constants
        in the ExportedProgram using an ExportedProgram-aware pass.
        """

        class DoubleConstantsPass(ExportedProgramPassBase):
            """Pass that doubles all constant tensor values in the ExportedProgram."""

            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
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
        epm = to_edge(
            torch.export.export(module, (torch.randn(3),))
        )
        exported_program = epm.exported_program()

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

        result_epm = epm.transform([DoubleConstantsPass()])

        # Verify constants were doubled
        new_ep = result_epm.exported_program()
        for key, original_const in original_values.items():
            new_const = new_ep.constants[key]
            self.assertTrue(
                torch.allclose(new_const, original_const * 2),
                f"Constant {key} was not doubled correctly",
            )

    def test_adds_constant_to_exported_program(self) -> None:
        """
        Tests that transform() can add a new constant
        to the ExportedProgram, including updating the graph and input specs.
        """

        class AddConstantPass(ExportedProgramPassBase):
            """Pass that adds a new constant tensor to the ExportedProgram."""

            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
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

        epm = to_edge(
            torch.export.export(IdentityModule(), (torch.randn(3),))
        )
        exported_program = epm.exported_program()
        assert len(exported_program.constants) == 0
        assert len(exported_program.graph_signature.input_specs) == 1

        result_epm = epm.transform([AddConstantPass()])

        new_ep = result_epm.exported_program()

        # Verify the new constant was added to constants dict
        self.assertEqual(len(new_ep.constants), 1)
        self.assertIn("_test_added_constant", new_ep.constants)
        self.assertTrue(
            torch.allclose(
                new_ep.constants["_test_added_constant"],
                torch.tensor([1.0, 2.0, 3.0]),
            )
        )

        # Verify input_specs was updated
        self.assertEqual(
            len(new_ep.graph_signature.input_specs),
            2,
        )

        # Verify the new placeholder exists in the graph
        placeholder_names = [
            node.target
            for node in new_ep.graph_module.graph.find_nodes(
                op="placeholder"
            )
        ]
        self.assertTrue(len(placeholder_names) == 2)

        # Verify the new input spec has the correct kind
        new_spec = None
        for spec in new_ep.graph_signature.input_specs:
            if spec.target == "_test_added_constant":
                new_spec = spec
                break
        self.assertIsNotNone(new_spec)
        self.assertEqual(new_spec.kind, InputKind.CONSTANT_TENSOR)

    def test_invalid_pass_creates_call_method(self) -> None:
        """
        Tests that transform() detects invalid passes
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

        epm = to_edge(
            exir.capture(f, (torch.randn(10),), exir.CaptureConfig())
            .to_edge()
            .exported_program
        )

        with self.assertRaisesRegex(Exception, "call_method"):
            epm.transform(
                [introduce_call_method], run_checks_after_each_pass=True
            )


class TestEdgeProgramManagerWrappers(unittest.TestCase):
    """Tests for the new EPM-level pass wrappers and MethodFilteredEdgeProgramManagerPass."""

    def _make_simple_epm(self):
        """Helper to create a simple EdgeProgramManager with a single 'forward' method."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.add(x, x)

        return to_edge(torch.export.export(SimpleModule(), (torch.randn(10),)))

    def test_runs_epm_pass_directly(self) -> None:
        """
        Tests that EdgeProgramManagerPassBase subclasses can operate
        directly on the EPM (e.g., modifying config methods).
        """

        class AddConfigMethodPass(EdgeProgramManagerPassBase):
            def call(self, epm):
                import copy

                new_epm = copy.copy(epm)
                new_epm._config_methods = dict(epm._config_methods or {})
                new_epm._config_methods["new_config"] = "test_value"
                return EdgeProgramManagerPassResult(new_epm, modified=True)

        epm = self._make_simple_epm()

        result_epm = epm.transform([AddConfigMethodPass()])

        self.assertIn("new_config", result_epm.config_methods)

    def test_exported_program_to_epm_wrapper(self) -> None:
        """
        Tests that ExportedProgramToEdgeProgramManagerPassWrapper correctly
        iterates over all methods in the EPM.
        """

        class NoOpPass(ExportedProgramPassBase):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                self.call_count += 1
                return ExportedProgramPassResult(ep, modified=False)

        epm = self._make_simple_epm()
        inner_pass = NoOpPass()
        wrapper = ExportedProgramToEdgeProgramManagerPassWrapper(inner_pass)

        result = wrapper(epm)
        self.assertIsInstance(result, EdgeProgramManagerPassResult)
        self.assertFalse(result.modified)
        self.assertEqual(inner_pass.call_count, len(epm.methods))

    def test_graph_module_to_epm_two_step_wrapping(self) -> None:
        """
        Tests that wrapping a GraphModule pass with
        GraphModuleBackedExportedProgramPassWrapper and then
        ExportedProgramToEdgeProgramManagerPassWrapper correctly
        applies it to all methods.
        """
        from executorch.exir.edge_program_manager_pass_base import (
            GraphModuleBackedExportedProgramPassWrapper,
        )
        from torch.fx.passes.infra.pass_manager import pass_result_wrapper

        call_count = 0

        def counting_pass(gm: torch.fx.GraphModule) -> PassResult:
            nonlocal call_count
            call_count += 1
            return PassResult(gm, False)

        epm = self._make_simple_epm()
        ep_pass = GraphModuleBackedExportedProgramPassWrapper(
            pass_result_wrapper(counting_pass)
        )
        wrapper = ExportedProgramToEdgeProgramManagerPassWrapper(ep_pass)

        result = wrapper(epm)
        self.assertIsInstance(result, EdgeProgramManagerPassResult)
        self.assertFalse(result.modified)
        self.assertEqual(call_count, len(epm.methods))

    def test_method_filtered_pass(self) -> None:
        """
        Tests that MethodFilteredEdgeProgramManagerPass applies passes
        only to specified methods.
        """

        def replace_add_with_mul(gm: torch.fx.GraphModule) -> PassResult:
            modified = False
            for node in gm.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.add.Tensor
            ):
                node.target = exir_ops.edge.aten.mul.Tensor
                modified = True
            return PassResult(gm, modified)

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.add(x, x)

        epm = to_edge(
            {
                "forward": torch.export.export(AddModule(), (torch.randn(10),)),
                "other": torch.export.export(AddModule(), (torch.randn(10),)),
            }
        )

        filtered_pass = MethodFilteredEdgeProgramManagerPass(
            {"forward": [replace_add_with_mul]}
        )

        result = filtered_pass(epm)
        self.assertTrue(result.modified)

        # 'forward' should have mul ops (no add remaining)
        add_nodes = result.edge_program_manager.exported_program("forward").graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.add.Tensor
        )
        self.assertEqual(len(add_nodes), 0)

        # 'other' should still have add ops
        add_nodes = result.edge_program_manager.exported_program("other").graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.add.Tensor
        )
        self.assertGreater(len(add_nodes), 0, "Expected 'other' method to still have add ops")
