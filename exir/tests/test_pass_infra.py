# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest

import executorch.exir as exir
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
    ExportPassBaseError,
    NodeMetadata,
    ProxyValue,
)
from executorch.exir.pass_manager import ExportedProgramPassManager, PassManager
from executorch.exir.passes import ScalarToTensorPass
from executorch.exir.passes.pass_registry import PassRegistry
from executorch.exir.program import to_edge
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import Dim, export, ExportedProgram
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


class TestProxyValueSymbolicCoercions(unittest.TestCase):
    @staticmethod
    def _symbolic_values() -> tuple[torch.SymInt, torch.SymFloat]:
        class ViewModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.size(0), -1)

        exported = export(
            ViewModule(),
            (torch.randn(2, 3),),
            dynamic_shapes=({0: Dim("batch", min=1, max=8)},),
            strict=True,
        )
        gm = to_edge(exported).exported_program().graph_module
        for node in gm.graph.nodes:
            value = node.meta.get("val")
            if isinstance(value, torch.SymInt):
                return value, torch.sym_float(value)
        raise AssertionError("Expected a symbolic scalar in exported graph metadata")

    def test_rejects_implicit_symbolic_scalar_coercions(self) -> None:
        sym_int, sym_float = self._symbolic_values()

        with self.assertRaisesRegex(ExportPassBaseError, "boolean context"):
            bool(ProxyValue(sym_int, torch.fx.Graph().placeholder("x")))

        with self.assertRaisesRegex(ExportPassBaseError, "converted to int"):
            int(ProxyValue(sym_int, torch.fx.Graph().placeholder("x")))

        with self.assertRaisesRegex(ExportPassBaseError, "used in index context"):
            ProxyValue(sym_int, torch.fx.Graph().placeholder("x")).__index__()

        with self.assertRaisesRegex(ExportPassBaseError, "converted to float"):
            float(ProxyValue(sym_float, torch.fx.Graph().placeholder("x")))


class TestExportedProgramPassManager(unittest.TestCase):
    def test_runs_graph_module_passes_on_exported_program(self) -> None:
        """
        Tests that ExportedProgramPassManager runs GraphModule passes
        on an ExportedProgram and the graph is correctly modified.
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
        self.assertEqual(
            len(
                result.exported_program.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.add.Tensor
                )
            ),
            0,
        )

    def test_updates_constants_on_exported_program(self) -> None:
        """
        Tests that ExportedProgramPassManager can update constants
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


class TestPassBaseSymbolicInputs(unittest.TestCase):
    class SymSizeModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.view(x.size(0), -1)

    @staticmethod
    def _find_input_node(gm: torch.fx.GraphModule) -> torch.fx.Node:
        for node in gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                return node
        raise AssertionError("Expected to find an input placeholder")

    @staticmethod
    def _symbolic_input_shape(node: torch.fx.Node) -> tuple[str | None, ...]:
        value = node.meta["val"]
        assert isinstance(value, torch.Tensor)
        return tuple(
            str(dim) if isinstance(dim, torch.SymInt) else None for dim in value.shape
        )

    def _export_dynamic_graph_module(self) -> torch.fx.GraphModule:
        exported = export(
            self.SymSizeModule(),
            (torch.randn(2, 3),),
            dynamic_shapes=({0: Dim("batch", min=1, max=8)},),
            strict=True,
        )
        return to_edge(exported).exported_program().graph_module

    def test_export_pass_preserves_symbolic_input_metadata(self) -> None:
        graph_module = self._export_dynamic_graph_module()
        original_input = self._find_input_node(graph_module)
        original_snapshot = self._symbolic_input_shape(original_input)
        self.assertTrue(any(dim is not None for dim in original_snapshot))

        new_graph_module = ExportPass()(graph_module).graph_module
        new_input = self._find_input_node(new_graph_module)

        self.assertEqual(self._symbolic_input_shape(new_input), original_snapshot)

    def test_export_pass_matches_symbolic_inputs_by_position(self) -> None:
        class RenamePlaceholderPass(ExportPass):
            def placeholder(
                self,
                name: str,
                arg: torch.Tensor,
                meta: NodeMetadata,
            ) -> ProxyValue:
                return super().placeholder(f"renamed_{name}", arg, meta)

        new_graph_module = RenamePlaceholderPass()(
            self._export_dynamic_graph_module()
        ).graph_module
        new_input = self._find_input_node(new_graph_module)

        self.assertEqual(new_input.name, "renamed_x")
        self.assertTrue(
            any(dim is not None for dim in self._symbolic_input_shape(new_input))
        )

    def test_export_pass_ignores_symbolic_metadata_for_constant_input(self) -> None:
        graph_module = self._export_dynamic_graph_module()
        original_input = self._find_input_node(graph_module)
        original_value = original_input.meta["val"]
        self.assertIsInstance(original_value, FakeTensor)
        assert isinstance(original_value, FakeTensor)
        constant = torch.randn(2, 3)
        original_value.constant = constant

        new_graph_module = ExportPass()(graph_module).graph_module
        new_input = self._find_input_node(new_graph_module)
        new_value = new_input.meta["val"]

        self.assertIsInstance(new_value, FakeTensor)
        assert isinstance(new_value, FakeTensor)
        self.assertIs(new_value.constant, constant)
        self.assertEqual(
            self._symbolic_input_shape(new_input),
            self._symbolic_input_shape(original_input),
        )

    def test_export_pass_rejects_collapsed_symbolic_input_metadata(self) -> None:
        class CollapseSymbolicInputPass(ExportPass):
            def placeholder(
                self,
                name: str,
                arg: torch.Tensor,
                meta: NodeMetadata,
            ) -> ProxyValue:
                proxy = super().placeholder(name, arg, meta)
                if any(isinstance(dim, torch.SymInt) for dim in arg.shape):
                    proxy.node.meta["val"] = torch.empty(2, 3, device="meta")
                return proxy

        with self.assertRaisesRegex(
            ExportPassBaseError,
            "Input at position 0 did not preserve symbolic metadata",
        ):
            CollapseSymbolicInputPass()(self._export_dynamic_graph_module())

    def test_export_pass_can_disable_symbolic_input_validation(self) -> None:
        class CollapseSymbolicInputPass(ExportPass):
            def should_preserve_symbolic_input_metadata(self) -> bool:
                return False

            def placeholder(
                self,
                name: str,
                arg: torch.Tensor,
                meta: NodeMetadata,
            ) -> ProxyValue:
                proxy = super().placeholder(name, arg, meta)
                if any(isinstance(dim, torch.SymInt) for dim in arg.shape):
                    proxy.node.meta["val"] = torch.empty(2, 3, device="meta")
                return proxy

        graph_module = self._export_dynamic_graph_module()
        original_snapshot = self._symbolic_input_shape(
            self._find_input_node(graph_module)
        )

        new_graph_module = CollapseSymbolicInputPass()(graph_module).graph_module
        new_input = self._find_input_node(new_graph_module)

        self.assertNotEqual(self._symbolic_input_shape(new_input), original_snapshot)


class ExportedProgramPassBaseOutputSpecTest(unittest.TestCase):
    """__call__ realigns output specs with the graph before ensures() runs."""

    class _Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + x

    def _program(self) -> ExportedProgram:
        return to_edge(export(self._Model(), (torch.randn(2, 2),))).exported_program()

    def test_replacing_the_output_node_updates_the_signature(self) -> None:
        class ReplaceOutputPass(ExportedProgramPassBase):
            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                graph = ep.graph_module.graph
                output_node = graph.output_node()
                (old,) = output_node.args[0]
                with graph.inserting_before(output_node):
                    new = graph.call_function(
                        exir_ops.edge.aten.mul.Tensor, (old.args[0], old.args[1])
                    )
                new.meta = dict(old.meta)
                output_node.args = ((new,),)
                return ExportedProgramPassResult(ep, True)

        program = self._program()
        result = ReplaceOutputPass()(program)

        (spec,) = result.exported_program.graph_signature.output_specs
        graph_output_name = result.exported_program.graph.output_node().args[0][0].name
        self.assertEqual(spec.arg.name, graph_output_name)
        result.exported_program.validate()

    def test_a_signature_only_change_is_reported_as_modified(self) -> None:
        """A pass that renames the output reports modified even if it says False."""

        class RenameOutputPass(ExportedProgramPassBase):
            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                ep.graph.output_node().args[0][0].name = "renamed_output"
                return ExportedProgramPassResult(ep, False)

        result = RenameOutputPass()(self._program())

        self.assertTrue(result.modified)
        (spec,) = result.exported_program.graph_signature.output_specs
        self.assertEqual(spec.arg.name, "renamed_output")

    def test_output_count_mismatch_is_rejected(self) -> None:
        class DropOutputPass(ExportedProgramPassBase):
            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                output_node = ep.graph.output_node()
                output_node.args = ((*output_node.args[0], output_node.args[0][0]),)
                return ExportedProgramPassResult(ep, True)

        with self.assertRaisesRegex(ExportPassBaseError, "output specs"):
            DropOutputPass()(self._program())

    def test_the_replace_hook_updates_the_signature_during_the_pass(self) -> None:
        """A pass using the replacement APIs sees a valid signature as it runs."""

        signature_during_pass = []

        class ReplaceViaApiPass(ExportedProgramPassBase):
            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                graph = ep.graph_module.graph
                (old,) = graph.output_node().args[0]
                with graph.inserting_before(graph.output_node()):
                    new = graph.call_function(
                        exir_ops.edge.aten.mul.Tensor, (old.args[0], old.args[1])
                    )
                new.meta = dict(old.meta)
                old.replace_all_uses_with(new)
                signature_during_pass.append(
                    ep.graph_signature.output_specs[0].arg.name
                )
                return ExportedProgramPassResult(ep, True)

        result = ReplaceViaApiPass()(self._program())

        graph_output_name = result.exported_program.graph.output_node().args[0][0].name
        self.assertEqual(signature_during_pass, [graph_output_name])

    def test_replacing_a_returned_buffer_leaves_input_specs_alone(self) -> None:
        """The hook is output-only, so the replaced placeholder stays deletable."""

        class TwoBuffers(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("a", torch.ones(2, 2))
                self.register_buffer("b", torch.ones(2, 2))

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return self.a, x + self.b

        replaced_name = []

        class ReplaceReturnedBufferPass(ExportedProgramPassBase):
            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                names = {
                    target: name
                    for name, target in ep.graph_signature.inputs_to_buffers.items()
                }
                placeholders = {
                    node.name: node
                    for node in ep.graph.nodes
                    if node.op == "placeholder"
                }
                old = placeholders[names["a"]]
                old.replace_all_uses_with(placeholders[names["b"]])
                replaced_name.append(old.name)
                return ExportedProgramPassResult(ep, True)

        program = to_edge(export(TwoBuffers(), (torch.randn(2, 2),))).exported_program()

        result = ReplaceReturnedBufferPass()(program)

        signature = result.exported_program.graph_signature
        self.assertIn(replaced_name[0], signature.inputs_to_buffers)
        result.exported_program.validate()

    def test_mutating_a_copied_graph_leaves_the_original_signature_alone(self) -> None:
        """GraphModule.__deepcopy__ carries the replace hook over to the copy."""

        signature_after_copy_edit = []

        class MutateACopyPass(ExportedProgramPassBase):
            def call(self, ep: ExportedProgram) -> ExportedProgramPassResult:
                graph_module = copy.deepcopy(ep.graph_module)
                graph = graph_module.graph
                (old,) = graph.output_node().args[0]
                with graph.inserting_before(graph.output_node()):
                    new = graph.call_function(
                        exir_ops.edge.aten.mul.Tensor, (old.args[0], old.args[1])
                    )
                new.meta = dict(old.meta)
                old.replace_all_uses_with(new)
                signature_after_copy_edit.append(
                    ep.graph_signature.output_specs[0].arg.name
                )
                return ExportedProgramPassResult(ep, False)

        program = self._program()
        original_output_name = program.graph_signature.output_specs[0].arg.name

        result = MutateACopyPass()(program)

        self.assertEqual(signature_after_copy_edit, [original_output_name])
        self.assertFalse(result.modified)
        program.validate()
