# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any

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
from torch.fx.passes.shape_prop import _extract_tensor_metadata


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


class TestExportPassFastCopy(unittest.TestCase):
    class _CountingTargetedPass(ExportPass):
        enable_fast_copy = True

        def __init__(
            self,
            targeted_ops: tuple[Any, ...] = (torch.ops.aten.mul.Tensor,),
        ) -> None:
            super().__init__()
            self.targeted_ops = targeted_ops
            self.operator_calls = 0

        def call_operator(
            self,
            op: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            meta: NodeMetadata,
        ) -> ProxyValue:
            self.operator_calls += 1
            return super().call_operator(op, args, kwargs, meta)

    @staticmethod
    def _edge_graph_module(module: torch.nn.Module) -> torch.fx.GraphModule:
        return (
            to_edge(export(module, (torch.randn(2),), strict=True))
            .exported_program()
            .graph_module
        )

    @staticmethod
    def _raw_add_graph_module(
        dynamic_shapes: Any | None = None,
    ) -> torch.fx.GraphModule:
        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        return export(
            AddModule(),
            (torch.randn(2),),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).graph_module

    @staticmethod
    def _ensure_tensor_meta(graph_module: torch.fx.GraphModule) -> None:
        for node in graph_module.graph.nodes:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and "tensor_meta" not in node.meta:
                node.meta["tensor_meta"] = _extract_tensor_metadata(value)

    @staticmethod
    def _call_function_targets(
        graph_module: torch.fx.GraphModule,
    ) -> list[torch.fx.node.Target]:
        return [
            node.target
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        ]

    def test_target_ops_alone_does_not_enable_fast_copy(self) -> None:
        graph_module = self._edge_graph_module(self._AddModule())

        class TargetOpsOnlyPass(ExportPass):
            target_ops = (exir_ops.edge.aten.mul.Tensor,)

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                return super().call_operator(op, args, kwargs, meta)

        pass_ = TargetOpsOnlyPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 1)

    def test_targeted_ops_alone_does_not_enable_fast_copy(self) -> None:
        graph_module = self._edge_graph_module(self._AddModule())

        class TargetedOpsOnlyPass(ExportPass):
            targeted_ops: tuple[()] = ()

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                return super().call_operator(op, args, kwargs, meta)

        pass_ = TargetedOpsOnlyPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 1)

    def test_explicit_empty_targeted_ops_enables_fast_copy(self) -> None:
        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        class EmptyTargetedOpsPass(ExportPass):
            enable_fast_copy = True
            targeted_ops: tuple[()] = ()
            target_ops = {exir_ops.edge.aten.add.Tensor}

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                return super().call_operator(op, args, kwargs, meta)

        graph_module = self._edge_graph_module(AddModule())
        pass_ = EmptyTargetedOpsPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 0)

    def test_missing_tensor_meta_uses_normal_replay(self) -> None:
        graph_module = self._edge_graph_module(self._AddModule())
        add_node = self._single_call_function_node(
            graph_module, exir_ops.edge.aten.add.Tensor
        )
        del add_node.meta["tensor_meta"]

        pass_ = self._CountingTargetedPass((exir_ops.edge.aten.mul.Tensor,))
        new_graph_module = pass_(graph_module).graph_module
        new_add_node = self._single_call_function_node(
            new_graph_module, exir_ops.edge.aten.add.Tensor
        )

        self.assertEqual(pass_.operator_calls, 1)
        self.assertIn("tensor_meta", new_add_node.meta)

    def test_node_debug_str_is_current_on_fast_and_slow_paths(self) -> None:
        class AddThenMulModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x + x) * x

        graph_module = export(
            AddThenMulModule(), (torch.randn(2),), strict=True
        ).graph_module
        self._ensure_tensor_meta(graph_module)
        expected = {
            node.target: node.format_node()
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        }

        class DebugTrackingPass(self._CountingTargetedPass):
            def __init__(self) -> None:
                super().__init__((torch.ops.aten.mul.Tensor,))
                self.debug_strings: dict[torch.fx.node.Target, str | None] = {}

            def should_fast_copy_node(self, target: torch.fx.node.Target) -> bool:
                self.debug_strings[target] = self.node_debug_str
                return super().should_fast_copy_node(target)

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.debug_strings[op] = self.node_debug_str
                return super().call_operator(op, args, kwargs, meta)

        pass_ = DebugTrackingPass()

        pass_(graph_module)

        self.assertEqual(pass_.debug_strings, expected)

    def test_should_fast_copy_node_hook_keeps_selected_cold_ops_on_slow_path(
        self,
    ) -> None:
        graph_module = self._edge_graph_module(self._AddModule())

        class HookedPass(ExportPass):
            enable_fast_copy = True
            targeted_ops: tuple[()] = ()

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def should_fast_copy_node(self, target: torch.fx.node.Target) -> bool:
                return target is not exir_ops.edge.aten.add.Tensor

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                return super().call_operator(op, args, kwargs, meta)

        pass_ = HookedPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 1)

    def test_convolution_and_linear_targets_are_not_fast_copied(self) -> None:
        unsafe_targets = (
            torch.ops.aten.convolution,
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv1d,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv1d.padding,
            torch.ops.aten.conv2d,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
            torch.ops.aten.conv3d,
            torch.ops.aten.conv3d.default,
            torch.ops.aten.conv3d.padding,
            torch.ops.aten.conv_transpose1d,
            torch.ops.aten.conv_transpose1d.default,
            torch.ops.aten.conv_transpose2d,
            torch.ops.aten.conv_transpose2d.input,
            torch.ops.aten.conv_transpose3d,
            torch.ops.aten.conv_transpose3d.input,
            torch.ops.aten.linear,
            torch.ops.aten.linear.default,
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.conv2d.default,
            exir_ops.edge.aten.conv2d.padding,
            exir_ops.edge.aten.conv3d.default,
            exir_ops.edge.aten.conv3d.padding,
            exir_ops.edge.aten.linear.default,
        )
        pass_ = ExportPass()

        for target in unsafe_targets:
            with self.subTest(target=target):
                self.assertFalse(pass_.should_fast_copy_node(target))
        self.assertTrue(pass_.should_fast_copy_node(torch.ops.aten.add.Tensor))

    def test_packet_target_does_not_match_overload_target(self) -> None:
        graph_module = self._raw_add_graph_module()
        self._ensure_tensor_meta(graph_module)

        pass_ = self._CountingTargetedPass((torch.ops.aten.add,))
        new_graph_module = pass_(graph_module).graph_module

        self.assertEqual(pass_.operator_calls, 0)
        self.assertEqual(
            self._call_function_targets(new_graph_module), [torch.ops.aten.add.Tensor]
        )

    def test_symbolic_metadata_drift_check_does_not_force_symint_bool(
        self,
    ) -> None:
        graph_module = self._raw_add_graph_module(
            dynamic_shapes=({0: Dim("batch", min=1, max=8)},)
        )

        pass_ = self._CountingTargetedPass((torch.ops.aten.add.Tensor,))
        new_graph_module = pass_(graph_module).graph_module

        self.assertEqual(pass_.operator_calls, 1)
        self.assertEqual(
            self._call_function_targets(new_graph_module), [torch.ops.aten.add.Tensor]
        )

    def test_nested_target_output_metadata_drift_disables_downstream_fast_copy(
        self,
    ) -> None:
        class MaxThenAddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values, _ = torch.max(x, dim=1)
                return values + values

        graph_module = export(
            MaxThenAddModule(),
            (torch.randn(2, 3),),
            strict=True,
        ).graph_module
        self._ensure_tensor_meta(graph_module)

        class TupleMetadataDriftPass(ExportPass):
            enable_fast_copy = True
            targeted_ops = (torch.ops.aten.max.dim,)

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue | tuple[ProxyValue, ProxyValue]:
                self.operator_calls += 1
                result = super().call_operator(op, args, kwargs, meta)
                if op is not torch.ops.aten.max.dim:
                    return result

                values = self.call_getitem(result, 0, meta)
                indices = self.call_getitem(result, 1, meta)
                return (ProxyValue(values.data.unsqueeze(0), values.proxy), indices)

        pass_ = TupleMetadataDriftPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 2)

    def test_tuple_result_keeps_downstream_fast_copy_enabled(self) -> None:
        class MaxThenAddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values, _ = torch.max(x, dim=1)
                return values + values

        graph_module = export(
            MaxThenAddModule(),
            (torch.randn(2, 3),),
            strict=True,
        ).graph_module
        self._ensure_tensor_meta(graph_module)

        class TupleResultPass(ExportPass):
            enable_fast_copy = True
            targeted_ops = (torch.ops.aten.max.dim,)

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue | tuple[ProxyValue, ProxyValue]:
                self.operator_calls += 1
                result = super().call_operator(op, args, kwargs, meta)
                if op is not torch.ops.aten.max.dim:
                    return result
                return (
                    self.call_getitem(result, 0, meta),
                    self.call_getitem(result, 1, meta),
                )

        pass_ = TupleResultPass()
        new_graph_module = pass_(graph_module).graph_module
        test_input = torch.randn(2, 3)

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 1)
        torch.testing.assert_close(
            new_graph_module(test_input),
            graph_module(test_input),
        )

    def test_copied_get_attr_is_reused_by_hot_node_and_calls_on_attr(self) -> None:
        value = torch.ones(2)
        root = torch.nn.Module()
        root.register_buffer("weight", value, persistent=False)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(2)
        weight = graph.get_attr("weight")
        weight.meta["val"] = value
        cold = graph.call_function(torch.ops.aten.add.Tensor, (x, weight))
        cold.meta["val"] = x.meta["val"] + value
        cold.meta["tensor_meta"] = _extract_tensor_metadata(cold.meta["val"])
        hot = graph.call_function(torch.ops.aten.mul.Tensor, (cold, weight))
        hot.meta["val"] = cold.meta["val"] * value
        hot.meta["tensor_meta"] = _extract_tensor_metadata(hot.meta["val"])
        graph.output(hot)
        graph_module = torch.fx.GraphModule(root, graph)

        class AttrTrackingPass(ExportPass):
            enable_fast_copy = True
            targeted_ops = (torch.ops.aten.mul.Tensor,)

            def __init__(self) -> None:
                super().__init__()
                self.attrs: list[ProxyValue] = []
                self.targeted_attr: Any = None

            def on_attr(self, attr: ProxyValue) -> None:
                self.attrs.append(attr)

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                if op is torch.ops.aten.mul.Tensor:
                    self.targeted_attr = args[1]
                return super().call_operator(op, args, kwargs, meta)

        pass_ = AttrTrackingPass()
        new_graph_module = pass_(graph_module).graph_module
        get_attrs = list(new_graph_module.graph.find_nodes(op="get_attr"))

        new_graph_module.graph.lint()
        self.assertEqual(len(get_attrs), 2)
        self.assertTrue(all(node.target == "weight" for node in get_attrs))
        self.assertEqual(len(pass_.attrs), 2)
        self.assertTrue(all(attr.data is value for attr in pass_.attrs))
        self.assertIs(pass_.targeted_attr, value)
        self.assertIn("weight", new_graph_module._buffers)
        self.assertIn("weight", new_graph_module._non_persistent_buffers_set)
        self.assertEqual(len(list(new_graph_module.named_buffers())), 1)
        torch.testing.assert_close(
            new_graph_module(torch.full((2,), 2.0)),
            graph_module(torch.full((2,), 2.0)),
        )

    def test_non_module_dotted_get_attr_fast_copy_fallback_is_atomic(self) -> None:
        weight = torch.ones(2)
        weight.x = torch.full((2,), 2.0)
        root = torch.nn.Module()
        root.add_module("w", torch.nn.Module())
        root.w.register_buffer("x", weight.x)
        graph = torch.fx.Graph()
        wx = graph.get_attr("w.x")
        wx.meta["val"] = weight.x
        cold_node = graph.call_function(torch.ops.aten.add.Tensor, (wx, wx))
        cold_node.meta["val"] = weight.x + weight.x
        cold_node.meta["tensor_meta"] = _extract_tensor_metadata(cold_node.meta["val"])
        graph.output(cold_node)
        graph_module = torch.fx.GraphModule(root, graph)
        delattr(graph_module, "w")
        graph_module.register_buffer("w", weight)
        pass_ = self._CountingTargetedPass()

        new_graph_module = pass_(graph_module).graph_module

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 1)
        self.assertFalse(
            any(
                isinstance(child, torch.nn.Module)
                for child in new_graph_module.children()
            )
        )
        self.assertFalse(
            any(
                node.op == "get_attr" and len(node.users) == 0
                for node in new_graph_module.graph.nodes
            )
        )
        torch.testing.assert_close(new_graph_module(), graph_module())

    def test_overlapping_get_attr_fast_copy_fallback_is_atomic(self) -> None:
        weight = torch.ones(2)
        weight.x = torch.ones(2)
        root = torch.nn.Module()
        root.register_buffer("w", weight)
        graph = torch.fx.Graph()
        w = graph.get_attr("w")
        w.meta["val"] = weight
        wx = graph.get_attr("w.x")
        wx.meta["val"] = weight.x
        cold_node = graph.call_function(torch.ops.aten.add.Tensor, (w, wx))
        cold_node.meta["val"] = weight + weight.x
        cold_node.meta["tensor_meta"] = _extract_tensor_metadata(cold_node.meta["val"])
        graph.output(cold_node)
        graph_module = torch.fx.GraphModule(root, graph)
        pass_ = self._CountingTargetedPass()

        new_graph_module = pass_(graph_module).graph_module

        self.assertEqual(pass_.operator_calls, 1)
        self.assertFalse(
            any(
                node.op == "get_attr" and len(node.users) == 0
                for node in new_graph_module.graph.nodes
            )
        )

    def test_missing_source_val_does_not_disable_downstream_fast_copy(self) -> None:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(2)
        without_val = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
        copied = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
        copied.meta["val"] = x.meta["val"] * x.meta["val"]
        copied.meta["tensor_meta"] = _extract_tensor_metadata(copied.meta["val"])
        graph.output((without_val, copied))
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        class MissingValPass(self._CountingTargetedPass):
            def should_fast_copy_node(self, target: torch.fx.node.Target) -> bool:
                return target is not torch.ops.aten.add.Tensor

        pass_ = MissingValPass(())
        new_graph_module = pass_(graph_module).graph_module

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 1)
        expected = graph_module(torch.full((2,), 3.0))
        actual = new_graph_module(torch.full((2,), 3.0))
        torch.testing.assert_close(actual, expected)

    def test_slow_path_metadata_drift_disables_downstream_fast_copy(self) -> None:
        class AddThenMulModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x + x) * x

        graph_module = export(
            AddThenMulModule(), (torch.randn(2),), strict=True
        ).graph_module
        self._ensure_tensor_meta(graph_module)

        class DriftPass(ExportPass):
            enable_fast_copy = True
            targeted_ops: tuple[()] = ()

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def should_fast_copy_node(self, target: torch.fx.node.Target) -> bool:
                return target is not torch.ops.aten.add.Tensor

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                result = super().call_operator(op, args, kwargs, meta)
                if op is torch.ops.aten.add.Tensor:
                    return ProxyValue(result.data.to(torch.float64), result.proxy)
                return result

        pass_ = DriftPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 2)

    def test_local_fallback_without_drift_keeps_fast_copy_enabled(self) -> None:
        value = torch.ones(2)
        value.x = torch.full((2,), 2.0)
        root = torch.nn.Module()
        root.add_module("w", torch.nn.Module())
        root.w.register_buffer("x", value.x)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.randn(2)
        wx = graph.get_attr("w.x")
        wx.meta["val"] = value.x
        fallback = graph.call_function(torch.ops.aten.add.Tensor, (wx, wx))
        fallback.meta["val"] = value.x + value.x
        fallback.meta["tensor_meta"] = _extract_tensor_metadata(fallback.meta["val"])
        copied = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
        copied.meta["val"] = x.meta["val"] * x.meta["val"]
        copied.meta["tensor_meta"] = _extract_tensor_metadata(copied.meta["val"])
        graph.output((fallback, copied))
        graph_module = torch.fx.GraphModule(root, graph)
        delattr(graph_module, "w")
        graph_module.register_buffer("w", value)

        pass_ = self._CountingTargetedPass(())
        new_graph_module = pass_(graph_module).graph_module

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 1)
        expected = graph_module(torch.full((2,), 3.0))
        actual = new_graph_module(torch.full((2,), 3.0))
        torch.testing.assert_close(actual, expected)

    def test_node_backed_proxy_value_supports_fast_copy(self) -> None:
        graph_module = self._raw_add_graph_module()
        self._ensure_tensor_meta(graph_module)

        class NodeBackedPlaceholderPass(ExportPass):
            enable_fast_copy = True
            targeted_ops: tuple[()] = ()

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def placeholder(
                self, name: str, arg: Any, meta: NodeMetadata
            ) -> ProxyValue:
                result = super().placeholder(name, arg, meta)
                return ProxyValue(result.data, result.node)

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                return super().call_operator(op, args, kwargs, meta)

        pass_ = NodeBackedPlaceholderPass()
        new_graph_module = pass_(graph_module).graph_module

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 0)
        torch.testing.assert_close(
            new_graph_module(torch.full((2,), 3.0)),
            graph_module(torch.full((2,), 3.0)),
        )

    def test_placeholder_metadata_drift_disables_fast_copy(self) -> None:
        graph_module = self._raw_add_graph_module()
        self._ensure_tensor_meta(graph_module)

        class PlaceholderDriftPass(ExportPass):
            enable_fast_copy = True
            targeted_ops: tuple[()] = ()

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def placeholder(
                self, name: str, arg: Any, meta: NodeMetadata
            ) -> ProxyValue:
                result = super().placeholder(name, arg, meta)
                return ProxyValue(result.data.to(torch.float64), result.proxy)

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> ProxyValue:
                self.operator_calls += 1
                return super().call_operator(op, args, kwargs, meta)

        pass_ = PlaceholderDriftPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 1)

    def test_unknown_metadata_leaf_drift_disables_fast_copy(self) -> None:
        graph_module = self._raw_add_graph_module()
        self._ensure_tensor_meta(graph_module)
        graph = graph_module.graph
        output = next(node for node in graph.nodes if node.op == "output")
        add = self._single_call_function_node(graph_module, torch.ops.aten.add.Tensor)

        class UnknownMetadataLeaf:
            def __repr__(self) -> str:
                raise AssertionError("metadata comparison must not call repr")

        add.meta["val"] = (add.meta["val"], UnknownMetadataLeaf())
        with graph.inserting_after(add):
            mul = graph.call_function(torch.ops.aten.mul.Tensor, add.args)
            mul.meta["val"] = add.meta["val"][0] * add.meta["val"][0]
            mul.meta["tensor_meta"] = _extract_tensor_metadata(mul.meta["val"])
        output.args = (mul,)
        graph_module.recompile()

        class UnknownLeafDriftPass(ExportPass):
            enable_fast_copy = True
            targeted_ops = (torch.ops.aten.add.Tensor,)

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> Any:
                self.operator_calls += 1
                result = super().call_operator(op, args, kwargs, meta)
                if op is torch.ops.aten.add.Tensor:
                    return (result, UnknownMetadataLeaf())
                return result

        pass_ = UnknownLeafDriftPass()

        pass_(graph_module)

        self.assertEqual(pass_.operator_calls, 2)

    def test_foreign_node_backed_proxy_value_disables_fast_copy(self) -> None:
        graph_module = self._raw_add_graph_module()
        self._ensure_tensor_meta(graph_module)

        graph = graph_module.graph
        output = next(node for node in graph.nodes if node.op == "output")
        add = self._single_call_function_node(graph_module, torch.ops.aten.add.Tensor)
        with graph.inserting_after(add):
            mul = graph.call_function(torch.ops.aten.mul.Tensor, add.args)
            mul.meta.update(add.meta)
        output.args = (mul,)
        graph_module.recompile()

        class ForeignNodePass(ExportPass):
            enable_fast_copy = True
            targeted_ops = (torch.ops.aten.add.Tensor,)

            def __init__(self) -> None:
                super().__init__()
                self.operator_calls = 0

            def call_operator(
                self,
                op: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                meta: NodeMetadata,
            ) -> Any:
                self.operator_calls += 1
                result = super().call_operator(op, args, kwargs, meta)
                if op is torch.ops.aten.add.Tensor:
                    foreign_node = torch.fx.Graph().placeholder("foreign")
                    return ProxyValue(result.data, foreign_node)
                return result

        pass_ = ForeignNodePass()
        new_graph_module = pass_(graph_module).graph_module

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 2)

    def test_hot_to_cold_dependency_is_remapped_and_executable(self) -> None:
        class AddThenMulModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x + x) * x

        graph_module = export(
            AddThenMulModule(), (torch.randn(2),), strict=True
        ).graph_module
        self._ensure_tensor_meta(graph_module)

        pass_ = self._CountingTargetedPass((torch.ops.aten.add.Tensor,))
        new_graph_module = pass_(graph_module).graph_module

        new_graph_module.graph.lint()
        self.assertEqual(pass_.operator_calls, 1)
        torch.testing.assert_close(
            new_graph_module(torch.full((2,), 3.0)),
            graph_module(torch.full((2,), 3.0)),
        )

    def test_on_attr_runtime_error_propagates(self) -> None:
        root = torch.nn.Module()
        root.register_buffer("weight", torch.ones(2))
        graph = torch.fx.Graph()
        weight = graph.get_attr("weight")
        cold = graph.call_function(torch.ops.aten.add.Tensor, (weight, weight))
        cold.meta["val"] = root.weight + root.weight
        cold.meta["tensor_meta"] = _extract_tensor_metadata(cold.meta["val"])
        graph.output(cold)
        graph_module = torch.fx.GraphModule(root, graph)

        class RaisingOnAttrPass(ExportPass):
            enable_fast_copy = True
            targeted_ops: tuple[()] = ()

            def on_attr(self, _attr: ProxyValue) -> None:
                raise RuntimeError("unrelated on_attr failure")

        with self.assertRaisesRegex(RuntimeError, "unrelated on_attr failure"):
            RaisingOnAttrPass()(graph_module)

    class _AddModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + x

    @staticmethod
    def _single_call_function_node(
        graph_module: torch.fx.GraphModule,
        target: torch.fx.node.Target,
    ) -> torch.fx.Node:
        matches = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target is target
        ]
        if len(matches) != 1:
            raise AssertionError(f"Expected exactly one {target} node, found {matches}")
        return matches[0]


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
            torch.testing.assert_close(new_const, original_const * 2)

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
        torch.testing.assert_close(
            result.exported_program.constants["_test_added_constant"],
            torch.tensor([1.0, 2.0, 3.0]),
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
        self.assertEqual(len(placeholder_names), 2)

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
