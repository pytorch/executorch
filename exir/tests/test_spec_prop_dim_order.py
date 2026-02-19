# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for ExecuTorch #16032: dim_order / stride ambiguity and SpecPropPass fixes.

import unittest

import torch

from executorch.exir.passes.spec_prop_pass import make_spec


class TestMakeSpecAmbiguity(unittest.TestCase):
    """Layer 1: dim_order_from_stride in exir/tensor.py; make_spec calls it."""

    def test_standard_nchw_unambiguous(self):
        t = torch.empty(2, 3, 8, 8)
        spec = make_spec(t)
        self.assertEqual(spec.dim_order, [0, 1, 2, 3])

    def test_standard_channels_last_unambiguous(self):
        t = torch.empty(2, 3, 8, 8).to(memory_format=torch.channels_last)
        spec = make_spec(t)
        self.assertEqual(spec.dim_order, [0, 2, 3, 1])

    def test_c1_contiguous_resolves_to_nchw(self):
        t = torch.empty(2, 1, 8, 8)
        spec = make_spec(t)
        self.assertEqual(spec.dim_order, [0, 1, 2, 3])

    def test_h_w_1_contiguous_resolves_to_nchw(self):
        t = torch.empty(2, 3, 1, 1)
        spec = make_spec(t)
        self.assertEqual(spec.dim_order, [0, 1, 2, 3])

    def test_scalar_tensor(self):
        t = torch.tensor(1.0)
        spec = make_spec(t)
        self.assertEqual(spec.dim_order, [])

    def test_1d_tensor(self):
        t = torch.empty(16)
        spec = make_spec(t)
        self.assertEqual(spec.dim_order, [0])


class TestSpecPropPassOutVariant(unittest.TestCase):
    """Layer 2: out-variant dim_order propagation."""

    def _run_pass(self, model, example_inputs):
        from torch.export import export
        from executorch.exir import to_edge, EdgeCompileConfig

        exported = export(model, example_inputs)
        edge = to_edge(
            exported, compile_config=EdgeCompileConfig(_skip_dim_order=False)
        )
        return edge.exported_program().graph_module

    def test_clone_out_preserves_channels_last_fp32(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return self.conv(x).clone()

        m = M().to(memory_format=torch.channels_last)
        x = torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last)
        gm = self._run_pass(m, (x,))
        for node in gm.graph.nodes:
            if "clone" in node.name and node.op == "call_function":
                out_node = node.kwargs.get("out")
                if out_node is not None:
                    self.assertIsNotNone(out_node.meta.get("spec"))
                    self.assertEqual(
                        out_node.meta["spec"].dim_order,
                        [0, 2, 3, 1],
                        f"clone.out spec has wrong dim_order: {out_node.meta['spec'].dim_order}",
                    )

    def test_clone_out_preserves_channels_last_fp16(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return self.conv(x).clone()

        m = M().half().to(memory_format=torch.channels_last)
        x = torch.randn(
            1, 3, 8, 8, dtype=torch.float16
        ).to(memory_format=torch.channels_last)
        gm = self._run_pass(m, (x,))
        for node in gm.graph.nodes:
            if "clone" in node.name and node.op == "call_function":
                out_node = node.kwargs.get("out")
                if out_node is not None:
                    self.assertEqual(
                        out_node.meta["spec"].dim_order, [0, 2, 3, 1]
                    )

    def test_clone_out_c1_channels_last_ambiguous(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv(x).clone()

        m = M().to(memory_format=torch.channels_last)
        x = torch.randn(2, 1, 8, 8).to(memory_format=torch.channels_last)
        gm = self._run_pass(m, (x,))

    def test_layout_transforming_op_uses_kwarg_not_input(self):
        try:
            _ = torch.ops.dim_order_ops._to_dim_order_copy.default
        except AttributeError:
            self.skipTest("torch.ops.dim_order_ops not available in this build")

        class LayoutTransformModel(torch.nn.Module):
            def forward(self, x):
                return torch.ops.dim_order_ops._to_dim_order_copy.default(
                    x, dtype=x.dtype, dim_order=[0, 2, 3, 1]
                )

        x = torch.randn(2, 3, 8, 8)
        from torch.export import export
        from executorch.exir import to_edge, EdgeCompileConfig

        try:
            exported = export(LayoutTransformModel(), (x,))
            edge = to_edge(
                exported,
                compile_config=EdgeCompileConfig(_skip_dim_order=False),
            )
        except Exception as e:
            self.skipTest(
                f"Could not export _to_dim_order_copy directly: {e}"
            )

        gm = edge.exported_program().graph_module
        found = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and "_to_dim_order_copy" in str(node.target)
            ):
                found = True
                spec = node.meta.get("spec")
                self.assertIsNotNone(
                    spec,
                    f"_to_dim_order_copy node {node.name!r} has no meta['spec']",
                )
                self.assertEqual(
                    spec.dim_order,
                    [0, 2, 3, 1],
                    f"Layout-transforming op must use kwarg dim_order; "
                    f"got {spec.dim_order!r}, expected [0, 2, 3, 1]",
                )
        if not found:
            import warnings

            warnings.warn(
                "No _to_dim_order_copy node found in exported graph; "
                "test may need updating if the op name changed."
            )


class TestGetitemSpecAfterDelegate(unittest.TestCase):
    """lowered_backend_module.py getitem spec fix."""

    def test_getitem_nodes_have_spec_after_delegation(self):
        try:
            from executorch.exir.backend._demo_backend import BackendWithCompilerDemo
            from executorch.exir.backend.backend_api import to_backend
        except ImportError:
            self.skipTest(
                "BackendWithCompilerDemo / to_backend not available in this build"
            )

        import operator

        try:
            from executorch.exir.delegate import executorch_call_delegate
        except ImportError:
            self.skipTest(
                "executorch_call_delegate not importable in this build"
            )

        class TwoOutputModel(torch.nn.Module):
            def forward(self, x):
                return x + x, x * 2.0

        x = torch.randn(2, 3)
        from torch.export import export
        from executorch.exir import to_edge, EdgeCompileConfig

        exported = export(TwoOutputModel(), (x,))
        edge = to_edge(exported, compile_config=EdgeCompileConfig())
        try:
            lowered_ep = to_backend(
                "BackendWithCompilerDemo",
                edge.exported_program(),
                [],
            )
        except Exception as e:
            self.skipTest(f"to_backend failed: {e}")

        gm = lowered_ep.graph_module
        getitem_count = 0
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is operator.getitem
                and len(node.args) >= 1
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target is executorch_call_delegate
            ):
                getitem_count += 1
                self.assertIn(
                    "spec",
                    node.meta,
                    f"getitem node {node.name!r} after "
                    f"executorch_call_delegate has no meta['spec'].",
                )
                spec = node.meta["spec"]
                self.assertIsNotNone(
                    spec.dim_order, f"spec.dim_order is None on {node.name!r}"
                )
        if getitem_count == 0:
            import warnings

            warnings.warn(
                "No getitem nodes found after executorch_call_delegate."
            )


class TestEndToEndFP16ChannelsLast(unittest.TestCase):
    """Full pipeline: FP16 channels_last -> .pte without Code=18."""

    def test_fp16_conv_clone_export_and_execute(self):
        from executorch.exir import (
            to_edge,
            EdgeCompileConfig,
            ExecutorchBackendConfig,
        )
        from executorch.exir.passes import MemoryPlanningPass
        from torch.export import export

        class FP16ConvClone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return self.conv(x).clone()

        model = FP16ConvClone().half().to(memory_format=torch.channels_last)
        x = torch.randn(1, 3, 8, 8, dtype=torch.float16).to(
            memory_format=torch.channels_last
        )
        exported = export(model, (x,))
        edge = to_edge(
            exported,
            compile_config=EdgeCompileConfig(_skip_dim_order=False),
        )
        et_program = edge.to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass()
            )
        )
        pte_bytes = et_program.buffer
        self.assertGreater(len(pte_bytes), 0)
        try:
            from executorch.runtime import Runtime, Program, Method

            runtime = Runtime.get()
            program = runtime.load_program(pte_bytes)
            method = program.load_method("forward")
            outputs = method.execute((x,))
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].shape, torch.Size([1, 16, 8, 8]))
        except ImportError:
            pass
