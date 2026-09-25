# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from executorch.exir import memory, to_edge
from executorch.exir.passes.replace_slice_copy_with_slice_pass import (
    _compute_slice_byte_offset,
    _is_slice_copy,
    _is_static_slice_argument,
    is_contiguous_slice_copy,
    ReplaceSliceCopyWithSlicePass,
)
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export
from torch.testing import assert_close


class TestReplaceSliceCopyWithSlicePass(unittest.TestCase):
    def _edge_graph_module(
        self, module: torch.nn.Module, inputs: tuple
    ) -> torch.fx.GraphModule:
        ep = export(module.eval(), inputs, strict=True)
        return to_edge(ep).exported_program().graph_module

    def test_contiguity_classification(self) -> None:
        """A unit-step slice along the outermost dim is contiguous (eligible);
        inner-dim or strided slices are not."""

        class M(torch.nn.Module):
            def forward(self, x):
                a = x[0:2]  # dim 0, step 1 -> contiguous  (eligible)
                b = x[:, 1:3]  # dim 1        -> strided     (not eligible)
                c = x[0:4:2]  # dim 0, step 2 -> strided     (not eligible)
                return a.sum() + b.sum() + c.sum()

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        slice_nodes = [n for n in gm.graph.nodes if _is_slice_copy(n)]
        eligible = [n for n in slice_nodes if is_contiguous_slice_copy(n)]

        self.assertEqual(len(slice_nodes), 3)
        self.assertEqual(len(eligible), 1)

    def test_negative_outermost_dim_is_contiguous(self) -> None:
        """A negative dim that resolves to the outermost dim is still eligible."""

        class M(torch.nn.Module):
            def forward(self, x):
                # dim=-2 on a rank-2 tensor resolves to dim 0.
                return torch.ops.aten.slice_copy.Tensor(x, -2, 0, 2).sum()

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        eligible = [n for n in gm.graph.nodes if is_contiguous_slice_copy(n)]
        self.assertEqual(len(eligible), 1)

    def _annotate_input_spec(self, gm: torch.fx.GraphModule) -> None:
        """Populate ``spec`` on every tensor placeholder.

        The lowering pipeline normally does this before the pass runs.  Note
        that ``to_edge`` lifts scalar constants to their own placeholders, so
        annotating only the first placeholder would miss the real input.
        """
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                node.meta["spec"] = TensorSpec.from_tensor(val)

    def _annotate_tensor_specs(self, gm: torch.fx.GraphModule) -> None:
        """Populate static specs for all tensor nodes in a small FX test graph."""
        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                node.meta["spec"] = TensorSpec.from_tensor(val)

    def test_pass_replaces_annotated_contiguous_slice(self) -> None:
        """A statically annotated dim-0 slice becomes a memory alias."""

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[0:2] + 1.0

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self._annotate_tensor_specs(gm)
        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertIsNotNone(result)
        self.assertTrue(result.modified)
        self.assertEqual(
            len(
                [
                    n
                    for n in result.graph_module.graph.nodes
                    if n.op == "call_function" and n.target == memory.slice
                ]
            ),
            1,
        )

    def test_pass_skips_nondefault_base_dim_order(self) -> None:
        """Avoid aliases that would reinterpret a non-contiguous base layout."""

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[0:2] + 1.0

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self._annotate_tensor_specs(gm)
        # Mutate the layout of the slice's own base, not just any placeholder.
        slice_node = next(n for n in gm.graph.nodes if _is_slice_copy(n))
        slice_node.args[0].meta["spec"].dim_order = (1, 0)

        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertFalse(result.modified)

    def test_pass_skips_negative_start(self) -> None:
        """Negative starts need shape-dependent normalization, so keep copying."""

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[-2:] + 1.0

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self._annotate_tensor_specs(gm)

        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertFalse(result.modified)
        with self.assertRaises(ValueError):
            _compute_slice_byte_offset(
                next(n for n in gm.graph.nodes if n.op == "placeholder").meta["spec"],
                0,
                -2,
            )

    def test_static_slice_argument_check_rejects_runtime_nodes(self) -> None:
        """Runtime graph values cannot be encoded as fixed alias offsets."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        start = graph.placeholder("start")
        base = graph.call_function(torch.ops.aten.relu.default, (x,))
        sliced = graph.call_function(
            torch.ops.aten.slice_copy.Tensor, (base, 0, start, 2)
        )
        relu = graph.call_function(torch.ops.aten.relu.default, (sliced,))
        graph.output(relu)

        x.meta["val"] = torch.empty(4, 8)
        x.meta["spec"] = TensorSpec.from_tensor(x.meta["val"])
        base.meta["val"] = torch.empty(4, 8)
        base.meta["spec"] = TensorSpec.from_tensor(base.meta["val"])
        sliced.meta["val"] = torch.empty(2, 8)
        sliced.meta["spec"] = TensorSpec.from_tensor(sliced.meta["val"])
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        self.assertFalse(_is_static_slice_argument(start))
        self.assertTrue(_is_static_slice_argument(1))
        self.assertTrue(_is_static_slice_argument(None))
        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertFalse(result.modified)
        self.assertEqual(sliced.target, torch.ops.aten.slice_copy.Tensor)

    def test_pass_skips_dynamic_output_shape(self) -> None:
        """A dynamic slice must retain its copy kernel even with a static base."""

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[0:2] + 1.0

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self._annotate_tensor_specs(gm)
        slice_node = next(n for n in gm.graph.nodes if _is_slice_copy(n))
        slice_node.meta["spec"].shape_dynamism = TensorShapeDynamism.DYNAMIC_BOUND

        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertFalse(result.modified)

    def test_pass_skips_placeholder_base(self) -> None:
        """External input tensors have no memory-planned allocation to alias."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        sliced = graph.call_function(
            torch.ops.aten.slice_copy.Tensor, (x, 0, 1, 3)
        )
        relu = graph.call_function(torch.ops.aten.relu.default, (sliced,))
        graph.output(relu)
        x.meta["val"] = torch.empty(4, 8)
        x.meta["spec"] = TensorSpec.from_tensor(x.meta["val"])
        sliced.meta["val"] = torch.empty(2, 8)
        sliced.meta["spec"] = TensorSpec.from_tensor(sliced.meta["val"])
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertFalse(result.modified)
        self.assertEqual(sliced.target, torch.ops.aten.slice_copy.Tensor)

    def _emitted_operators(self, program) -> List[str]:
        return [
            str(op) for op in program.executorch_program.execution_plan[0].operators
        ]

    def test_lowered_program_matches_eager_output(self) -> None:
        """The emitted sub-buffer alias executes with the original semantics."""

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[1:3] + 1.0

        model = M().eval()
        example_input = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        et = to_edge(export(model, (example_input,), strict=True)).to_executorch()

        # The slice must be aliased away, not merely produce the right answer --
        # falling back to a copy would also pass a numerical check alone.
        self.assertFalse(
            any("slice_copy" in op for op in self._emitted_operators(et)),
            "expected the contiguous slice to be elided, but slice_copy was emitted",
        )

        runtime_module = _load_for_executorch_from_buffer(et.buffer)
        assert_close(runtime_module.forward((example_input,))[0], model(example_input))

    def test_base_outlives_slice_when_reused(self) -> None:
        """The base buffer must not be reused while the alias is still live."""

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                sliced = base[1:3] + 1.0
                # ``base`` is consumed *after* the slice, so the planner has to keep
                # the base alive across the alias's lifetime.
                return sliced.sum() + base.sum()

        model = M().eval()
        example_input = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        et = to_edge(export(model, (example_input,), strict=True)).to_executorch()
        runtime_module = _load_for_executorch_from_buffer(et.buffer)

        assert_close(runtime_module.forward((example_input,))[0], model(example_input))

    def test_chained_slice_falls_back_to_copy(self) -> None:
        """A slice of a slice has no concrete base allocation to offset from."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x[0:3][1:2] + 1.0

        model = M().eval()
        example_input = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        # Must lower and execute correctly rather than tripping over an
        # aliasing base during memory planning.
        et = to_edge(export(model, (example_input,), strict=True)).to_executorch()
        runtime_module = _load_for_executorch_from_buffer(et.buffer)

        assert_close(runtime_module.forward((example_input,))[0], model(example_input))

    def test_non_slice_nodes_are_ignored(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return (x + 1.0).relu()

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self.assertEqual([n for n in gm.graph.nodes if is_contiguous_slice_copy(n)], [])


if __name__ == "__main__":
    unittest.main()
