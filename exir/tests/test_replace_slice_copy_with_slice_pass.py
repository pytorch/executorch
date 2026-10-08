# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import List

import torch
from executorch.exir import ExecutorchBackendConfig, memory, to_edge
from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
from executorch.exir.passes.normalize_view_copy_base_pass import (
    NormalizeViewCopyBasePass,
)
from executorch.exir.passes.reinplace import reinplace_pass
from executorch.exir.passes.replace_slice_copy_with_slice_pass import (
    _compute_slice_byte_offset,
    _is_slice_copy,
    _is_static_slice_argument,
    is_contiguous_slice_copy,
    ReplaceSliceCopyWithSlicePass,
)
from executorch.exir.passes.replace_view_copy_with_view_pass import (
    _ViewSpec,
    ReplaceViewCopyWithViewPass,
)
from executorch.exir.passes.spec_prop_pass import SpecPropPass
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

    def test_pass_normalizes_negative_start(self) -> None:

        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[-2:] + 1.0

        gm = self._edge_graph_module(M(), (torch.randn(4, 8),))
        self._annotate_tensor_specs(gm)

        result = ReplaceSliceCopyWithSlicePass()(gm)
        self.assertTrue(result.modified)
        sliced = next(n for n in gm.graph.nodes if n.target == memory.slice)
        self.assertEqual(sliced.args[2:4], (2, 4))
        base_spec = sliced.args[0].meta["spec"]
        base_spec.mem_offset = 128
        self.assertEqual(sliced.meta["spec"].mem_offset, 128 + 2 * 8 * 4)
        self.assertEqual(_compute_slice_byte_offset(base_spec, 0, -2), 2 * 8 * 4)

    def test_clamped_bounds_and_empty_slices(self) -> None:
        class M(torch.nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                base = x + 0.0
                return base[self.start : self.end] + 1.0

        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        for start, end, expected_alias in (
            (2, 99, True),
            (-99, 2, True),
            (-3, -1, True),
            (None, None, True),
            (99, 100, False),
            (3, 1, False),
            (0, -99, False),
        ):
            with self.subTest(start=start, end=end):
                model = M(start, end)
                ep = to_edge(export(model, (x,), strict=True)).exported_program()
                gm = ep.graph_module
                self._annotate_tensor_specs(gm)
                result = ReplaceSliceCopyWithSlicePass()(gm)
                # Export may remove a no-op full slice.
                if start is not None or end is not None:
                    self.assertEqual(result.modified, expected_alias)
                assert_close(ep.module()(x), model(x))
                for node in gm.graph.nodes:
                    if node.target == memory.slice:
                        base_spec = node.args[0].meta["spec"]
                        base_spec.mem_offset = 256
                        spec = node.meta["spec"]
                        self.assertGreaterEqual(spec.mem_offset, 256)
                        self.assertLessEqual(
                            spec.mem_offset + spec.nbytes(), 256 + base_spec.nbytes()
                        )
                et = to_edge(export(model, (x,), strict=True)).to_executorch()
                runtime = _load_for_executorch_from_buffer(et.buffer)
                assert_close(runtime.forward((x,))[0], model(x))

    def test_shared_view_spec_byte_range(self) -> None:
        base = TensorSpec.from_tensor(torch.empty(4, 8))
        sliced = _ViewSpec(base, [2, 8], byte_offset=32)
        viewed = _ViewSpec(sliced, [16])
        self.assertIsNone(viewed.mem_offset)
        base.mem_offset = 128
        base.mem_id = 1
        self.assertEqual(sliced.mem_offset, 160)
        self.assertEqual(viewed.mem_offset, 160)
        self.assertEqual(viewed.mem_id, 1)
        base.mem_offset = 256
        self.assertEqual(viewed.mem_offset, 288)
        for offset in (-1, 100):
            with self.subTest(offset=offset), self.assertRaises(Exception):
                _ViewSpec(base, [2, 8], byte_offset=offset)

    def test_reinplace_mutation_safety(self) -> None:
        class M(torch.nn.Module):
            def __init__(self, mutate_slice, other_is_live):
                super().__init__()
                self.mutate_slice = mutate_slice
                self.other_is_live = other_is_live

            def forward(self, x, indices, values):
                base = torch.relu(x)
                sliced = torch.ops.aten.slice_copy.Tensor(base, 0, 1, 3)
                # Include a downstream view so alias families span both passes.
                viewed = sliced.view(2, 8)
                if self.mutate_slice:
                    changed = torch.ops.aten.index_put.default(
                        viewed, [indices], values
                    )
                    return (base, changed) if self.other_is_live else changed
                if self.other_is_live:
                    changed = torch.ops.aten.index_put.default(base, [indices], values)
                    return changed, viewed.clone()
                observed = viewed.clone()
                changed = torch.ops.aten.index_put.default(base, [indices], values)
                return observed, changed

        inputs = (
            torch.arange(32, dtype=torch.float32).reshape(4, 8),
            torch.tensor([1]),
            torch.full((1, 8), 100.0),
        )
        for mutate_slice in (False, True):
            for other_is_live in (False, True):
                with self.subTest(mutate_slice=mutate_slice, live=other_is_live):
                    model = M(mutate_slice, other_is_live)
                    expected = model(*copy.deepcopy(inputs))
                    ep = to_edge(export(model, inputs, strict=True)).exported_program()
                    reinplace_pass(ep)
                    gm = SpecPropPass()(ep.graph_module).graph_module
                    NormalizeViewCopyBasePass()(gm)
                    ReplaceViewCopyWithViewPass()(gm)
                    ReplaceSliceCopyWithSlicePass()(gm)
                    aliases = [n for n in gm.graph.nodes if n.target == memory.slice]
                    self.assertEqual(bool(aliases), not other_is_live)
                    actual = gm(*copy.deepcopy(inputs))
                    assert_close(
                        actual, expected if isinstance(expected, tuple) else (expected,)
                    )
                    ToOutVarPass()(gm)
                    MemoryPlanningPass()(gm)
                    program = to_edge(
                        export(model, copy.deepcopy(inputs), strict=True)
                    ).to_executorch(ExecutorchBackendConfig(run_reinplace_pass=True))
                    runtime = _load_for_executorch_from_buffer(program.buffer)
                    assert_close(
                        tuple(runtime.forward(copy.deepcopy(inputs))),
                        expected if isinstance(expected, tuple) else (expected,),
                    )

    def test_view_of_slice_uses_updated_spec(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                base = x + 0.0
                return base[1:3].view(16) + 1.0

        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        model = M()
        gm = self._edge_graph_module(model, (x,))
        self._annotate_tensor_specs(gm)
        NormalizeViewCopyBasePass()(gm)
        ReplaceViewCopyWithViewPass()(gm)
        ReplaceSliceCopyWithSlicePass()(gm)
        sliced = next(n for n in gm.graph.nodes if n.target == memory.slice)
        viewed = next(n for n in gm.graph.nodes if n.target == memory.view)
        self.assertIs(viewed.meta["spec"]._base, sliced.meta["spec"])
        sliced.args[0].meta["spec"].mem_offset = 128
        self.assertEqual(viewed.meta["spec"].mem_offset, 160)
        et = to_edge(export(model, (x,), strict=True)).to_executorch()
        runtime = _load_for_executorch_from_buffer(et.buffer)
        assert_close(runtime.forward((x,))[0], model(x))

    def test_sibling_slice_aliases_preserve_mutation_safety(self) -> None:
        for annotated in (False, True):
            with self.subTest(allocation_sharing_annotation=annotated):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                base = graph.call_function(torch.ops.aten.relu.default, (x,))
                first = graph.call_function(
                    torch.ops.aten.slice_copy.Tensor, (base, 0, 1, 3)
                )
                second = graph.call_function(
                    torch.ops.aten.slice_copy.Tensor, (base, 0, 1, 3)
                )
                changed = graph.call_function(
                    (
                        torch.ops.aten.add.Tensor
                        if annotated
                        else torch.ops.aten.add_.Tensor
                    ),
                    (first, 10),
                )
                if annotated:
                    changed.meta["_share_alloc_with_arg_idx"] = 0
                observed = graph.call_function(torch.ops.aten.clone.default, (second,))
                graph.output((changed, observed))
                for node in (x, base):
                    node.meta["val"] = torch.empty(4, 8)
                for node in (first, second, changed, observed):
                    node.meta["val"] = torch.empty(2, 8)
                gm = torch.fx.GraphModule(torch.nn.Module(), graph)
                self._annotate_tensor_specs(gm)
                inputs = torch.arange(32, dtype=torch.float32).reshape(4, 8)
                expected = gm(inputs)
                ReplaceSliceCopyWithSlicePass()(gm)
                self.assertTrue(_is_slice_copy(first))
                self.assertEqual(second.target, memory.slice)
                assert_close(gm(inputs), expected)

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
        sliced = graph.call_function(torch.ops.aten.slice_copy.Tensor, (x, 0, 1, 3))
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
