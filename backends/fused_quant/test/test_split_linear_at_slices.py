# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any, Optional, TypeAlias

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import get_constant
from executorch.backends.fused_quant.optimization_passes.constant_fold import (
    ConstantFold,
)
from executorch.backends.fused_quant.optimization_passes.split_linear_at_slices import (
    SplitLinearAtSlices,
)
from executorch.backends.fused_quant.test.helpers import (
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ProxyValue
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_QBlock: TypeAlias = tuple[Any, Any, torch.dtype, int, int]


class SplitLinearAtSlicesTest(unittest.TestCase):
    """Tests for the SplitLinearAtSlices optimization pass."""

    def _build_linear_with_slices(
        self,
        in_features: int = 8,
        out_features: int = 16,
        slices: list[tuple[int, int, int]] | None = None,
        with_bias: bool = True,
    ) -> tuple[ExportedProgram, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build a linear → (slice, slice, ...) graph.

        Returns (program, input_tensor, weight_tensor, bias_tensor).
        """

        if slices is None:
            slices = [(0, out_features // 2, 1), (out_features // 2, out_features, 1)]

        builder = ProgramBuilder()
        inp_tensor = torch.randn(1, in_features)
        weight_tensor = torch.randint(
            -16, 16, (out_features, in_features), dtype=torch.int8
        )
        bias_tensor = torch.randn(out_features) if with_bias else None

        x = builder.placeholder("x", inp_tensor)
        w = builder.placeholder("weight", weight_tensor, input_kind=InputKind.BUFFER)
        b = (
            builder.placeholder("bias", bias_tensor, input_kind=InputKind.BUFFER)
            if bias_tensor is not None
            else None
        )

        weight_qparams = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        out_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                b,
                None,
                None,
                torch.float32,
                0,
                0,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        outputs = []
        for start, end, step in slices:
            sliced = builder.call_operator(
                op=exir_ops.edge.aten.slice_copy.Tensor,
                args=(linear, -1, start, end, step),
            )
            outputs.append(sliced)

        builder.output(outputs)
        return builder.get_program(), inp_tensor, weight_tensor, bias_tensor

    def _assert_signature_consistent(self, ep: ExportedProgram) -> None:
        """Assert that the graph signature, graph placeholders, and state_dict are in sync."""
        sig = ep.graph_signature
        placeholders = [n for n in ep.graph_module.graph.nodes if n.op == "placeholder"]
        placeholder_names = {n.name for n in placeholders}

        spec_names = {s.arg.name for s in sig.input_specs}
        self.assertEqual(
            placeholder_names,
            spec_names,
            "Placeholder nodes and input_specs should match",
        )

        for spec in sig.input_specs:
            if spec.kind == InputKind.BUFFER:
                self.assertIn(
                    spec.target,
                    ep.state_dict,
                    f"Buffer {spec.target} missing from state_dict",
                )

    def _run_program(
        self, ep: ExportedProgram, inp: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run a program and return all outputs."""
        full_inputs = []
        sig = ep.graph_signature
        for spec in sig.input_specs:
            if spec.kind == InputKind.USER_INPUT:
                full_inputs.append(inp)
            else:
                assert spec.target is not None
                # Params/buffers live in state_dict; constant-folded tensors
                # (e.g. from ConstantFold) live in constants.
                if spec.target in ep.state_dict:
                    full_inputs.append(ep.state_dict[spec.target])
                else:
                    full_inputs.append(ep.constants[spec.target])
        out = ep.graph_module(*full_inputs)
        if isinstance(out, torch.Tensor):
            return [out]
        return list(out)

    def test_split_two_contiguous_slices(self) -> None:
        """linear → (slice[0:8], slice[8:16]) splits into two smaller linears."""
        ep, inp, _, _ = self._build_linear_with_slices(slices=[(0, 8, 1), (8, 16, 1)])
        before_outputs = self._run_program(ep, inp)

        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 2, "Should have 2 split linears")

        slice_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.slice_copy.Tensor,
        )
        self.assertEqual(len(slice_nodes), 0, "Slice nodes should be removed")

        self._assert_signature_consistent(result.exported_program)

        after_outputs = self._run_program(result.exported_program, inp)
        for before, after in zip(before_outputs, after_outputs):
            self.assertTrue(
                torch.equal(before, after),
                f"Numerical mismatch: {before} vs {after}",
            )

    def test_split_strided_even_odd(self) -> None:
        """linear → (slice[0::2], slice[1::2]) splits into even/odd rows."""
        ep, inp, _, _ = self._build_linear_with_slices(slices=[(0, 16, 2), (1, 16, 2)])
        before_outputs = self._run_program(ep, inp)

        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 2, "Should have 2 split linears")

        self._assert_signature_consistent(result.exported_program)

        after_outputs = self._run_program(result.exported_program, inp)
        for before, after in zip(before_outputs, after_outputs):
            self.assertTrue(
                torch.equal(before, after),
                f"Numerical mismatch: {before} vs {after}",
            )

    def test_split_no_bias(self) -> None:
        """Split works correctly without bias."""
        ep, inp, _, _ = self._build_linear_with_slices(
            slices=[(0, 8, 1), (8, 16, 1)], with_bias=False
        )
        before_outputs = self._run_program(ep, inp)

        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)

        self._assert_signature_consistent(result.exported_program)

        after_outputs = self._run_program(result.exported_program, inp)
        for before, after in zip(before_outputs, after_outputs):
            self.assertTrue(
                torch.equal(before, after),
                f"Numerical mismatch: {before} vs {after}",
            )

    def test_no_split_overlapping_slices(self) -> None:
        """Overlapping slices exceed out_features — should not split."""
        ep, _, _, _ = self._build_linear_with_slices(slices=[(0, 12, 1), (4, 16, 1)])

        result = SplitLinearAtSlices().call(ep)
        self.assertFalse(
            result.modified,
            "Should not split when overlapping slices exceed out_features",
        )

    def test_split_single_slice(self) -> None:
        """A single slice on the output dim replaces the linear with a smaller one."""
        ep, inp, _, _ = self._build_linear_with_slices(
            slices=[(0, 8, 1)],
        )
        before_outputs = self._run_program(ep, inp)

        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1, "Should have 1 smaller linear")

        slice_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.slice_copy.Tensor,
        )
        self.assertEqual(len(slice_nodes), 0, "Slice node should be removed")

        self._assert_signature_consistent(result.exported_program)

        after_outputs = self._run_program(result.exported_program, inp)
        for before, after in zip(before_outputs, after_outputs):
            self.assertTrue(
                torch.equal(before, after),
                f"Numerical mismatch: {before} vs {after}",
            )

    def test_split_negative_start_end(self) -> None:
        """Negative start/end are normalized so range() and slicing agree."""
        # slice[-4:, :] is the last 4 rows, slice[:-4, :] is the first 12 rows
        ep, inp, _, _ = self._build_linear_with_slices(slices=[(0, -4, 1), (-4, 16, 1)])
        before_outputs = self._run_program(ep, inp)

        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 2, "Should have 2 split linears")

        slice_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.slice_copy.Tensor,
        )
        self.assertEqual(len(slice_nodes), 0, "Slice nodes should be removed")

        self._assert_signature_consistent(result.exported_program)

        after_outputs = self._run_program(result.exported_program, inp)
        for before, after in zip(before_outputs, after_outputs):
            self.assertTrue(
                torch.equal(before, after),
                f"Numerical mismatch: {before} vs {after}",
            )

    def test_no_split_non_output_dim_slice(self) -> None:
        """Slices on the batch dimension (dim=0) should not trigger split."""

        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(4, 8))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (16, 8), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )

        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(builder, 16, dtype=torch.float32)
        out_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                None,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        slice0 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(linear, 0, 0, 2, 1),
        )
        slice1 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(linear, 0, 2, 4, 1),
        )
        builder.output([slice0, slice1])

        ep = builder.get_program()
        result = SplitLinearAtSlices().call(ep)
        self.assertFalse(
            result.modified, "Should not split when slices are on batch dimension"
        )

    # ---- helpers for the qparam-handling tests below ----

    def _make_linear(
        self,
        builder: ProgramBuilder,
        x: ProxyValue,
        w: ProxyValue,
        b: Optional[ProxyValue],
        weight_block: _QBlock,
        out_block: _QBlock,
        bias_block: Optional[_QBlock] = None,
    ) -> ProxyValue:
        """Assemble a fused_quant.linear with per-tensor (None) input qparams."""
        if bias_block is None:
            bias_block = (None, None, torch.float32, 0, 0)
        return builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                w,
                b,
                None,
                None,
                torch.float32,
                0,
                0,
                *weight_block,
                *bias_block,
                *out_block,
            ),
        )

    def _add_slices(
        self,
        builder: ProgramBuilder,
        linear: ProxyValue,
        slices: list[tuple[int, int, int]],
    ) -> None:
        outs = []
        for start, end, step in slices:
            outs.append(
                builder.call_operator(
                    op=exir_ops.edge.aten.slice_copy.Tensor,
                    args=(linear, -1, start, end, step),
                )
            )
        builder.output(outs)

    def test_split_slices_varying_per_channel_qparams(self) -> None:
        """Per-channel weight scale/zp that VARY per output channel are sliced to
        match each split linear. (The uniform-scale tests above would not catch a
        mis-slice, since every channel shares the same value.)"""
        out_features, in_features = 16, 8
        builder = ProgramBuilder()
        inp = torch.randn(1, in_features)
        x = builder.placeholder("x", inp)
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        ws = builder.placeholder(
            "wt_scale",
            (torch.arange(out_features, dtype=torch.float32) * 0.01 + 0.02).reshape(
                out_features, 1
            ),
            input_kind=InputKind.BUFFER,
        )
        wzp = builder.placeholder(
            "wt_zp",
            (torch.arange(out_features, dtype=torch.int64) % 5).reshape(
                out_features, 1
            ),
            input_kind=InputKind.BUFFER,
        )
        out_block = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)
        # weight qparam "dtype" is the dequantize OUTPUT dtype (float); the int8
        # weight is stored in the tensor, its range is in quant_min/max.
        linear = self._make_linear(
            builder, x, w, None, (ws, wzp, torch.float32, -128, 127), out_block
        )
        slices = [(0, 8, 1), (8, 16, 1)]
        self._add_slices(builder, linear, slices)
        ep = builder.get_program()

        before = self._run_program(ep, inp)
        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)
        self._assert_signature_consistent(result.exported_program)

        after = self._run_program(result.exported_program, inp)
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a), f"Numerical mismatch: {b} vs {a}")

        # Each split linear must carry a sliced (not full-rank) scale/zp.
        linears = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linears), 2)
        for lin in linears:
            ws_node = get_arg(lin, "weight_scale", torch.fx.Node)
            sliced = get_constant(result.exported_program, ws_node)
            assert sliced is not None
            self.assertEqual(list(sliced.shape), [8, 1])

    def test_split_unquantized_weight_no_qparams(self) -> None:
        """A linear with no weight scale/zp (unquantized weight) still splits."""
        out_features, in_features = 16, 8
        builder = ProgramBuilder()
        inp = torch.randn(1, in_features)
        x = builder.placeholder("x", inp)
        w = builder.placeholder(
            "weight",
            torch.randn(out_features, in_features),
            input_kind=InputKind.BUFFER,
        )
        out_block = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)
        linear = self._make_linear(
            builder, x, w, None, (None, None, torch.float32, 0, 0), out_block
        )
        self._add_slices(builder, linear, [(0, 8, 1), (8, 16, 1)])
        ep = builder.get_program()

        before = self._run_program(ep, inp)
        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified, "no-qparam weight should still split")
        self._assert_signature_consistent(result.exported_program)

        linears = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linears), 2)

        after = self._run_program(result.exported_program, inp)
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a), f"Numerical mismatch: {b} vs {a}")

    def test_split_slices_quantized_bias_qparams(self) -> None:
        """A per-output-channel quantized bias has its bias_scale/zp sliced too,
        not just the bias tensor."""
        out_features, in_features = 16, 8
        builder = ProgramBuilder()
        inp = torch.randn(1, in_features)
        x = builder.placeholder("x", inp)
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        b = builder.placeholder(
            "bias",
            torch.randint(-8, 8, (out_features,), dtype=torch.int32),
            input_kind=InputKind.BUFFER,
        )
        weight_block = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        bias_scale = builder.placeholder(
            "bias_scale",
            torch.arange(out_features, dtype=torch.float32) * 0.01 + 0.05,
            input_kind=InputKind.BUFFER,
        )
        bias_zp = builder.placeholder(
            "bias_zp",
            torch.zeros(out_features, dtype=torch.int64),
            input_kind=InputKind.BUFFER,
        )
        out_block = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)
        linear = self._make_linear(
            builder,
            x,
            w,
            b,
            weight_block,
            out_block,
            bias_block=(bias_scale, bias_zp, torch.float32, -(2**31), 2**31 - 1),
        )
        slices = [(0, 8, 1), (8, 16, 1)]
        self._add_slices(builder, linear, slices)
        ep = builder.get_program()

        before = self._run_program(ep, inp)
        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)
        self._assert_signature_consistent(result.exported_program)

        after = self._run_program(result.exported_program, inp)
        for b_out, a_out in zip(before, after):
            self.assertTrue(
                torch.equal(b_out, a_out), f"Numerical mismatch: {b_out} vs {a_out}"
            )

        linears = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linears), 2)
        for lin in linears:
            bs_node = get_arg(lin, "bias_scale", torch.fx.Node)
            sliced_bs = get_constant(result.exported_program, bs_node)
            assert sliced_bs is not None
            self.assertEqual(list(sliced_bs.shape), [8], "bias_scale must be sliced")

    def test_per_tensor_weight_scale_reused_not_sliced(self) -> None:
        """A per-tensor (scalar) weight scale broadcasts, so it is reused/shared
        across the split linears rather than sliced."""
        out_features, in_features = 16, 8
        builder = ProgramBuilder()
        inp = torch.randn(1, in_features)
        x = builder.placeholder("x", inp)
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        # Per-tensor scalar weight qparams; dtype is the dequant output dtype
        # (float), the int8 weight range is in quant_min/max.
        weight_block = create_per_tensor_qparams(
            builder, scale=0.05, dtype=torch.float32
        )
        out_block = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)
        linear = self._make_linear(builder, x, w, None, weight_block, out_block)
        self._add_slices(builder, linear, [(0, 8, 1), (8, 16, 1)])
        ep = builder.get_program()

        before = self._run_program(ep, inp)
        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)

        linears = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linears), 2)
        ws_nodes = {get_arg(lin, "weight_scale", torch.fx.Node) for lin in linears}
        self.assertEqual(
            len(ws_nodes), 1, "per-tensor weight scale should be shared, not sliced"
        )

        after = self._run_program(result.exported_program, inp)
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a), f"Numerical mismatch: {b} vs {a}")

    def _build_linear_with_scale_behind_view(
        self, out_features: int = 16, in_features: int = 8
    ) -> tuple[ExportedProgram, torch.Tensor]:
        """Build a split graph whose per-channel weight scale/zp ride behind an
        ``aten.view_copy`` (the shape fusion inserts), so they are NOT directly
        resolvable constants until folded."""
        builder = ProgramBuilder()
        inp = torch.randn(1, in_features)
        x = builder.placeholder("x", inp)
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        ws_1d = builder.placeholder(
            "wt_scale_1d",
            torch.arange(out_features, dtype=torch.float32) * 0.01 + 0.02,
            input_kind=InputKind.BUFFER,
        )
        wzp_1d = builder.placeholder(
            "wt_zp_1d",
            torch.arange(out_features, dtype=torch.int64) % 5,
            input_kind=InputKind.BUFFER,
        )
        ws_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(ws_1d, [out_features, 1])
        )
        wzp_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(wzp_1d, [out_features, 1])
        )
        out_block = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)
        linear = self._make_linear(
            builder,
            x,
            w,
            None,
            (ws_view, wzp_view, torch.float32, -128, 127),
            out_block,
        )
        self._add_slices(builder, linear, [(0, 8, 1), (8, 16, 1)])
        return builder.get_program(), inp

    def test_no_split_per_channel_out_qparams(self) -> None:
        """Per-channel OUTPUT qparams (out_scale varying along the output dim) are
        on a different axis than the weight/bias qparams and aren't handled, so the
        split bails rather than reuse a full-rank out_scale on a smaller linear."""
        out_features, in_features = 16, 8
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, in_features))
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (out_features, in_features), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        weight_block = create_per_channel_qparams(
            builder, out_features, scale=0.05, dtype=torch.float32
        )
        # Per-channel (non-scalar) output scale/zp.
        out_scale = builder.placeholder(
            "out_scale",
            torch.full((out_features,), 0.5, dtype=torch.float32),
            input_kind=InputKind.BUFFER,
        )
        out_zp = builder.placeholder(
            "out_zp",
            torch.zeros(out_features, dtype=torch.int64),
            input_kind=InputKind.BUFFER,
        )
        linear = self._make_linear(
            builder,
            x,
            w,
            None,
            weight_block,
            (out_scale, out_zp, torch.int8, -128, 127),
        )
        self._add_slices(builder, linear, [(0, 8, 1), (8, 16, 1)])
        ep = builder.get_program()

        result = SplitLinearAtSlices().call(ep)
        self.assertFalse(result.modified, "Should bail on per-channel output qparams")

    def test_no_split_when_per_channel_qparam_behind_view(self) -> None:
        """If a per-output weight scale is still behind a view_copy (not yet folded
        to a constant), the split aborts -- it must not slice the weight while
        leaving a mismatched full-rank scale."""
        ep, _ = self._build_linear_with_scale_behind_view()
        result = SplitLinearAtSlices().call(ep)
        self.assertFalse(
            result.modified,
            "Should abort when a per-output qparam is not a resolvable constant",
        )

    def test_split_after_constant_fold_resolves_view(self) -> None:
        """ConstantFold bakes the weight scale/zp view_copy, after which the split
        slices them correctly -- mirroring the real pipeline ordering."""
        ep, inp = self._build_linear_with_scale_behind_view()
        before = self._run_program(ep, inp)

        ep = ConstantFold().call(ep).exported_program
        result = SplitLinearAtSlices().call(ep)
        self.assertTrue(result.modified)
        self._assert_signature_consistent(result.exported_program)

        after = self._run_program(result.exported_program, inp)
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a), f"Numerical mismatch: {b} vs {a}")

        linears = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        self.assertEqual(len(linears), 2)
        for lin in linears:
            ws_node = get_arg(lin, "weight_scale", torch.fx.Node)
            sliced = get_constant(result.exported_program, ws_node)
            assert sliced is not None
            self.assertEqual(list(sliced.shape), [8, 1])
