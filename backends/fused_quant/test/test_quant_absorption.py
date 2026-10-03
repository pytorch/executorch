# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
import unittest
from typing import Optional

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    get_input_kind,
    get_qparams_from_node,
    get_scale,
    get_zero_point,
)
from executorch.backends.fused_quant.optimization_passes.quant_absorption import (
    QuantAbsorptionPass,
)
from executorch.backends.fused_quant.test.helpers import (
    create_per_axis_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


class QuantAbsorptionTest(unittest.TestCase):
    """Tests for the QuantAbsorptionPass."""

    def _build_fused_linear_with_dequant_quant(
        self,
        out_scale: float = 0.5,
        out_zp: int = 0,
        dequant_scale: Optional[float] = None,
        dequant_zp: Optional[int] = None,
        downstream_scale: float = 0.3,
        downstream_zp: int = 2,
    ) -> ExportedProgram:
        """Build: fused_quant.linear(Q(A)) → DQ(D) → Q(B)."""

        dequant_scale = out_scale if dequant_scale is None else dequant_scale
        dequant_zp = out_zp if dequant_zp is None else dequant_zp

        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))

        out_qparams = create_per_tensor_qparams(
            builder, scale=out_scale, zero_point=out_zp, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(linear,),
            kwargs={
                "scale": dequant_scale,
                "zero_point": dequant_zp,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": downstream_scale,
                "zero_point": downstream_zp,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        builder.output([quant])
        return builder.get_program()

    def test_absorb_mints_constant_tensor_qparams_no_cross_kind_clash(self) -> None:
        """The out qparams QuantAbsorption mints must be the SAME kind as the
        fusion's per-tensor _scale/_zero_point (lifted CONSTANT_TENSORs). Minting
        them as a different kind (e.g. BUFFER) gives a b__scale node whose fqn
        '_scale' still collides with the fusion's ep.constants entry -- a
        cross-kind clash add_constant's node-name uniquification can't catch. Seed
        the fusion-style constants and assert the absorb still succeeds and lands
        its qparams in the same store."""
        ep = self._build_fused_linear_with_dequant_quant(
            out_scale=0.5, out_zp=0, downstream_scale=0.3, downstream_zp=2
        )
        linear = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.fused_quant.linear.default
        )
        # Seed the exact fqns QuantAbsorption will mint, as CONSTANT_TENSOR --
        # exactly what FusedQuantFusion leaves for a per-tensor qparam.
        add_constant(ep, "_scale", torch.tensor(1.0), linear, InputKind.CONSTANT_TENSOR)
        add_constant(
            ep,
            "_zero_point",
            torch.tensor(0, dtype=torch.int64),
            linear,
            InputKind.CONSTANT_TENSOR,
        )

        # A BUFFER mint would raise AssertionError (fqn '_scale' already backed).
        result = QuantAbsorptionPass().call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        fused = next(
            n
            for n in result.exported_program.graph.nodes
            if n.op == "call_function"
            and n.target == exir_ops.edge.fused_quant.linear.default
        )
        out_qp = get_qparams_from_node(fused, "out")
        assert out_qp is not None
        self.assertEqual(
            get_input_kind(result.exported_program, out_qp.scale),
            InputKind.CONSTANT_TENSOR,
        )
        self.assertEqual(
            get_input_kind(result.exported_program, out_qp.zero_point),
            InputKind.CONSTANT_TENSOR,
        )

    def test_absorb_multi_output_layer_norm_through_getitem(self) -> None:
        """Absorb the output-0 requant of a MULTI-output fused op.

        ``fused_quant.native_layer_norm`` returns ``(out, mean, rstd)``; only
        output 0 is quantized and its value flows through ``getitem(op, 0)``. The
        pass must reach the ``getitem(0) → dequant → quant`` through the accessor,
        fold the quant into the op's out qparams, and leave mean/rstd (getitem
        1/2) untouched."""
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(2, 3, 4))
        weight = builder.placeholder(
            "weight", torch.randn(4), input_kind=InputKind.BUFFER
        )
        bias = builder.placeholder("bias", torch.randn(4), input_kind=InputKind.BUFFER)
        out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, zero_point=0, dtype=torch.int8
        )
        ln = builder.call_operator(
            op=exir_ops.edge.fused_quant.native_layer_norm.default,
            args=(
                x,
                None,  # inp_scale (input unquantized)
                None,  # inp_zero_point
                torch.float32,
                0,
                0,
                *out_qparams,  # output-0 qparams
                [4],  # normalized_shape
                weight,
                bias,
                1e-5,  # eps
            ),
        )
        g0 = builder.call_getitem(ln, 0)
        g1 = builder.call_getitem(ln, 1)
        g2 = builder.call_getitem(ln, 2)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(g0,),
            kwargs={
                "scale": 0.5,
                "zero_point": 0,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.3,
                "zero_point": 2,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        builder.output([quant, g1, g2])
        ep = builder.get_program()

        result = QuantAbsorptionPass().call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
            0,
            "output-0 dequant should be absorbed",
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            0,
            "output-0 quant should be absorbed",
        )

        ln_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.native_layer_norm.default,
        )
        self.assertEqual(len(ln_nodes), 1)
        out_qp = get_qparams_from_node(ln_nodes[0], "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)
        self.assertEqual(get_zero_point(result.exported_program, out_qp), 2)

        # out/mean/rstd getitems all survive; getitem(0) now yields int8.
        getitems = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target == operator.getitem
        ]
        self.assertEqual(len(getitems), 3)
        g0_after = next(n for n in getitems if n.args[1] == 0)
        self.assertEqual(g0_after.meta["val"].dtype, torch.int8)

    def test_absorb_dequant_quant(self) -> None:
        """fused_quant.linear → dequant → quant should be absorbed."""

        ep = self._build_fused_linear_with_dequant_quant(
            out_scale=0.5,
            out_zp=0,
            downstream_scale=0.3,
            downstream_zp=2,
        )

        result = QuantAbsorptionPass().call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 0, "dequant should be removed")

        quant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(len(quant_nodes), 0, "quant should be removed")

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1)

        out_qp = get_qparams_from_node(linear_nodes[0], "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)

        self.assertEqual(get_zero_point(result.exported_program, out_qp), 2)

    def test_absorb_composes_mismatched_dequant_affine_effect(self) -> None:
        ep = self._build_fused_linear_with_dequant_quant(
            out_scale=0.5,
            out_zp=4,
            dequant_scale=1.0,
            dequant_zp=2,
            downstream_scale=0.25,
            downstream_zp=3,
        )

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        graph = result.exported_program.graph
        self.assertFalse(
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            )
        )
        self.assertFalse(
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            )
        )
        (linear,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        out_qparams = get_qparams_from_node(linear, "out")
        assert out_qparams is not None
        replacement_scale = get_scale(result.exported_program, out_qparams)
        replacement_zp = get_zero_point(result.exported_program, out_qparams)
        self.assertAlmostEqual(0.125, replacement_scale)
        self.assertEqual(11, replacement_zp)

        source = torch.tensor([-2.0, 0.0, 3.0])
        original_zero_error_codes = ((source / 0.5 + 4) - 2) * 1.0 / 0.25 + 3
        replacement_zero_error_codes = source / replacement_scale + replacement_zp
        torch.testing.assert_close(
            original_zero_error_codes, replacement_zero_error_codes
        )

    def test_no_absorb_when_composed_zero_point_is_fractional(self) -> None:
        ep = self._build_fused_linear_with_dequant_quant(
            out_scale=0.5,
            out_zp=1,
            dequant_scale=0.3,
            dequant_zp=0,
            downstream_scale=0.2,
            downstream_zp=0,
        )

        result = QuantAbsorptionPass()(ep)

        self.assertFalse(result.modified)
        graph = result.exported_program.graph
        self.assertEqual(
            1,
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
        )
        self.assertEqual(
            1,
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
        )

    def _build_fused_embedding_with_bare_quant(
        self,
        downstream_scale: float = 0.3,
        downstream_zp: int = 2,
    ) -> ExportedProgram:
        """Build: fused_quant.embedding(out=None, float gather) → quant(B).

        The embedding leaves its output unquantized and a consumer quantizes the
        gathered rows -- the bare-quantize shape that QuantAbsorptionPass folds
        into the embedding's (previously null) output qparams.
        """
        builder = ProgramBuilder()
        table = builder.placeholder(
            "table", torch.randint(-8, 7, (6, 8), dtype=torch.int8)
        )
        indices = builder.placeholder("indices", torch.tensor([0, 3, 5, 1]))
        # Table quantized per-tensor; dtype is the dequantized (float) output
        # dtype, so the gather is float. Output left unquantized.
        weight_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.float32
        )
        embedding = builder.call_operator(
            op=exir_ops.edge.fused_quant.embedding.default,
            args=(
                table,
                *weight_qparams,
                None,  # out_scale
                None,  # out_zero_point
                torch.int8,
                -128,
                127,
                indices,
            ),
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(embedding,),
            kwargs={
                "scale": downstream_scale,
                "zero_point": downstream_zp,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        builder.output([quant])
        return builder.get_program()

    def test_absorb_bare_quant_into_unquantized_embedding_out(self) -> None:
        """fused_quant.embedding(out=None) → quant folds the quantize into the
        embedding's output qparams (the precondition for prequantization)."""
        ep = self._build_fused_embedding_with_bare_quant(
            downstream_scale=0.3, downstream_zp=2
        )
        result = QuantAbsorptionPass().call(ep)
        self.assertTrue(result.modified)
        graph = result.exported_program.graph

        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            0,
            "the bare quantize should be absorbed",
        )
        embedding_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.embedding.default
        )
        self.assertEqual(len(embedding_nodes), 1)
        emb = embedding_nodes[0]
        out_scale = get_arg(emb, "out_scale", Optional[torch.fx.Node])
        assert isinstance(out_scale, torch.fx.Node)
        out_qp = get_qparams_from_node(emb, "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)
        self.assertEqual(get_zero_point(result.exported_program, out_qp), 2)
        # The fused op now emits the quantized dtype.
        self.assertEqual(emb.meta["val"].dtype, torch.int8)

    def test_no_absorb_bare_quant_when_embedding_output_forks(self) -> None:
        """A fused op feeding both a quantize and another consumer keeps a float
        path, so the bare-quantize fold must not fire (single-user only)."""
        builder = ProgramBuilder()
        table = builder.placeholder(
            "table", torch.randint(-8, 7, (6, 8), dtype=torch.int8)
        )
        indices = builder.placeholder("indices", torch.tensor([0, 3, 5, 1]))
        weight_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.float32
        )
        embedding = builder.call_operator(
            op=exir_ops.edge.fused_quant.embedding.default,
            args=(
                table,
                *weight_qparams,
                None,
                None,
                torch.int8,
                -128,
                127,
                indices,
            ),
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(embedding,),
            kwargs={
                "scale": 0.3,
                "zero_point": 2,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        # A second (float) consumer of the embedding output.
        relu = builder.call_operator(
            op=exir_ops.edge.aten.relu.default, args=(embedding,)
        )
        builder.output([quant, relu])
        ep = builder.get_program()

        result = QuantAbsorptionPass()(ep)
        assert result is not None
        self.assertFalse(result.modified)
        emb = result.exported_program.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.embedding.default
        )[0]
        self.assertIsNone(get_arg(emb, "out_scale", Optional[torch.fx.Node]))

    def test_absorbed_qparams_vals_are_fake(self) -> None:
        """The new out_scale/out_zero_point nodes must carry FakeTensor
        meta['val']s, since the pass runs in fake tensor mode and downstream
        passes rely on meta['val'] being fake."""

        ep = self._build_fused_linear_with_dequant_quant(
            out_scale=0.5,
            out_zp=0,
            downstream_scale=0.3,
            downstream_zp=2,
        )

        result = QuantAbsorptionPass().call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph
        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1)

        for arg_name in ("out_scale", "out_zero_point"):
            qparam_node = get_arg(linear_nodes[0], arg_name, Optional[torch.fx.Node])
            assert isinstance(qparam_node, torch.fx.Node)
            val = qparam_node.meta.get("val")
            self.assertIsInstance(
                val,
                FakeTensor,
                f"{arg_name} node's meta['val'] must be a FakeTensor, got {type(val)}",
            )

    def test_no_absorb_when_no_mismatch(self) -> None:
        """No absorption when there's no downstream requantize/dequant-quant."""
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))

        out_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8)

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )
        builder.output([linear])
        ep = builder.get_program()

        result = QuantAbsorptionPass().call(ep)
        assert result is not None
        self.assertFalse(result.modified)

    def _build_fused_linear_with_forked_dequant(
        self,
        out_scale: float = 0.5,
        out_zp: int = 0,
        dequant_scale: Optional[float] = None,
        dequant_zp: Optional[int] = None,
        downstream_scale: float = 0.3,
        downstream_zp: int = 2,
    ) -> ExportedProgram:
        """Build: fused_quant.linear(Q(A)) → DQ(D) → Q(B)
        └→ relu (float user)
        """
        dequant_scale = out_scale if dequant_scale is None else dequant_scale
        dequant_zp = out_zp if dequant_zp is None else dequant_zp

        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))

        out_qparams = create_per_tensor_qparams(
            builder, scale=out_scale, zero_point=out_zp, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(linear,),
            kwargs={
                "scale": dequant_scale,
                "zero_point": dequant_zp,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": downstream_scale,
                "zero_point": downstream_zp,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        float_user = builder.call_operator(
            op=exir_ops.edge.aten.relu.default,
            args=(dequant,),
        )
        builder.output([quant, float_user])
        return builder.get_program()

    def test_no_absorb_forked_dequant_by_default(self) -> None:
        """Forked dequant should NOT be absorbed when absorb_with_fork=False."""
        ep = self._build_fused_linear_with_forked_dequant()

        result = QuantAbsorptionPass(absorb_with_fork=False).call(ep)
        assert result is not None
        self.assertFalse(result.modified)

    def test_absorb_forked_dequant_when_enabled(self) -> None:
        """Forked dequant should be absorbed when absorb_with_fork=True.

        Before: linear(out_scale=0.5) → dequant(0.5) → quant(0.3)
                                              └→ relu
        After:  linear(out_scale=0.3) → dequant(0.3) → relu
                                      └→ (direct int8, quant removed)
        """
        ep = self._build_fused_linear_with_forked_dequant(
            out_scale=0.5,
            out_zp=0,
            downstream_scale=0.3,
            downstream_zp=2,
        )

        result = QuantAbsorptionPass(absorb_with_fork=True).call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        quant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(len(quant_nodes), 0, "quant should be removed")

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 1, "dequant should remain for float user")
        self.assertEqual(
            get_arg(dequant_nodes[0], "scale", float),
            0.3,
            "dequant scale should be updated to match new out_qparams",
        )
        self.assertEqual(
            get_arg(dequant_nodes[0], "zero_point", int),
            2,
            "dequant zero_point should be updated to match new out_qparams",
        )

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1)
        out_qp = get_qparams_from_node(linear_nodes[0], "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)

    def test_absorb_forked_mismatched_dequant_preserves_float_effect(self) -> None:
        ep = self._build_fused_linear_with_forked_dequant(
            out_scale=0.5,
            out_zp=4,
            dequant_scale=1.0,
            dequant_zp=2,
            downstream_scale=0.25,
            downstream_zp=3,
        )

        result = QuantAbsorptionPass(absorb_with_fork=True)(ep)

        self.assertTrue(result.modified)
        graph = result.exported_program.graph
        (linear,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        out_qparams = get_qparams_from_node(linear, "out")
        assert out_qparams is not None
        replacement_scale = get_scale(result.exported_program, out_qparams)
        replacement_zp = get_zero_point(result.exported_program, out_qparams)
        self.assertAlmostEqual(0.125, replacement_scale)
        self.assertEqual(11, replacement_zp)

        (dequant,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        downstream_scale = get_arg(dequant, "scale", float)
        downstream_zp = get_arg(dequant, "zero_point", int)
        self.assertAlmostEqual(0.25, downstream_scale)
        self.assertEqual(3, downstream_zp)

        source = torch.tensor([-2.0, 0.0, 3.0])
        original_float = (source / 0.5 + 4 - 2) * 1.0
        replacement_codes = source / replacement_scale + replacement_zp
        replacement_float = (replacement_codes - downstream_zp) * downstream_scale
        torch.testing.assert_close(original_float, replacement_float)

    def _build_fused_linear_with_permute_and_forked_dequant(
        self,
        out_scale: float = 0.5,
        out_zp: int = 0,
        downstream_scale: float = 0.3,
        downstream_zp: int = 2,
        downstream_dtype: torch.dtype = torch.int8,
    ) -> ExportedProgram:
        """Build: fused_quant.linear → permute → dequant → quant
        └→ relu (float user)
        """
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))

        out_qparams = create_per_tensor_qparams(
            builder, scale=out_scale, zero_point=out_zp, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(linear, [0, 1]),
        )

        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(permute,),
            kwargs={
                "scale": out_scale,
                "zero_point": out_zp,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": downstream_scale,
                "zero_point": downstream_zp,
                "quant_min": torch.iinfo(downstream_dtype).min,
                "quant_max": torch.iinfo(downstream_dtype).max,
                "dtype": downstream_dtype,
            },
        )
        float_user = builder.call_operator(
            op=exir_ops.edge.aten.relu.default,
            args=(dequant,),
        )
        builder.output([quant, float_user])
        return builder.get_program()

    def test_absorb_through_permute(self) -> None:
        """Absorb dequant→quant through a permute passthrough op.

        Before: linear(out=0.5) → permute → dequant(0.5) → quant(0.3)
                                                  └→ relu
        After:  linear(out=0.3) → permute → dequant(0.3) → relu
                                          └→ (direct int8, quant removed)
        """
        ep = self._build_fused_linear_with_permute_and_forked_dequant(
            out_scale=0.5,
            out_zp=0,
            downstream_scale=0.3,
            downstream_zp=2,
        )

        result = QuantAbsorptionPass(absorb_with_fork=True).call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        quant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(len(quant_nodes), 0, "quant should be removed")

        permute_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.permute_copy.default,
        )
        self.assertEqual(len(permute_nodes), 1, "permute should remain")

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 1, "dequant should remain for float user")
        self.assertEqual(
            get_arg(dequant_nodes[0], "scale", float),
            0.3,
            "dequant scale should be updated",
        )
        self.assertEqual(
            get_arg(dequant_nodes[0], "zero_point", int),
            2,
            "dequant zero_point should be updated",
        )

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1)
        out_qp = get_qparams_from_node(linear_nodes[0], "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)

    def test_no_absorb_through_permute_without_flag(self) -> None:
        """Passthrough absorption should not fire when dequant is forked
        and absorb_with_fork is False."""
        ep = self._build_fused_linear_with_permute_and_forked_dequant()

        result = QuantAbsorptionPass(absorb_with_fork=False).call(ep)
        assert result is not None
        self.assertFalse(result.modified)

    def test_absorb_through_permute_single_user(self) -> None:
        """Passthrough absorption works without fork flag when dequant
        has a single user (the quant).

        Before: linear(out=0.5) → permute → dequant(0.5) → quant(0.3)
        After:  linear(out=0.3) → permute → ...
        """
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))

        out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, zero_point=0, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )

        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(linear, [0, 1]),
        )

        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(permute,),
            kwargs={
                "scale": 0.5,
                "zero_point": 0,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.3,
                "zero_point": 2,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        builder.output([quant])
        ep = builder.get_program()

        result = QuantAbsorptionPass(absorb_with_fork=False).call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        quant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(len(quant_nodes), 0, "quant should be removed")

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 0, "dequant should be removed")

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        out_qp = get_qparams_from_node(linear_nodes[0], "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)

    def test_absorb_through_permute_changes_dtype_forked(self) -> None:
        """When the downstream quant targets a different dtype than the fused
        op's original output (int8 → uint8), the new dtype must propagate to the
        fused op out qparams, the permute's meta['val'], and the kept dequant.
        """
        ep = self._build_fused_linear_with_permute_and_forked_dequant(
            out_scale=0.5,
            out_zp=0,
            downstream_scale=0.3,
            downstream_zp=128,
            downstream_dtype=torch.uint8,
        )

        result = QuantAbsorptionPass(absorb_with_fork=True).call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        permute_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.permute_copy.default,
        )
        self.assertEqual(len(permute_nodes), 1)
        self.assertEqual(
            permute_nodes[0].meta["val"].dtype,
            torch.uint8,
            "permute meta['val'] should take the downstream quant's out dtype",
        )

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1)
        self.assertEqual(
            get_arg(linear_nodes[0], "out_dtype", torch.dtype), torch.uint8
        )
        self.assertEqual(get_arg(linear_nodes[0], "out_quant_min", int), 0)
        self.assertEqual(get_arg(linear_nodes[0], "out_quant_max", int), 255)
        self.assertEqual(linear_nodes[0].meta["val"].dtype, torch.uint8)

        # The forked dequant is kept for the float (relu) user, but its input is
        # now the requantized uint8 value, so its input qparams must follow.
        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 1)
        self.assertEqual(get_arg(dequant_nodes[0], "dtype", torch.dtype), torch.uint8)
        self.assertEqual(get_arg(dequant_nodes[0], "quant_min", int), 0)
        self.assertEqual(get_arg(dequant_nodes[0], "quant_max", int), 255)
        self.assertEqual(get_arg(dequant_nodes[0], "scale", float), 0.3)

    def test_absorb_through_permute_changes_dtype_single_user(self) -> None:
        """Non-forked path: dequant is dropped, and the new dtype still
        propagates to the fused op and the permute it now feeds."""
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1, 8))

        out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, zero_point=0, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(linear, [0, 1]),
        )
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(permute,),
            kwargs={
                "scale": 0.5,
                "zero_point": 0,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.3,
                "zero_point": 128,
                "quant_min": 0,
                "quant_max": 255,
                "dtype": torch.uint8,
            },
        )
        builder.output([quant])
        ep = builder.get_program()

        result = QuantAbsorptionPass(absorb_with_fork=False).call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 0, "dequant should be removed")

        permute_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.permute_copy.default,
        )
        self.assertEqual(len(permute_nodes), 1)
        self.assertEqual(permute_nodes[0].meta["val"].dtype, torch.uint8)

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(
            get_arg(linear_nodes[0], "out_dtype", torch.dtype), torch.uint8
        )
        self.assertEqual(linear_nodes[0].meta["val"].dtype, torch.uint8)

    def test_absorb_through_view_changes_dtype_single_user(self) -> None:
        """A view_copy is a per-tensor-transparent passthrough, like permute.

        Before: linear(out=0.5, int8) -> view_copy -> dequant(0.5) -> quant(0.3, uint8)
        After:  linear(out=0.3, uint8) -> view_copy -> ...

        Covers both that view is a valid passthrough and that the dtype change
        (int8 -> uint8) propagates to the fused op and the view's meta['val'].
        """
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(4, 8))

        out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, zero_point=0, dtype=torch.int8
        )

        linear = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                x,
                x,
                None,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.uint8,
                0,
                0,
                *out_qparams,
            ),
        )
        # A genuine reshape (4, 4) -> (2, 8) to exercise the view passthrough.
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(linear, [2, 8]),
        )
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(view,),
            kwargs={
                "scale": 0.5,
                "zero_point": 0,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.3,
                "zero_point": 128,
                "quant_min": 0,
                "quant_max": 255,
                "dtype": torch.uint8,
            },
        )
        builder.output([quant])
        ep = builder.get_program()

        result = QuantAbsorptionPass(absorb_with_fork=False).call(ep)
        assert result is not None
        self.assertTrue(result.modified)

        graph = result.exported_program.graph_module.graph

        quant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(len(quant_nodes), 0, "quant should be removed")

        dequant_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(len(dequant_nodes), 0, "dequant should be removed")

        view_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.view_copy.default,
        )
        self.assertEqual(len(view_nodes), 1, "view should remain")
        self.assertEqual(
            view_nodes[0].meta["val"].dtype,
            torch.uint8,
            "view carrying the absorbed output must be retyped to uint8",
        )

        linear_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.linear.default,
        )
        self.assertEqual(len(linear_nodes), 1)
        out_qp = get_qparams_from_node(linear_nodes[0], "out")
        assert out_qp is not None
        self.assertAlmostEqual(get_scale(result.exported_program, out_qp), 0.3)
        self.assertEqual(
            get_arg(linear_nodes[0], "out_dtype", torch.dtype), torch.uint8
        )
        self.assertEqual(linear_nodes[0].meta["val"].dtype, torch.uint8)

    def _build_dequant_quant_pool(
        self,
        *,
        dequant_scale: float = 0.25,
        quant_scale: float = 0.5,
        consumer_input_scale: float = 0.5,
        consumer_input_zero_point: int = -5,
        input_step: int = 1,
        expose_quant: bool = False,
        expose_dequant: bool = False,
        input_passthrough: str | None = None,
        expose_passthrough: bool = False,
    ) -> tuple[ExportedProgram, torch.Tensor]:
        builder = ProgramBuilder()
        inp = (torch.arange(16, dtype=torch.uint8) * input_step).reshape(1, 4, 4, 1)
        inp_node = builder.placeholder("inp", inp)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(inp_node,),
            kwargs={
                "scale": dequant_scale,
                "zero_point": 0,
                "quant_min": 0,
                "quant_max": 255,
                "dtype": torch.uint8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": quant_scale,
                "zero_point": -5,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        consumer_input = quant
        if input_passthrough == "view":
            consumer_input = builder.call_operator(
                op=exir_ops.edge.aten.view_copy.default,
                args=(quant, [1, 2, 8, 1]),
            )
        elif input_passthrough == "permute":
            consumer_input = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(quant, [0, 2, 1, 3]),
            )
        elif input_passthrough is not None:
            raise ValueError(f"Unsupported input passthrough: {input_passthrough}")
        pool_input_qparams = create_per_tensor_qparams(
            builder,
            scale=consumer_input_scale,
            zero_point=consumer_input_zero_point,
            dtype=torch.float32,
        )
        pool_output_qparams = create_per_tensor_qparams(
            builder,
            scale=0.1,
            dtype=torch.uint8,
        )
        pool = builder.call_operator(
            op=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
            args=(
                consumer_input,
                *pool_input_qparams,
                *pool_output_qparams[:3],
                0,
                255,
                [2, 2],
                [2, 2],
                [0, 0],
                False,
                True,
                None,
            ),
        )
        outputs = [pool]
        if expose_quant:
            outputs.append(quant)
        if expose_dequant:
            outputs.append(dequant)
        if expose_passthrough:
            outputs.append(consumer_input)
        builder.output(outputs)
        return builder.get_program(), inp

    @staticmethod
    def _direct_pool(inp: torch.Tensor) -> torch.Tensor:
        return torch.ops.fused_quant.avg_pool2d_channels_last.default(
            inp,
            torch.tensor(0.25, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int64),
            torch.float32,
            0,
            255,
            torch.tensor(0.1, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int64),
            torch.uint8,
            0,
            255,
            [2, 2],
            [2, 2],
            [0, 0],
            False,
            True,
            None,
        )

    def _build_dequant_quant_add(self) -> tuple[ExportedProgram, torch.Tensor]:
        builder = ProgramBuilder()
        inp = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        inp_node = builder.placeholder("inp", inp)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(inp_node,),
            kwargs={
                "scale": 0.25,
                "zero_point": 0,
                "quant_min": 0,
                "quant_max": 255,
                "dtype": torch.uint8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.5,
                "zero_point": -5,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        input_qparams = create_per_tensor_qparams(
            builder,
            scale=0.5,
            zero_point=-5,
            dtype=torch.float32,
        )
        other_qparams = create_per_tensor_qparams(
            builder,
            scale=0.5,
            zero_point=-5,
            dtype=torch.float32,
        )
        output_qparams = create_per_tensor_qparams(builder, scale=0.1)
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.default,
            args=(
                quant,
                quant,
                *input_qparams,
                *other_qparams,
                *output_qparams,
                1.0,
            ),
        )
        builder.output([add])
        return builder.get_program(), inp

    def test_absorbs_dequant_quant_into_fused_consumer_input(self) -> None:
        ep, inp = self._build_dequant_quant_pool()

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        (actual,) = result.exported_program.module()(inp)
        self.assertTrue(torch.equal(self._direct_pool(inp), actual))

        graph = result.exported_program.graph
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
        )
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
        )
        (pool,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
        )
        pool_input = get_arg(pool, "inp", torch.fx.Node)
        self.assertEqual("placeholder", pool_input.op)
        input_qparams = get_qparams_from_node(pool, "inp")
        assert input_qparams is not None
        self.assertAlmostEqual(0.25, get_scale(result.exported_program, input_qparams))
        self.assertEqual(0, get_zero_point(result.exported_program, input_qparams))
        self.assertEqual(torch.float32, input_qparams.dtype)
        self.assertEqual(0, input_qparams.quant_min)
        self.assertEqual(255, input_qparams.quant_max)
        result.exported_program.validate()

    def test_absorbs_consumer_input_through_view_copy(self) -> None:
        ep, inp = self._build_dequant_quant_pool(input_passthrough="view")

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        (actual,) = result.exported_program.module()(inp)
        expected = self._direct_pool(
            torch.ops.aten.view_copy.default(inp, [1, 2, 8, 1])
        )
        self.assertTrue(torch.equal(expected, actual))

        graph = result.exported_program.graph
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
        )
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
        )
        (view,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.view_copy.default
        )
        self.assertEqual("placeholder", view.args[0].op)
        self.assertEqual(torch.uint8, view.meta["val"].dtype)
        (pool,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
        )
        self.assertIs(view, get_arg(pool, "inp", torch.fx.Node))
        input_qparams = get_qparams_from_node(pool, "inp")
        assert input_qparams is not None
        self.assertAlmostEqual(0.25, get_scale(result.exported_program, input_qparams))
        self.assertEqual(0, get_zero_point(result.exported_program, input_qparams))
        result.exported_program.validate()

    def test_absorbs_consumer_input_through_permute_copy(self) -> None:
        ep, inp = self._build_dequant_quant_pool(input_passthrough="permute")

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        (actual,) = result.exported_program.module()(inp)
        expected = self._direct_pool(
            torch.ops.aten.permute_copy.default(inp, [0, 2, 1, 3])
        )
        self.assertTrue(torch.equal(expected, actual))

        graph = result.exported_program.graph
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
        )
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
        )
        (permute,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )
        self.assertEqual("placeholder", permute.args[0].op)
        self.assertEqual(torch.uint8, permute.meta["val"].dtype)
        (pool,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
        )
        self.assertIs(permute, get_arg(pool, "inp", torch.fx.Node))
        input_qparams = get_qparams_from_node(pool, "inp")
        assert input_qparams is not None
        self.assertAlmostEqual(0.25, get_scale(result.exported_program, input_qparams))
        self.assertEqual(0, get_zero_point(result.exported_program, input_qparams))
        result.exported_program.validate()

    def test_does_not_absorb_through_shared_input_passthrough(self) -> None:
        ep, _ = self._build_dequant_quant_pool(
            input_passthrough="view", expose_passthrough=True
        )

        result = QuantAbsorptionPass()(ep)

        self.assertFalse(result.modified)
        self.assertEqual(
            1,
            len(
                result.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
        )

    def test_absorbs_all_quantized_inputs_of_binary_op(self) -> None:
        ep, inp = self._build_dequant_quant_add()

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        (actual,) = result.exported_program.module()(inp)
        expected = torch.ops.fused_quant.add.default(
            inp,
            inp,
            torch.tensor(0.25, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int64),
            torch.float32,
            0,
            255,
            torch.tensor(0.25, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int64),
            torch.float32,
            0,
            255,
            torch.tensor(0.1, dtype=torch.float32),
            torch.tensor(0, dtype=torch.int64),
            torch.int8,
            -128,
            127,
            1.0,
        )
        self.assertTrue(torch.equal(expected, actual))

        graph = result.exported_program.graph
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
        )
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
        )
        (add,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.add.default,
        )
        for input_name in ("inp", "other"):
            self.assertEqual("placeholder", get_arg(add, input_name, torch.fx.Node).op)
            input_qparams = get_qparams_from_node(add, input_name)
            assert input_qparams is not None
            self.assertAlmostEqual(
                0.25, get_scale(result.exported_program, input_qparams)
            )
            self.assertEqual(0, get_zero_point(result.exported_program, input_qparams))
            self.assertEqual(torch.float32, input_qparams.dtype)
            self.assertEqual(0, input_qparams.quant_min)
            self.assertEqual(255, input_qparams.quant_max)
        result.exported_program.validate()

    def test_composes_mismatched_consumer_input_qparams(self) -> None:
        ep, inp = self._build_dequant_quant_pool(
            consumer_input_scale=0.4,
            consumer_input_zero_point=-3,
            input_step=2,
        )
        expected = ep.module()(inp)

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        actual = result.exported_program.module()(inp)
        self.assertEqual(len(expected), len(actual))
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(actual_output, expected_output)

        graph = result.exported_program.graph
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
        )
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
        )
        (pool,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
        )
        input_qparams = get_qparams_from_node(pool, "inp")
        assert input_qparams is not None
        self.assertAlmostEqual(0.2, get_scale(result.exported_program, input_qparams))
        self.assertEqual(4, get_zero_point(result.exported_program, input_qparams))
        result.exported_program.validate()

    def test_does_not_absorb_unrepresentable_consumer_input_qparams(self) -> None:
        ep, _ = self._build_dequant_quant_pool(
            quant_scale=0.4,
            consumer_input_scale=0.5,
            consumer_input_zero_point=-4,
        )

        result = QuantAbsorptionPass()(ep)

        self.assertFalse(result.modified)

    def test_does_not_absorb_per_channel_consumer_input(self) -> None:
        builder = ProgramBuilder()
        inp = torch.arange(8, dtype=torch.uint8).reshape(1, 4, 2)
        inp_node = builder.placeholder("inp", inp)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(inp_node,),
            kwargs={
                "scale": 0.25,
                "zero_point": 0,
                "quant_min": 0,
                "quant_max": 255,
                "dtype": torch.uint8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.5,
                "zero_point": -5,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        input_qparams = create_per_axis_qparams(
            builder,
            num_channels=2,
            ndim=3,
            axis=2,
            dtype=torch.float32,
        )
        output_qparams = create_per_tensor_qparams(builder)
        relu = builder.call_operator(
            op=exir_ops.edge.fused_quant.relu.default,
            args=(quant, *input_qparams, *output_qparams),
        )
        builder.output([relu])
        ep = builder.get_program()

        result = QuantAbsorptionPass()(ep)

        self.assertFalse(result.modified)
        self.assertEqual(
            1,
            len(
                result.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
        )

    def test_does_not_absorb_shared_quantize(self) -> None:
        ep, _ = self._build_dequant_quant_pool(expose_quant=True)

        result = QuantAbsorptionPass()(ep)

        self.assertFalse(result.modified)
        self.assertEqual(
            1,
            len(
                result.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
        )

    def test_absorbs_quantize_and_preserves_shared_dequantize(self) -> None:
        ep, inp = self._build_dequant_quant_pool(expose_dequant=True)
        _, expected_dequant = ep.module()(inp)

        result = QuantAbsorptionPass()(ep)

        self.assertTrue(result.modified)
        actual_pool, actual_dequant = result.exported_program.module()(inp)
        self.assertTrue(torch.equal(self._direct_pool(inp), actual_pool))
        self.assertTrue(torch.equal(expected_dequant, actual_dequant))
        self.assertEqual(
            [],
            result.exported_program.graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
        )
        self.assertEqual(
            1,
            len(
                result.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
        )
        result.exported_program.validate()


# ── SinkConstantCat models ────────────────────────────────────────────
