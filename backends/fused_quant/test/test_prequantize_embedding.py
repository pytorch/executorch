# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.optimization_passes.prequantize_embedding import (
    PrequantizeEmbedding,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


class PrequantizeEmbeddingTest(unittest.TestCase):
    """Tests for the PrequantizeEmbedding pass."""

    _NUM_EMBEDDINGS = 6
    _EMBEDDING_DIM = 8
    _NUM_GROUPS = 2

    def _build_quantized_embedding_program(
        self, out_scale: float = 0.05, out_zp: int = 3
    ) -> ExportedProgram:
        """fused_quant.embedding with a group-quantized table and a per-tensor
        quantized output -- the shape PrequantizeEmbedding folds into a plain int8
        gather. The gather feeds a nop view (not the graph output directly) so the
        rewrite replaces an interior node and the output signature stays stable.
        """
        torch.manual_seed(0)
        builder = ProgramBuilder()
        num_embeddings = self._NUM_EMBEDDINGS
        embedding_dim = self._EMBEDDING_DIM
        num_groups = self._NUM_GROUPS

        table = builder.placeholder(
            "table",
            torch.randint(-8, 7, (num_embeddings, embedding_dim), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        indices = builder.placeholder("indices", torch.tensor([0, 3, 5, 1]))
        # Group-quant weight: full-rank [vocab, num_groups] scale/zp encodes block
        # (1, dim/num_groups). Strictly-positive scales make dequant non-trivial.
        scale = builder.placeholder(
            "weight_scale",
            torch.rand(num_embeddings, num_groups) + 0.1,
            input_kind=InputKind.BUFFER,
        )
        zp = builder.placeholder(
            "weight_zero_point",
            torch.zeros(num_embeddings, num_groups, dtype=torch.int64),
            input_kind=InputKind.BUFFER,
        )
        # Per-tensor output qparams are 0-dim lifted-constant placeholders (not
        # aten.full ops).
        out_scale_node = builder.placeholder(
            "out_scale",
            torch.tensor(out_scale, dtype=torch.float32),
            input_kind=InputKind.BUFFER,
        )
        out_zp_node = builder.placeholder(
            "out_zero_point",
            torch.tensor(out_zp, dtype=torch.int64),
            input_kind=InputKind.BUFFER,
        )
        embedding = builder.call_operator(
            op=exir_ops.edge.fused_quant.embedding.default,
            args=(
                table,
                scale,
                zp,
                torch.float32,
                -8,
                7,
                out_scale_node,
                out_zp_node,
                torch.int8,
                -128,
                127,
                indices,
            ),
        )
        out = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(embedding, [4, embedding_dim]),
        )
        builder.output([out])
        return builder.get_program()

    @staticmethod
    def _run(ep: ExportedProgram, indices: torch.Tensor) -> torch.Tensor:
        """Run the program, feeding `indices` as the sole user input and pulling
        params/buffers from the EP's backing stores."""
        full_inputs = []
        for spec in ep.graph_signature.input_specs:
            if spec.kind == InputKind.USER_INPUT:
                full_inputs.append(indices)
            else:
                assert spec.target is not None
                if spec.target in ep.state_dict:
                    full_inputs.append(ep.state_dict[spec.target])
                else:
                    full_inputs.append(ep.constants[spec.target])
        return ep.graph_module(*full_inputs)[0]

    def test_prequantize_replaces_with_aten_embedding(self) -> None:
        ep = self._build_quantized_embedding_program()

        result = PrequantizeEmbedding().call(ep)
        self.assertTrue(result.modified)
        ep = result.exported_program
        graph = ep.graph_module.graph

        # fused_quant.embedding is replaced by a plain aten.embedding gather.
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.embedding.default,
                )
            ),
            0,
        )
        gather_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.embedding.default
        )
        self.assertEqual(len(gather_nodes), 1)
        # The gather emits int8 directly (no quant math downstream).
        self.assertEqual(gather_nodes[0].meta["val"].dtype, torch.int8)

        # The scale/zero_point constants are dead and dropped from the signature.
        buffer_targets = [
            spec.target
            for spec in ep.graph_signature.input_specs
            if spec.kind == InputKind.BUFFER
        ]
        self.assertNotIn("weight_scale", buffer_targets)
        self.assertNotIn("weight_zero_point", buffer_targets)

    def test_prequantize_is_bit_exact(self) -> None:
        ep = self._build_quantized_embedding_program()
        indices = torch.tensor([0, 3, 5, 1])
        before = self._run(ep, indices)

        result = PrequantizeEmbedding().call(ep)
        after = self._run(result.exported_program, indices)

        self.assertEqual(before.dtype, torch.int8)
        self.assertTrue(
            torch.equal(before, after),
            f"prequantization changed the gather result:\n{before}\nvs\n{after}",
        )

    def test_prequantize_skips_float_output_embedding(self) -> None:
        """An embedding whose output stays float (no out qparams) is left alone
        for the cadence byte-gather path."""
        torch.manual_seed(0)
        builder = ProgramBuilder()
        table = builder.placeholder(
            "table",
            torch.randint(-8, 7, (6, 8), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        indices = builder.placeholder("indices", torch.tensor([0, 3, 5, 1]))
        scale = builder.placeholder(
            "weight_scale",
            torch.rand(6, 2) + 0.1,
            input_kind=InputKind.BUFFER,
        )
        zp = builder.placeholder(
            "weight_zero_point",
            torch.zeros(6, 2, dtype=torch.int64),
            input_kind=InputKind.BUFFER,
        )
        embedding = builder.call_operator(
            op=exir_ops.edge.fused_quant.embedding.default,
            args=(
                table,
                scale,
                zp,
                torch.float32,
                -8,
                7,
                None,  # out_scale -> float output
                None,  # out_zero_point
                torch.int8,
                -128,
                127,
                indices,
            ),
        )
        builder.output([embedding])
        ep = builder.get_edge_program().exported_program()

        result = PrequantizeEmbedding().call(ep)
        self.assertFalse(result.modified)
        self.assertEqual(
            len(
                result.exported_program.graph_module.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.embedding.default,
                )
            ),
            1,
        )
