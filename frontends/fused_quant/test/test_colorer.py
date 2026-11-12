# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.transforms.fused_quant.test.helpers import (
    create_per_tensor_qparams,
)
from executorch.exir import EdgeProgramManager
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.frontends.fused_quant.colorer import (
    CLAIM_KEY,
    ColorerBase,
    ReleaseClaims,
)
from torch.export import ExportedProgram
from torch.fx import Node


def _build_relu_add_program() -> ExportedProgram:
    """inp -> fused_quant.relu -> fused_quant.add(., inp) -> output."""
    builder = ProgramBuilder()
    inp = builder.placeholder("inp", torch.randn(1, 4).to(torch.int8))

    relu = builder.call_operator(
        op=exir_ops.edge.fused_quant.relu.default,
        args=(
            inp,
            *create_per_tensor_qparams(builder),
            *create_per_tensor_qparams(builder),
        ),
    )
    add = builder.call_operator(
        op=exir_ops.edge.fused_quant.add.default,
        args=(
            relu,
            inp,
            *create_per_tensor_qparams(builder),
            *create_per_tensor_qparams(builder),
            *create_per_tensor_qparams(builder),
        ),
    )
    builder.output([add])
    return builder.get_program()


def _tagged(program: ExportedProgram) -> dict[str, str]:
    return {
        node.name: node.meta[CLAIM_KEY]
        for node in program.graph_module.graph.nodes
        if CLAIM_KEY in node.meta
    }


def _build_two_relu_program() -> ExportedProgram:
    """inp -> fused_quant.relu -> fused_quant.relu -> output."""
    builder = ProgramBuilder()
    inp = builder.placeholder("inp", torch.randn(1, 4).to(torch.int8))

    first = builder.call_operator(
        op=exir_ops.edge.fused_quant.relu.default,
        args=(
            inp,
            *create_per_tensor_qparams(builder),
            *create_per_tensor_qparams(builder),
        ),
    )
    second = builder.call_operator(
        op=exir_ops.edge.fused_quant.relu.default,
        args=(
            first,
            *create_per_tensor_qparams(builder),
            *create_per_tensor_qparams(builder),
        ),
    )
    builder.output([second])
    return builder.get_program()


def _fused_quant_targets(program: ExportedProgram) -> list[str]:
    return [
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and getattr(node.target, "namespace", None) == "fused_quant"
    ]


class ReluOnlyColorer(ColorerBase):
    def __init__(self) -> None:
        super().__init__(backend="toy")

    def can_lower(self, exported_program: ExportedProgram, node: Node) -> bool:
        return node.target == exir_ops.edge.fused_quant.relu.default


class SecondReluColorer(ColorerBase):
    """A lower-priority colorer that also wants relu."""

    def __init__(self) -> None:
        super().__init__(backend="second")

    def can_lower(self, exported_program: ExportedProgram, node: Node) -> bool:
        return node.target == exir_ops.edge.fused_quant.relu.default


class EmptyColorer(ColorerBase):
    """Claims no ops -- the shape a backend starts from."""

    def __init__(self) -> None:
        super().__init__(backend="empty")

    def can_lower(self, exported_program: ExportedProgram, node: Node) -> bool:
        return False


class ColorerBaseTest(unittest.TestCase):
    def test_a_colorer_that_lowers_nothing_tags_nothing(self) -> None:
        program = _build_relu_add_program()

        result = EmptyColorer()(program)

        self.assertFalse(result.modified)
        self.assertEqual(_tagged(result.exported_program), {})

    def test_tags_only_nodes_it_can_lower(self) -> None:
        """The colorer can lower relu but not add, so only relu is claimed."""
        program = _build_relu_add_program()

        result = ReluOnlyColorer()(program)

        self.assertTrue(result.modified)
        self.assertEqual(
            list(_tagged(result.exported_program).values()),
            ["toy"],
        )
        (tagged_name,) = _tagged(result.exported_program)
        self.assertIn("relu", tagged_name)

    def test_claims_leave_delegation_tag_to_the_partitioner(self) -> None:
        program = _build_relu_add_program()

        result = ReluOnlyColorer()(program)

        self.assertFalse(
            any(
                "delegation_tag" in node.meta
                for node in result.exported_program.graph_module.graph.nodes
            )
        )

    def test_first_claim_wins(self) -> None:
        program = _build_relu_add_program()
        ReluOnlyColorer()(program)

        result = SecondReluColorer()(program)

        self.assertFalse(result.modified)
        self.assertEqual(
            list(_tagged(result.exported_program).values()),
            ["toy"],
        )


class AllReluColorer(ColorerBase):
    """Claims every fused_quant.relu."""

    def __init__(self) -> None:
        super().__init__(backend="toy")

    def can_lower(self, exported_program: ExportedProgram, node: Node) -> bool:
        return node.target == exir_ops.edge.fused_quant.relu.default


class ColoredNodesAreLeftAloneTest(unittest.TestCase):
    """The ``ColorerBase`` contract: a claimed node is owned by its backend.

    Without this, claiming an op is meaningless -- the generic passes would
    defuse or relayout it before the backend ever saw it.

    Every fused op in these graphs is claimed, so the pass has nothing left to
    rewrite and must report no change. That keeps the assertion on the skip
    itself: the synthetic ProgramBuilder graphs carry int8 metadata and are not
    decomposable, so letting the pass touch one would fail for an unrelated
    reason. End-to-end decomposition is covered by ``test_compile.py``.
    """

    def test_decompose_skips_colored_nodes(self) -> None:
        program = _build_two_relu_program()
        AllReluColorer()(program)
        before = _fused_quant_targets(program)

        result = DecomposeFusedQuant()(program)

        self.assertFalse(result.modified)
        self.assertEqual(_fused_quant_targets(result.exported_program), before)

    def test_decompose_rewrites_uncolored_nodes(self) -> None:
        """Same graph, nothing claimed: the pass does try to rewrite."""
        program = _build_two_relu_program()

        with self.assertRaisesRegex(AssertionError, "Expecting input to have dtype"):
            DecomposeFusedQuant()(program)


def _build_requantize_edge_program() -> EdgeProgramManager:
    """inp -> fused_quant.requantize -> output, which DecomposeFusedQuant can defuse."""
    builder = ProgramBuilder()
    inp = builder.placeholder("inp", torch.tensor([[-4, 1, 7]], dtype=torch.int8))
    requantize = builder.call_operator(
        op=exir_ops.edge.fused_quant.requantize.default,
        args=(
            inp,
            *create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32),
            *create_per_tensor_qparams(builder, scale=0.25, zero_point=2),
        ),
    )
    builder.output([requantize])
    return builder.get_edge_program()


class RequantizeColorer(ColorerBase):
    def __init__(self) -> None:
        super().__init__(backend="toy")

    def can_lower(self, exported_program: ExportedProgram, node: Node) -> bool:
        return node.target == exir_ops.edge.fused_quant.requantize.default


class TakesNothingPartitioner(Partitioner):
    """Rejects every node, as a partitioner does with a claimed node it cannot lower."""

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags={}
        )


class ReleaseClaimsTest(unittest.TestCase):
    def test_drops_every_claim(self) -> None:
        program = _build_two_relu_program()
        AllReluColorer()(program)

        result = ReleaseClaims()(program)

        self.assertTrue(result.modified)
        self.assertEqual(_tagged(result.exported_program), {})

    def test_reports_no_change_without_claims(self) -> None:
        result = ReleaseClaims()(_build_two_relu_program())

        self.assertFalse(result.modified)

    def test_rejected_claim_falls_back_once_released(self) -> None:
        """to_backend leaves the claim on a node its partitioner rejected."""
        edge = (
            _build_requantize_edge_program()
            .transform([RequantizeColorer()])
            .to_backend(TakesNothingPartitioner())
        )
        self.assertEqual(list(_tagged(edge.exported_program()).values()), ["toy"])
        with self.assertRaisesRegex(RuntimeError, "claimed by toy.*ReleaseClaims"):
            AssertNoFusedQuantOps()(edge.exported_program())

        edge = edge.transform(
            [ReleaseClaims(), ToChannelsFirst(), DecomposeFusedQuant()]
        )

        AssertNoFusedQuantOps()(edge.exported_program())
        self.assertEqual(_tagged(edge.exported_program()), {})
