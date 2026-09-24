# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.fused_quant.colorer import ColorerBase
from executorch.backends.fused_quant.decompose_fused_quant import DecomposeFusedQuant
from executorch.backends.fused_quant.test.helpers import create_per_tensor_qparams
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
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
        node.name: node.meta["delegation_tag"]
        for node in program.graph_module.graph.nodes
        if "delegation_tag" in node.meta
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
        super().__init__(delegation_tag="toy")

    def supported_ops(self) -> set[EdgeOpOverload]:
        return {
            exir_ops.edge.fused_quant.relu.default,
            exir_ops.edge.fused_quant.add.default,
        }

    def is_legal(self, exported_program: ExportedProgram, node: Node) -> bool:
        return node.target == exir_ops.edge.fused_quant.relu.default


class EmptyColorer(ColorerBase):
    """Claims no ops -- the shape a backend starts from."""

    def __init__(self) -> None:
        super().__init__(delegation_tag="empty")

    def supported_ops(self) -> set[EdgeOpOverload]:
        return set()



class ColorerBaseTest(unittest.TestCase):
    def test_a_colorer_with_no_supported_ops_tags_nothing(self) -> None:
        program = _build_relu_add_program()

        result = EmptyColorer()(program)

        self.assertFalse(result.modified)
        self.assertEqual(_tagged(result.exported_program), {})

    def test_tags_only_supported_and_legal_nodes(self) -> None:
        """add is a supported op but illegal, so only relu is claimed."""
        program = _build_relu_add_program()

        result = ReluOnlyColorer()(program)

        self.assertTrue(result.modified)
        self.assertEqual(
            list(_tagged(result.exported_program).values()),
            ["toy"],
        )
        (tagged_name,) = _tagged(result.exported_program)
        self.assertIn("relu", tagged_name)

    def test_rejects_an_already_delegated_graph(self) -> None:
        program = _build_relu_add_program()
        ReluOnlyColorer()(program)

        with self.assertRaisesRegex(RuntimeError, "non-delegated"):
            ReluOnlyColorer()(program)


class AllReluColorer(ColorerBase):
    """Claims every fused_quant.relu."""

    def __init__(self) -> None:
        super().__init__(delegation_tag="toy")

    def supported_ops(self) -> set[EdgeOpOverload]:
        return {exir_ops.edge.fused_quant.relu.default}


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

        with self.assertRaises(Exception):
            DecomposeFusedQuant()(program)
