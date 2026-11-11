# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.backends.transforms.fused_quant.pre_quantize_passes.fold_input_free_subgraphs import (
    FoldInputFreeSubgraphs,
)
from executorch.backends.transforms.fused_quant.test.helpers import trace
from torch import nn
from torch.export import ExportedProgram


class _InGraphMaskModel(nn.Module):
    """Adds a causal mask built in-graph, as transformers' eager mask does."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        length = scores.shape[-1]
        q_idx = torch.arange(length)[:, None]
        kv_idx = torch.arange(length)[None, :]
        keep = q_idx.new_ones((), dtype=torch.bool) & (kv_idx <= q_idx)
        mask = torch.where(keep, torch.tensor(0.0), torch.finfo(torch.float32).min)
        return scores + mask


class _ParameterModel(nn.Module):
    """Scales by ``1 + weight``, a value derived from a parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1 + self.weight)


class _BareConstructorModel(nn.Module):
    """Adds a constructor's value that no other op refines."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.full((4,), 2.0)


def _targets(program: ExportedProgram) -> set[object]:
    return {node.target for node in program.graph.nodes if node.op == "call_function"}


class FoldInputFreeSubgraphsTest(unittest.TestCase):
    def test_folds_a_mask_built_in_graph(self) -> None:
        model = _InGraphMaskModel()
        scores = torch.randn(1, 4, 8, 8)
        program = trace(model, (scores,))

        result = FoldInputFreeSubgraphs()(program)

        self.assertTrue(result.modified)
        program = result.exported_program
        self.assertEqual({torch.ops.aten.add.Tensor}, _targets(program))
        torch.testing.assert_close(program.module()(scores), model(scores))

    def test_leaves_a_parameter_derived_value(self) -> None:
        model = _ParameterModel()
        program = trace(model, (torch.randn(4),))
        targets = _targets(program)

        result = FoldInputFreeSubgraphs()(program)

        self.assertFalse(result.modified)
        self.assertEqual(targets, _targets(result.exported_program))
        self.assertEqual(
            ["weight"],
            list(result.exported_program.graph_signature.inputs_to_parameters.values()),
        )

    def test_leaves_a_bare_constructor(self) -> None:
        program = trace(_BareConstructorModel(), (torch.randn(4),))
        targets = _targets(program)

        result = FoldInputFreeSubgraphs()(program)

        self.assertFalse(result.modified)
        self.assertEqual(targets, _targets(result.exported_program))
