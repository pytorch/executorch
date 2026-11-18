# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.fused_quant.optimization_passes.replace_sub_constant_with_add import (
    ReplaceSubConstantWithAdd,
)
from executorch.backends.transforms.fused_quant.test.helpers import (
    create_per_tensor_qparams,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


class ReplaceSubConstantWithAddTest(unittest.TestCase):
    def _build_sub(
        self, *, constant_other: bool
    ) -> tuple[ExportedProgram, tuple[torch.Tensor, ...]]:
        builder = ProgramBuilder()
        inp_value = torch.tensor([[[-128, -40, -3, 0, 60, 127]]], dtype=torch.int8)
        other_value = torch.tensor([-128, -7, 0, 5, 90, 127], dtype=torch.int8)
        inp = builder.placeholder("inp", inp_value)
        other = builder.placeholder(
            "other",
            other_value,
            input_kind=InputKind.BUFFER if constant_other else InputKind.USER_INPUT,
        )
        sub = builder.call_operator(
            op=exir_ops.edge.fused_quant.sub.default,
            args=(
                inp,
                other,
                *create_per_tensor_qparams(
                    builder, scale=0.25, zero_point=-3, dtype=torch.float32
                ),
                *create_per_tensor_qparams(
                    builder, scale=0.5, zero_point=5, dtype=torch.float32
                ),
                *create_per_tensor_qparams(
                    builder, scale=0.75, zero_point=1, dtype=torch.int8
                ),
                2.0,
            ),
        )
        builder.output([sub])
        inputs = (inp_value,) if constant_other else (inp_value, other_value)
        return builder.get_program(), inputs

    def test_replaces_constant_sub_with_add(self) -> None:
        ep, inputs = self._build_sub(constant_other=True)
        expected = ep.module()(*inputs)

        result = ReplaceSubConstantWithAdd()(ep)

        self.assertTrue(result.modified)
        graph = result.exported_program.graph
        self.assertEqual(
            graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.sub.default
            ),
            [],
        )
        (add,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.add.default
        )
        self.assertEqual(get_arg(add, "alpha", float), 2.0)
        torch.testing.assert_close(
            result.exported_program.module()(*inputs), expected, rtol=0, atol=0
        )
        result.exported_program.validate()

    def test_keeps_activation_sub(self) -> None:
        ep, _ = self._build_sub(constant_other=False)

        result = ReplaceSubConstantWithAdd()(ep)

        self.assertFalse(result.modified)
