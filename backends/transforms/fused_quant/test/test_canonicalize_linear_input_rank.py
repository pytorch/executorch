# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.exir.dialects.fused_quant.ops  # noqa
import torch
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.fused_quant.optimization_passes.canonicalize_linear_input_rank import (
    CanonicalizeLinearInputRank,
)
from executorch.backends.transforms.fused_quant.test.helpers import (
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_IN_FEATURES = 8
_OUT_FEATURES = 16


class CanonicalizeLinearInputRankTest(unittest.TestCase):
    def _build(
        self, input_shape: tuple[int, ...], user_is_view: bool = True
    ) -> tuple[ExportedProgram, torch.Tensor]:
        builder = ProgramBuilder()
        inp = torch.randn(*input_shape, _IN_FEATURES)
        x = builder.placeholder("x", inp)
        w = builder.placeholder(
            "weight",
            torch.randint(-16, 16, (_OUT_FEATURES, _IN_FEATURES), dtype=torch.int8),
            input_kind=InputKind.BUFFER,
        )
        b = builder.placeholder(
            "bias", torch.randn(_OUT_FEATURES), input_kind=InputKind.BUFFER
        )
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
                *create_per_channel_qparams(
                    builder, _OUT_FEATURES, scale=0.05, dtype=torch.float32
                ),
                None,
                None,
                torch.uint8,
                0,
                0,
                *create_per_tensor_qparams(builder, scale=0.5, dtype=torch.int8),
            ),
        )
        if user_is_view:
            numel = inp.numel() // _IN_FEATURES * _OUT_FEATURES
            out = builder.call_operator(
                op=exir_ops.edge.aten.view_copy.default, args=(linear, [numel])
            )
        else:
            out = builder.call_operator(
                op=exir_ops.edge.aten.slice_copy.Tensor,
                args=(linear, -1, 0, _OUT_FEATURES // 2, 1),
            )
        builder.output([out])
        return builder.get_program(), inp

    def _run(self, ep: ExportedProgram, inp: torch.Tensor) -> torch.Tensor:
        args = []
        for spec in ep.graph_signature.input_specs:
            if spec.kind == InputKind.USER_INPUT:
                args.append(inp)
            elif spec.target in ep.state_dict:
                args.append(ep.state_dict[spec.target])
            else:
                args.append(ep.constants[spec.target])
        return ep.graph_module(*args)[0]

    def _linear(self, ep: ExportedProgram) -> fx.Node:
        (linear,) = ep.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )
        return linear

    @parameterized.expand(
        [
            ("rank_2", (50,)),
            ("rank_3", (50, 1)),
        ]
    )
    def test_leading_non_unit_dim_moves_behind_unit_dim(
        self, _: str, leading: tuple[int, ...]
    ) -> None:
        ep, inp = self._build(leading)
        expected = self._run(ep, inp)

        result = CanonicalizeLinearInputRank()(ep.graph_module)
        assert result is not None

        self.assertTrue(result.modified)
        linear = self._linear(ep)
        self.assertEqual(
            list(get_arg(linear, "inp", fx.Node).meta["val"].shape),
            [1, 50, _IN_FEATURES],
        )
        self.assertEqual(list(linear.meta["val"].shape), [1, 50, _OUT_FEATURES])
        torch.testing.assert_close(self._run(ep, inp), expected)

    @parameterized.expand(
        [
            ("single_row", (1, 1)),
            ("two_non_unit_dims", (2, 50)),
            ("already_canonical", (1, 50)),
        ]
    )
    def test_other_shapes_unchanged(self, _: str, leading: tuple[int, ...]) -> None:
        ep, _inp = self._build(leading)
        result = CanonicalizeLinearInputRank()(ep.graph_module)
        assert result is not None
        self.assertFalse(result.modified)

    def test_non_view_user_unchanged(self) -> None:
        ep, _ = self._build((50, 1), user_is_view=False)
        result = CanonicalizeLinearInputRank()(ep.graph_module)
        assert result is not None
        self.assertFalse(result.modified)
