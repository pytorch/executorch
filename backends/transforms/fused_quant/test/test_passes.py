# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.fused_quant.optimization_passes.fuse_add_into_linear import (
    FuseAddIntoLinear,
)
from executorch.backends.transforms.fused_quant.optimization_passes.fuse_mul_into_linear import (
    FuseMulIntoLinear,
)
from executorch.backends.transforms.fused_quant.optimization_passes.prequantize_embedding import (
    PrequantizeEmbedding,
)
from executorch.backends.transforms.fused_quant.optimization_passes.split_linear_at_slices import (
    SplitLinearAtSlices,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_last import (
    ToChannelsLast,
)
from executorch.backends.transforms.fused_quant.passes import (
    _can_fuse_last_dim_softmax_through_view,
    _can_fuse_square_linear_through_view,
    get_optimization_passes,
)
from executorch.backends.transforms.fused_quant.test.helpers import (
    create_per_axis_qparams,
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.fx.passes.dialect.common.cse_pass import CSEPass

_PASSES_BY_FLAG: dict[str, set[type]] = {
    "channels_last": {ToChannelsLast},
    "fold_into_linear": {FuseAddIntoLinear, FuseMulIntoLinear},
    "prequantize_embedding": {PrequantizeEmbedding},
    "split_linear_at_slices": {SplitLinearAtSlices},
}


def _pass_types(**flags: bool) -> set[type]:
    return {type(pass_) for pass_ in get_optimization_passes(**flags)._passes}


class OptimizationPassesTest(unittest.TestCase):
    def test_flags_are_off_by_default(self) -> None:
        flagged = set().union(*_PASSES_BY_FLAG.values())

        self.assertEqual(_pass_types() & flagged, set())

    @parameterized.expand(list(_PASSES_BY_FLAG.items()))
    def test_each_flag_adds_only_its_passes(
        self, flag: str, expected: set[type]
    ) -> None:
        added = _pass_types(**{flag: True}) - _pass_types()

        self.assertEqual(added, expected)

    def test_cse_keeps_separate_random_draws(self) -> None:
        class RandTwice(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.rand(2, 3) - torch.rand(2, 3)

        graph_module = (
            to_edge(torch.export.export(RandTwice(), (torch.randn(2, 3),)))
            .exported_program()
            .graph_module
        )
        cse = next(
            pass_
            for pass_ in get_optimization_passes()._passes
            if isinstance(pass_, CSEPass)
        )

        result = cse(graph_module)

        rand_nodes = result.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.rand.default
        )
        self.assertEqual(len(rand_nodes), 2)


def _view_linear_view(
    per_axis_activations: bool,
) -> tuple[torch.fx.Node, torch.fx.Node]:
    """x[2, 4] -> view [1, 2, 4] -> fused linear -> view [2, 4], with per-tensor
    or per-token (axis 1) activation qparams. Returns the linear and the last
    view."""
    builder = ProgramBuilder()
    x = builder.placeholder("x", torch.randint(-128, 127, (2, 4)).to(torch.int8))
    weight = builder.placeholder(
        "weight", torch.randint(-16, 16, (4, 4)).to(torch.int8)
    )
    if per_axis_activations:
        inp_qparams = create_per_axis_qparams(
            builder, 2, ndim=3, axis=1, dtype=torch.float32
        )
        out_qparams = create_per_axis_qparams(builder, 2, ndim=3, axis=1)
    else:
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        out_qparams = create_per_tensor_qparams(builder)
    weight_qparams = create_per_channel_qparams(builder, 4, dtype=torch.float32)
    view = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 2, 4])
    )
    linear = builder.call_operator(
        op=exir_ops.edge.fused_quant.linear.default,
        args=(
            view,
            weight,
            None,
            *inp_qparams,
            *weight_qparams,
            None,
            None,
            torch.float32,
            0,
            0,
            *out_qparams,
        ),
    )
    final_view = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default, args=(linear, [2, 4])
    )
    builder.output([final_view])
    graph = builder.get_program().graph_module.graph
    (linear_node,) = graph.find_nodes(
        op="call_function", target=exir_ops.edge.fused_quant.linear.default
    )
    (final_view_node,) = linear_node.users
    return linear_node, final_view_node


class FuseViewFiltersTest(unittest.TestCase):
    def test_softmax_not_fused_through_a_rank_0_view(self) -> None:
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(1))
        softmax = builder.call_operator(
            op=exir_ops.edge.aten._softmax.default, args=(x, -1, False)
        )
        builder.output(
            [
                builder.call_operator(
                    op=exir_ops.edge.aten.view_copy.default, args=(softmax, [])
                )
            ]
        )
        graph = builder.get_program().graph_module.graph
        (softmax_node,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten._softmax.default
        )
        (final_view,) = softmax_node.users

        self.assertFalse(
            _can_fuse_last_dim_softmax_through_view(softmax_node, final_view)
        )

    def test_linear_fused_through_views_with_per_tensor_activations(self) -> None:
        self.assertTrue(_can_fuse_square_linear_through_view(*_view_linear_view(False)))

    def test_linear_not_fused_through_views_with_per_axis_activations(
        self,
    ) -> None:
        """The fold changes the linear's rank, which per-axis qparams encode."""
        self.assertFalse(_can_fuse_square_linear_through_view(*_view_linear_view(True)))
