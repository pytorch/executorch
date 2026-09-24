# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.optimization_passes.sink_constant_cat import (
    SinkConstantCat,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torch.export.graph_signature import InputKind, OutputKind


class _CatReluConstModel(torch.nn.Module):
    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.cat([x, self.const], dim=1))


class _CatPermuteConstModel(torch.nn.Module):
    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.const], dim=1).permute(0, 2, 1)


class _CatReluPermuteConstModel(torch.nn.Module):
    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.cat([x, self.const], dim=1)
        r = torch.relu(c)
        return r.permute(0, 2, 1)


class _CatViewNegativeDimConstModel(torch.nn.Module):
    """cat on dim=2, then view with -1 on a non-cat dimension.

    Input x is [2, 3, 4], const is [2, 3, 4], cat on dim=2 gives [2, 3, 8].
    The view reshapes to [2, -1] = [2, 24], merging dims 1 and 2.
    Since cat dim (2) is merged with dim 1, the tracker invalidates and
    the pass bails — but critically, the -1 must be resolved via
    meta["val"].shape, not the raw arg (which would contain -1).
    """

    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.cat([x, self.const], dim=2)  # [2, 3, 8]
        return c.view(c.shape[0], -1)  # [2, 24]


class _CatViewNegativeDimPreservesCatModel(torch.nn.Module):
    """cat on dim=1, then view with -1 that merges only non-cat dims.

    Input x is [2, 4, 2, 3], const is [2, 4, 2, 3], cat on dim=1 gives
    [2, 8, 2, 3]. The view reshapes to [2, 8, -1] = [2, 8, 6], merging
    dims 2 and 3 while cat dim (1) stays intact.
    """

    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.cat([x, self.const], dim=1)  # [2, 8, 2, 3]
        return c.view(c.shape[0], c.shape[1], -1)  # [2, 8, 6]


class _CatUnsqueezeConstModel(torch.nn.Module):
    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.cat([x, self.const], dim=0)
        return c.unsqueeze(0)


class _CatMultiUserConstModel(torch.nn.Module):
    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.cat([x, self.const], dim=1)
        a = torch.relu(c)
        b = c.permute(0, 2, 1)
        return a, b


class _CatThreeInputsOneConstModel(torch.nn.Module):
    """Two activations + one constant. The cat should shrink to two elements."""

    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.cat([x, y, self.const], dim=1))


class _CatAllActivationsModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.cat([x, y], dim=1))


class _CatPartialSinkableModel(torch.nn.Module):
    """Cat with two users: relu (sinkable) and matmul (not sinkable).

    Should bail because not all users are sinkable.
    """

    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.cat([x, self.const], dim=1)
        a = torch.relu(c)
        b = torch.bmm(c, c.transpose(1, 2))
        return a, b


class _CatSliceOnNonCatDimModel(torch.nn.Module):
    """cat on dim=1, then slice on dim=2 (not the cat dim). Safe to sink."""

    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.cat([x, self.const], dim=1)
        return c[:, :, 2:6]


class _CatSliceOnCatDimModel(torch.nn.Module):
    """cat on dim=1, then slice on the same dimension.

    Sinking would break because slicing on the cat dimension can cross
    input boundaries — the result cannot be reproduced by slicing each
    cat input independently.
    """

    const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("const", torch.randn(2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.cat([x, self.const], dim=1)
        return c[:, 2:6, :]


class SinkConstantCatTest(unittest.TestCase):
    def _run_pass(
        self,
        model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.fx.GraphModule, bool]:
        exported = export(model, inputs)
        edge = to_edge(exported, compile_config=EdgeCompileConfig(_skip_dim_order=True))
        ep = edge.exported_program()

        with torch.no_grad():
            ref_output = ep.module()(*inputs)

        result = SinkConstantCat()(ep)

        with torch.no_grad():
            new_output = result.exported_program.module()(*inputs)

        if isinstance(ref_output, torch.Tensor):
            self.assertTrue(
                (new_output == ref_output).all(),
                "Pass should not change numerical output",
            )
        else:
            for i, (new, ref) in enumerate(zip(new_output, ref_output)):
                self.assertTrue(
                    (new == ref).all(),
                    f"Pass changed numerical output at index {i}",
                )
        return result.exported_program.graph_module, result.modified

    def test_cat_relu_pushed(self) -> None:
        x = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatReluConstModel(), (x,))
        self.assertTrue(modified)

    def test_cat_permute_pushed(self) -> None:
        x = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatPermuteConstModel(), (x,))
        self.assertTrue(modified)

    def test_cat_relu_permute_pushed(self) -> None:
        x = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatReluPermuteConstModel(), (x,))
        self.assertTrue(modified)

    def test_cat_unsqueeze_pushed(self) -> None:
        x = torch.randn(4, 8)
        _, modified = self._run_pass(_CatUnsqueezeConstModel(), (x,))
        self.assertTrue(modified)

    def test_cat_multi_user_pushed(self) -> None:
        """Each user of the cat gets its own sunk chain.

        Before: cat([x, const]) → relu, permute
        After:  relu(x), relu(const_folded) → cat → output_a
                permute(x), permute(const_folded) → cat → output_b
        """
        x = torch.randn(2, 4, 8)
        gm, modified = self._run_pass(_CatMultiUserConstModel(), (x,))
        self.assertTrue(modified)

        graph = gm.graph

        cat_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.cat.default
        )
        self.assertEqual(len(cat_nodes), 2, "Should have one sunk cat per user path")

        relu_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.relu.default
        )
        self.assertEqual(
            len(relu_nodes), 1, "Should have one relu for the activation path"
        )

        permute_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )
        self.assertEqual(
            len(permute_nodes), 1, "Should have one permute for the activation path"
        )

        for cat_node in cat_nodes:
            tensors = cat_node.args[0]
            assert isinstance(tensors, (list, tuple))
            self.assertEqual(
                len(tensors), 2, "Each sunk cat should merge activation + folded const"
            )

    def test_skips_activation_only_inputs(self) -> None:
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatAllActivationsModel(), (x, y))
        self.assertFalse(modified)

    def test_skips_mutable_buffer(self) -> None:
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.randn(2, 4, 8))
        state = builder.placeholder(
            "state", torch.randn(2, 4, 8), input_kind=InputKind.BUFFER
        )
        cat = builder.call_operator(
            op=exir_ops.edge.aten.cat.default, args=([x, state], 1)
        )
        relu = builder.call_operator(op=exir_ops.edge.aten.relu.default, args=(cat,))
        builder.output(
            [state, relu],
            output_kinds=[OutputKind.BUFFER_MUTATION, OutputKind.USER_OUTPUT],
            output_targets=["state", None],
        )
        program = builder.get_program()
        self.assertTrue(program.graph_signature.buffers_to_mutate)

        result = SinkConstantCat()(program)

        self.assertFalse(result.modified)
        graph = result.exported_program.graph_module.graph
        (cat_node,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.cat.default
        )
        (relu_node,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.relu.default
        )
        state_node = next(node for node in graph.nodes if node.name == "state")
        cat_inputs = cat_node.args[0]
        assert isinstance(cat_inputs, (list, tuple))
        self.assertIn(state_node, cat_inputs)
        self.assertIs(relu_node.args[0], cat_node)

    def test_skips_when_not_all_users_sinkable(self) -> None:
        """Should bail when only some users of the cat are sinkable."""
        x = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatPartialSinkableModel(), (x,))
        self.assertFalse(modified)

    def test_three_inputs_one_const_keeps_activation_cat(self) -> None:
        """With [act, act, const], the original cat should shrink to [act, act]."""
        x = torch.randn(2, 4, 8)
        y = torch.randn(2, 4, 8)
        model = _CatThreeInputsOneConstModel()

        exported = export(model, (x, y))
        edge = to_edge(exported, compile_config=EdgeCompileConfig(_skip_dim_order=True))
        ep = edge.exported_program()

        with torch.no_grad():
            ref_output = ep.module()(x, y)

        result = SinkConstantCat()(ep)
        self.assertTrue(result.modified)

        with torch.no_grad():
            new_output = result.exported_program.module()(x, y)
        self.assertTrue(
            (new_output == ref_output).all(),
            "Pass should not change numerical output",
        )

        cat_nodes = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.cat.default
        )
        # Should have 2 cats: one with [act, act] feeding the chain,
        # and one at the sink merging the activation path with the folded const
        self.assertEqual(len(cat_nodes), 2)
        for cat_node in cat_nodes:
            tensors = cat_node.args[0]
            assert isinstance(tensors, (list, tuple))
            self.assertEqual(len(tensors), 2)

    def test_cat_slice_on_non_cat_dim_pushed(self) -> None:
        """Should sink when a slice is on a dimension other than the cat dim."""
        x = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatSliceOnNonCatDimModel(), (x,))
        self.assertTrue(modified)

    def test_skips_slice_on_cat_dim(self) -> None:
        """Should not sink when a slice is applied on the cat dimension,
        since the slice range may cross cat input boundaries."""
        x = torch.randn(2, 4, 8)
        _, modified = self._run_pass(_CatSliceOnCatDimModel(), (x,))
        self.assertFalse(modified)

    def test_skips_view_that_merges_cat_dim(self) -> None:
        """Should not sink when a view merges the cat dimension with another
        dimension, since the cat can no longer be reconstructed after the reshape."""
        x = torch.randn(2, 3, 4)
        _, modified = self._run_pass(_CatViewNegativeDimConstModel(), (x,))
        self.assertFalse(modified)

    def test_cat_view_negative_dim_preserves_cat_dim(self) -> None:
        """view with -1 that merges only non-cat dims should allow sinking
        and produce numerically correct results."""
        x = torch.randn(2, 4, 2, 3)
        _, modified = self._run_pass(_CatViewNegativeDimPreservesCatModel(), (x,))
        self.assertTrue(modified)
