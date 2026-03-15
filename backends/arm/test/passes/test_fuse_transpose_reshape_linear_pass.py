# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FuseTransposeReshapeLinearPass.
"""

import unittest

import torch
from executorch.backends.arm._passes.fuse_transpose_reshape_linear_pass import (
    FuseTransposeReshapeLinearPass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline


class TransposeReshapeLinearModel(torch.nn.Module):
    """Model with transpose -> reshape -> linear pattern."""

    def __init__(
        self, dims: list[int], in_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.dims = dims
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(self.dims)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.linear(x)
        return x


class TransposeReshapeLinearWithBiasModel(torch.nn.Module):
    """Model with transpose -> reshape -> linear (with bias) pattern."""

    def __init__(
        self, dims: list[int], in_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.dims = dims
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(self.dims)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.linear(x)
        return x


class SingleLinearModel(torch.nn.Module):
    """Model with just a linear layer - should not be modified."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TransposeModifiesBatchModel(torch.nn.Module):
    """Model where transpose modifies batch dim - should NOT be fused."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2, 3)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


def _count_permute_nodes(graph_module: torch.fx.GraphModule) -> int:
    """Count the number of permute_copy nodes in the graph."""
    count = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.permute_copy.default:
            count += 1
    return count


def _count_view_nodes(graph_module: torch.fx.GraphModule) -> int:
    """Count the number of view_copy nodes in the graph."""
    count = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target in (
            torch.ops.aten.view_copy.default,
            torch.ops.aten._unsafe_view.default,
        ):
            count += 1
    return count


class TestFuseTransposeReshapeLinearPass(unittest.TestCase):
    """Tests for FuseTransposeReshapeLinearPass."""

    def test_basic_fusion(self) -> None:
        """Test basic fusion of transpose -> reshape -> linear."""
        model = TransposeReshapeLinearModel(
            dims=[0, 2, 3, 1],
            in_features=512,
            out_features=128,
        )
        example_input = torch.randn(1, 64, 8, 1)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        permute_count_before = _count_permute_nodes(pipeline.graph_module)

        pipeline.run_passes([FuseTransposeReshapeLinearPass()])

        permute_count_after = _count_permute_nodes(pipeline.graph_module)

        self.assertLess(
            permute_count_after,
            permute_count_before,
            "Expected permute nodes to be reduced after fusion",
        )

    def test_transpose_modifies_batch_not_fused(self) -> None:
        """Ensure transpose that modifies batch dim is NOT fused."""
        model = TransposeModifiesBatchModel(
            in_features=512,
            out_features=128,
        )
        example_input = torch.randn(2, 1, 64, 8)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        permute_count_before = _count_permute_nodes(pipeline.graph_module)

        pipeline.run_passes([FuseTransposeReshapeLinearPass()])

        permute_count_after = _count_permute_nodes(pipeline.graph_module)

        self.assertEqual(
            permute_count_before,
            permute_count_after,
            "Should NOT fuse when transpose modifies batch dimension",
        )

    def test_single_linear_unchanged(self) -> None:
        """Ensure single linear models are not modified."""
        model = SingleLinearModel(in_features=512, out_features=128)
        example_input = torch.randn(1, 512)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        permute_count_before = _count_permute_nodes(pipeline.graph_module)
        view_count_before = _count_view_nodes(pipeline.graph_module)

        pipeline.run_passes([FuseTransposeReshapeLinearPass()])

        permute_count_after = _count_permute_nodes(pipeline.graph_module)
        view_count_after = _count_view_nodes(pipeline.graph_module)

        self.assertEqual(permute_count_before, permute_count_after)
        self.assertEqual(view_count_before, view_count_after)

    def test_nhwc_to_flatten_pattern(self) -> None:
        """Test the common NHWC flatten pattern."""
        model = TransposeReshapeLinearModel(
            dims=[0, 2, 3, 1],
            in_features=512,
            out_features=256,
        )
        example_input = torch.randn(1, 64, 8, 1)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        pipeline.run_passes([FuseTransposeReshapeLinearPass()])

        permute_count = _count_permute_nodes(pipeline.graph_module)

        self.assertEqual(permute_count, 0, "Expected all permutes to be fused")


if __name__ == "__main__":
    unittest.main()
