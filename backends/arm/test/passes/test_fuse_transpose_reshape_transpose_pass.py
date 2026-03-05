# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FuseTransposeReshapeTransposePass.
"""

import unittest

import torch
from executorch.backends.arm._passes.fuse_transpose_reshape_transpose_pass import (
    FuseTransposeReshapeTransposePass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline


class TransposeReshapeTransposeModel(torch.nn.Module):
    """Model with transpose -> reshape -> transpose pattern."""

    def __init__(self, dims1: list[int], shape: list[int], dims2: list[int]) -> None:
        super().__init__()
        self.dims1 = dims1
        self.shape = shape
        self.dims2 = dims2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(self.dims1)
        x = x.view(self.shape)
        x = x.permute(self.dims2)
        return x


class SingleTransposeModel(torch.nn.Module):
    """Model with only a single transpose - should not be modified."""

    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class TransposeReshapeModel(torch.nn.Module):
    """Model with transpose -> reshape but no second transpose."""

    def __init__(self, dims: list[int], shape: list[int]) -> None:
        super().__init__()
        self.dims = dims
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(self.dims)
        x = x.view(self.shape)
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


class TestFuseTransposeReshapeTransposePass(unittest.TestCase):
    """Tests for FuseTransposeReshapeTransposePass."""

    def test_basic_fusion(self) -> None:
        """Test basic fusion of transpose -> reshape -> transpose."""
        model = TransposeReshapeTransposeModel(
            dims1=[0, 2, 3, 1],
            shape=[1, 8, 8, 64],
            dims2=[0, 3, 1, 2],
        )
        example_input = torch.randn(1, 64, 8, 8)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        pipeline.run_passes([FuseTransposeReshapeTransposePass()])
        graph_module = pipeline.graph_module

        permute_count = _count_permute_nodes(graph_module)
        view_count = _count_view_nodes(graph_module)

        self.assertEqual(permute_count, 1, "Expected exactly 1 permute after fusion")
        self.assertEqual(view_count, 1, "Expected exactly 1 view after fusion")

    def test_numerical_correctness(self) -> None:
        """Verify that the pass produces numerically correct results."""
        model = TransposeReshapeTransposeModel(
            dims1=[0, 2, 3, 1],
            shape=[1, 8, 8, 64],
            dims2=[0, 3, 1, 2],
        )
        example_input = torch.randn(1, 64, 8, 8)

        expected_output = model(example_input)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        pipeline.run_passes([FuseTransposeReshapeTransposePass()])

        actual_output = pipeline.graph_module(example_input)

        torch.testing.assert_close(actual_output, expected_output)

    def test_single_transpose_unchanged(self) -> None:
        """Ensure single transpose models are not modified."""
        model = SingleTransposeModel(dims=[0, 2, 3, 1])
        example_input = torch.randn(1, 64, 8, 8)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        permute_count_before = _count_permute_nodes(pipeline.graph_module)

        pipeline.run_passes([FuseTransposeReshapeTransposePass()])

        permute_count_after = _count_permute_nodes(pipeline.graph_module)

        self.assertEqual(permute_count_before, permute_count_after)

    def test_transpose_reshape_only_unchanged(self) -> None:
        """Ensure transpose -> reshape (without second transpose) is not modified."""
        model = TransposeReshapeModel(dims=[0, 2, 3, 1], shape=[1, 512])
        example_input = torch.randn(1, 64, 8, 1)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        permute_count_before = _count_permute_nodes(pipeline.graph_module)
        view_count_before = _count_view_nodes(pipeline.graph_module)

        pipeline.run_passes([FuseTransposeReshapeTransposePass()])

        permute_count_after = _count_permute_nodes(pipeline.graph_module)
        view_count_after = _count_view_nodes(pipeline.graph_module)

        self.assertEqual(permute_count_before, permute_count_after)
        self.assertEqual(view_count_before, view_count_after)

    def test_nchw_to_nhwc_pattern(self) -> None:
        """Test the NCHW -> NHWC reordering pattern from the docstring."""
        model = TransposeReshapeTransposeModel(
            dims1=[0, 3, 1, 2],
            shape=[1, 512, 8, 8],
            dims2=[0, 2, 3, 1],
        )
        example_input = torch.randn(1, 8, 8, 64)

        expected_output = model(example_input)

        pipeline = PassPipeline[torch.nn.Module](model, example_input)
        pipeline.run_passes([FuseTransposeReshapeTransposePass()])

        permute_count = _count_permute_nodes(pipeline.graph_module)
        view_count = _count_view_nodes(pipeline.graph_module)

        self.assertEqual(permute_count, 1)
        self.assertEqual(view_count, 1)

        actual_output = pipeline.graph_module(example_input)
        torch.testing.assert_close(actual_output, expected_output)


if __name__ == "__main__":
    unittest.main()
