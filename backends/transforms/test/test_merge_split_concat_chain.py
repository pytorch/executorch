# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import operator
import unittest
from collections.abc import Callable, Sequence
from typing import cast

import torch
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.backends.transforms.merge_split_concat_chain import (
    MergeSplitConcatChainPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.fx import GraphModule


def _build_split_cat_graph(
    input_value: torch.Tensor,
    split_target: Callable[..., object],
    split_spec: int | list[int],
    split_dim: int,
    output_count: int,
    cat_target: Callable[..., object],
    cat_dim: int,
    order: Sequence[int] | None = None,
) -> GraphModule:
    builder = GraphBuilder()
    input_node = builder.placeholder("input", input_value)
    split = builder.call_operator(
        split_target,
        (input_node, split_spec, split_dim),
    )
    output_order = range(output_count) if order is None else order
    split_outputs = [
        builder.call_operator(operator.getitem, (split, index))
        for index in output_order
    ]
    cat = builder.call_operator(cat_target, (split_outputs, cat_dim))
    builder.output([cat])
    return builder.get_graph_module()


_VIEW_EQUIVALENT_CASES = (
    (
        "split",
        torch.ops.aten.split.Tensor,
        2,
        (2, 1, 6, 4, 4),
        2,
        3,
        torch.ops.aten.cat.default,
        1,
        torch.ops.aten.view_copy.default,
    ),
    (
        "chunk",
        torch.ops.aten.chunk.default,
        3,
        (2, 1, 5, 4, 4),
        2,
        3,
        torch.ops.aten.cat.default,
        2,
        torch.ops.aten.view_copy.default,
    ),
    (
        "split_with_sizes",
        torch.ops.aten.split_with_sizes.default,
        [1, 1, 1, 1],
        (1, 4, 3, 2),
        1,
        4,
        torch.ops.aten.cat.default,
        0,
        torch.ops.aten.view_copy.default,
    ),
    (
        "edge_split_with_sizes",
        exir_ops.edge.aten.split_with_sizes_copy.default,
        [1, 1, 1, 1],
        (4, 1, 3, 2),
        0,
        4,
        exir_ops.edge.aten.cat.default,
        1,
        exir_ops.edge.aten.view_copy.default,
    ),
)

_UNSAFE_CASES = (
    ("reordered_outputs", [1, 0, 2, 3], 0, False),
    ("non_singleton_crossing", [0, 1, 2, 3], 3, False),
    ("non_contiguous_input", [0, 1, 2, 3], 0, True),
)

_SPLIT_CAT_DIALECTS = (
    (
        "aten",
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.cat.default,
    ),
    (
        "edge",
        exir_ops.edge.aten.split_with_sizes_copy.default,
        exir_ops.edge.aten.cat.default,
    ),
)


class MergeSplitConcatChainPassTest(unittest.TestCase):
    def test_merges_view_equivalent_chains(self) -> None:
        for (
            name,
            split_target,
            split_spec,
            input_shape,
            split_dim,
            output_count,
            cat_target,
            cat_dim,
            view_target,
        ) in _VIEW_EQUIVALENT_CASES:
            with self.subTest(name=name):
                input_value = torch.randn(input_shape)
                graph_module = _build_split_cat_graph(
                    input_value,
                    split_target,
                    split_spec,
                    split_dim,
                    output_count,
                    cat_target,
                    cat_dim,
                )
                reference = copy.deepcopy(graph_module)

                result = cast(PassResult, MergeSplitConcatChainPass()(graph_module))

                self.assertTrue(result.modified)
                torch.testing.assert_close(
                    reference(input_value), result.graph_module(input_value)
                )
                for target, expected_count in (
                    (split_target, 0),
                    (cat_target, 0),
                    (view_target, 1),
                ):
                    self.assertEqual(
                        expected_count,
                        len(
                            result.graph_module.graph.find_nodes(
                                op="call_function", target=target
                            )
                        ),
                    )

    def test_does_not_merge_unsafe_chains(self) -> None:
        for dialect, split_target, cat_target in _SPLIT_CAT_DIALECTS:
            for name, order, cat_dim, non_contiguous in _UNSAFE_CASES:
                with self.subTest(dialect=dialect, name=name):
                    input_value = torch.randn(1, 4, 3, 2)
                    if non_contiguous:
                        input_value = input_value.transpose(2, 3)
                    graph_module = _build_split_cat_graph(
                        input_value,
                        split_target,
                        [1, 1, 1, 1],
                        1,
                        4,
                        cat_target,
                        cat_dim,
                        order,
                    )
                    reference = copy.deepcopy(graph_module)

                    result = cast(PassResult, MergeSplitConcatChainPass()(graph_module))

                    self.assertFalse(result.modified)
                    torch.testing.assert_close(
                        reference(input_value), result.graph_module(input_value)
                    )
                    self.assertEqual(
                        1,
                        len(
                            result.graph_module.graph.find_nodes(
                                op="call_function",
                                target=cat_target,
                            )
                        ),
                    )
