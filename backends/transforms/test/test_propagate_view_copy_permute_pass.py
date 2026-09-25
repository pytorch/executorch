# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import pytest
import torch
from executorch.backends.transforms.propagate_view_copy_permute_pass import (
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

PERMUTE = exir_ops.edge.aten.permute_copy.default
ABS = exir_ops.edge.aten.abs.default
NEG = exir_ops.edge.aten.neg.default
SIGMOID = exir_ops.edge.aten.sigmoid.default
ADD = exir_ops.edge.aten.add.Tensor
SUM = exir_ops.edge.aten.sum.dim_IntList
WHERE = exir_ops.edge.aten.where.self

SOURCE_SHAPE = (1, 2, 3, 4)
PERMUTED_SHAPE = (1, 3, 4, 2)
PERMUTATION = [0, 2, 3, 1]


def _forked_permute(graph: torch.fx.Graph, branches: int) -> list[torch.fx.Node]:
    """A permute feeding `branches` pointwise users."""
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SOURCE_SHAPE)
    permute = graph.call_function(PERMUTE, args=(x, PERMUTATION))
    permute.meta["val"] = torch.empty(PERMUTED_SHAPE)

    users = []
    for target in (ABS, NEG, SIGMOID)[:branches]:
        user = graph.call_function(target, args=(permute,))
        user.meta["val"] = torch.empty(PERMUTED_SHAPE)
        users.append(user)
    return users


def _permute_counts(graph: torch.fx.Graph) -> tuple[int, int, int]:
    """Permute counts before propagation, after the down pass, and after the up pass."""
    graph.lint()
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    def count() -> int:
        return [
            node.target
            for node in graph_module.graph.nodes
            if node.op == "call_function"
        ].count(PERMUTE)

    before = count()
    graph_module = PropagateViewCopyPermuteDownPass().call(graph_module).graph_module
    after_down = count()
    graph_module = PropagateViewCopyPermuteUpPass().call(graph_module).graph_module
    return before, after_down, count()


@pytest.mark.xfail(
    strict=True,
    reason="Splitting a fork where some branches rejoin and others do not leaves "
    "one copy below the meeting node and one at the source, and the up pass has "
    "no fork split of its own to hoist the first above the rejoin. No model in a "
    "15-model sweep produces this shape, so the driver does not special-case it; "
    "the general fix is to stop propagation increasing the copy count at all.",
)
def test_mixed_reconvergence_fork_does_not_strand_a_permute() -> None:
    graph = torch.fx.Graph()
    left, right, diverging = _forked_permute(graph, branches=3)
    rejoin = graph.call_function(ADD, args=(left, right))
    rejoin.meta["val"] = torch.empty(PERMUTED_SHAPE)
    graph.output((rejoin, diverging))

    assert _permute_counts(graph) == (1, 1, 1)


def test_fork_split_still_applies_when_every_branch_rejoins() -> None:
    graph = torch.fx.Graph()
    left, right = _forked_permute(graph, branches=2)
    rejoin = graph.call_function(ADD, args=(left, right))
    rejoin.meta["val"] = torch.empty(PERMUTED_SHAPE)
    graph.output(rejoin)

    assert _permute_counts(graph) == (1, 1, 1)


def test_fork_split_still_applies_when_every_branch_diverges() -> None:
    graph = torch.fx.Graph()
    left, right = _forked_permute(graph, branches=2)
    graph.output((left, right))

    # The down pass gives each branch its own copy; the up pass merges them
    # back onto the shared producer.
    assert _permute_counts(graph) == (1, 2, 1)


class _BlockAddToAddPass(PropagateViewCopyPermuteDownPass):
    def blocks_moving(
        self,
        moving_node: torch.fx.Node,
        frontier: torch.fx.Node,
        next_nodes: Sequence[torch.fx.Node],
    ) -> bool:
        return frontier.target == ADD and any(
            next_node.target == ADD for next_node in next_nodes
        )


def test_down_pass_remembers_path_blocked_before_shared_node() -> None:
    """Keep a permute when a blocked path reaches a shared node first.

    Here ``P`` means permute and ``X`` marks the blocked connection:

                    +-> abs --+
                    |         v
        P(x) --------+-------> add_1 --X-> add_2 -> output
                    |                     ^
                    +-> neg --------------+

    The depth-first search sees ``add_1 -> add_2`` before ``neg -> add_2``.
    The blocked connection must still prevent the whole region from moving.

    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SOURCE_SHAPE)
    permute = graph.call_function(PERMUTE, args=(x, PERMUTATION))
    permute.meta["val"] = torch.empty(PERMUTED_SHAPE)
    absolute = graph.call_function(ABS, args=(permute,))
    absolute.meta["val"] = torch.empty(PERMUTED_SHAPE)
    neg = graph.call_function(NEG, args=(permute,))
    neg.meta["val"] = torch.empty(PERMUTED_SHAPE)
    inner_add = graph.call_function(ADD, args=(permute, absolute))
    inner_add.meta["val"] = torch.empty(PERMUTED_SHAPE)
    outer_add = graph.call_function(ADD, args=(inner_add, neg))
    outer_add.meta["val"] = torch.empty(PERMUTED_SHAPE)
    graph.output(outer_add)
    graph.lint()
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = _BlockAddToAddPass().call(graph_module)
    targets = [
        node.target
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
    ]

    assert targets == [PERMUTE, ABS, NEG, ADD, ADD]


def test_is_swappable_declines_reduction_that_drops_the_dimension() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    pass_ = PropagateViewCopyPermuteUpPass()

    assert pass_.is_swappable(graph.call_function(SUM, args=(x, [1], True)))
    assert not pass_.is_swappable(graph.call_function(SUM, args=(x, [1], False)))


@pytest.mark.parametrize("shape", [(1, 3, 5), (1, 5, 5)])
def test_upward_permute_preserves_broadcast_comparison(shape: tuple[int, ...]) -> None:
    # Unequal axes expose reshape errors; equal axes expose incorrect mask values.
    values = torch.arange(shape[1] * shape[2]).reshape(shape) - 2
    limit = torch.zeros((1, 1, 1), dtype=values.dtype)
    graph = torch.fx.Graph()
    scalar = graph.placeholder("scalar")
    scalar.meta["val"] = limit
    data = graph.placeholder("data")
    data.meta["val"] = values
    comparison = graph.call_function(exir_ops.edge.aten.ge.Tensor, args=(scalar, data))
    comparison.meta["val"] = torch.ge(limit, values)
    permute = graph.call_function(PERMUTE, args=(comparison, [0, 2, 1]))
    permute.meta["val"] = comparison.meta["val"].permute(0, 2, 1)
    reduced = graph.call_function(exir_ops.edge.aten.any.dim, args=(permute, -1, True))
    reduced.meta["val"] = permute.meta["val"].any(dim=-1, keepdim=True)
    output = graph.call_function(
        exir_ops.edge.aten.view_copy.default, args=(reduced, [1, shape[2]])
    )
    output.meta["val"] = reduced.meta["val"].reshape(1, shape[2])
    graph.output(output)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    expected = graph_module(limit, values)

    result = PropagateViewCopyPermuteUpPass().call(graph_module)

    torch.testing.assert_close(result.graph_module(limit, values), expected)


def test_upward_permute_moves_through_broadcast_where() -> None:
    inputs = (
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.arange(15, dtype=torch.float32).reshape(1, 3, 5),
        torch.zeros((1, 1, 1)),
    )
    graph = torch.fx.Graph()
    condition, data, other = [
        graph.placeholder(name) for name in ("condition", "data", "other")
    ]
    for node, value in zip((condition, data, other), inputs):
        node.meta["val"] = value
    where = graph.call_function(WHERE, args=(condition, data, other))
    where.meta["val"] = torch.where(*inputs)
    permute = graph.call_function(PERMUTE, args=(where, [0, 2, 1]))
    permute.meta["val"] = where.meta["val"].permute(0, 2, 1)
    graph.output(permute)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    expected = graph_module(*inputs)

    result = PropagateViewCopyPermuteUpPass().call(graph_module)

    call_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert [node.target for node in call_nodes] == [PERMUTE, WHERE]
    assert call_nodes[1].args[1] is call_nodes[0]
    torch.testing.assert_close(result.graph_module(*inputs), expected)
