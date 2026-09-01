# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def test_is_swappable_declines_reduction_that_drops_the_dimension() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    pass_ = PropagateViewCopyPermuteUpPass()

    assert pass_.is_swappable(graph.call_function(SUM, args=(x, [1], True)))
    assert not pass_.is_swappable(graph.call_function(SUM, args=(x, [1], False)))
