# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from executorch.backends.transforms.fuse_duplicate_users_pass import (
    FuseDuplicateUsersPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

ADD = exir_ops.edge.aten.add.Tensor
RAND_LIKE = exir_ops.edge.aten.rand_like.default
RELU = exir_ops.edge.aten.relu.default
RELU_ = torch.ops.aten.relu_.default
ATEN_NEG = torch.ops.aten.neg.default
ATEN_VIEW = torch.ops.aten.view.default
ATEN_CLONE = torch.ops.aten.clone.default
STACK = exir_ops.edge.aten.stack.default

SHAPE = (1, 2, 3, 4)


def _count(graph_module: torch.fx.GraphModule, target: object) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def _two_users(target: object, returned: int = 0) -> torch.fx.GraphModule:
    """A placeholder with two identical users of ``target``.

    ``returned`` says how many of them the graph returns directly; the rest are
    consumed by an interior add so they stay live.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)

    users = []
    for _ in range(2):
        user = graph.call_function(target, args=(x,))
        user.meta["val"] = torch.empty(SHAPE)
        users.append(user)

    outputs = list(users[:returned])
    for interior in users[returned:]:
        consumed = graph.call_function(ADD, args=(interior, x))
        consumed.meta["val"] = torch.empty(SHAPE)
        outputs.append(consumed)

    graph.output(tuple(outputs))
    graph.lint()
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_identical_interior_users_are_fused() -> None:
    graph_module = _two_users(RELU, returned=0)

    result = FuseDuplicateUsersPass()(graph_module)

    assert result.modified
    assert _count(result.graph_module, RELU) == 1


def test_seeded_users_are_not_fused() -> None:
    """Two draws are two draws."""
    graph_module = _two_users(RAND_LIKE, returned=0)

    result = FuseDuplicateUsersPass()(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RAND_LIKE) == 2


def test_mutating_users_are_not_fused() -> None:
    """A mutating operator applies its effect once per call, not once."""
    graph_module = _two_users(RELU_, returned=0)

    result = FuseDuplicateUsersPass()(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RELU_) == 2


def test_two_returned_duplicates_are_not_fused() -> None:
    """Two returned values are two tensors, even when they compute the same thing."""
    graph_module = _two_users(RELU, returned=2)

    result = FuseDuplicateUsersPass()(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RELU) == 2


def test_one_returned_one_interior_still_fuses() -> None:
    """Sharing a value between an output and an interior consumer is ordinary."""
    graph_module = _two_users(RELU, returned=1)

    result = FuseDuplicateUsersPass()(graph_module)

    assert result.modified
    assert _count(result.graph_module, RELU) == 1


def test_caller_that_repairs_uniqueness_may_opt_in() -> None:
    """Arm fuses and then restores uniqueness, so it asks for the aliasing."""
    graph_module = _two_users(RELU, returned=2)

    result = FuseDuplicateUsersPass(may_alias_outputs=True)(graph_module)

    assert result.modified
    assert _count(result.graph_module, RELU) == 1
    returned = result.graph_module.graph.output_node().args[0]
    assert returned[0] is returned[1]


def test_excluded_targets_still_win() -> None:
    graph_module = _two_users(RELU, returned=0)

    result = FuseDuplicateUsersPass(frozenset({RELU}))(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RELU) == 2


def test_unfusable_returned_siblings_do_not_block_interior_fusion() -> None:
    """The refusal is per merge, not per target.

    Two returned adds with different operands are never candidates for each
    other, so they say nothing about a pair of interior duplicates that happen
    to share their target.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)

    def _add(operand: float) -> torch.fx.Node:
        node = graph.call_function(ADD, args=(x, operand))
        node.meta["val"] = torch.empty(SHAPE)
        return node

    consumed = []
    for interior in (_add(1.0), _add(1.0)):
        node = graph.call_function(RELU, args=(interior,))
        node.meta["val"] = torch.empty(SHAPE)
        consumed.append(node)
    graph.output((*consumed, _add(7.0), _add(9.0)))
    graph.lint()
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass()(graph_module)

    assert result.modified
    assert _count(result.graph_module, ADD) == 3


def test_duplicate_users_are_not_fused_across_mutation() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)
    first = graph.call_function(ATEN_NEG, args=(x,))
    first.meta["val"] = torch.empty(SHAPE)
    mutation = graph.call_function(torch.ops.aten.add_.Tensor, args=(x, 1))
    mutation.meta["val"] = torch.empty(SHAPE)
    second = graph.call_function(ATEN_NEG, args=(x,))
    second.meta["val"] = torch.empty(SHAPE)
    graph.output((first, mutation, second))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass()(graph_module)

    assert not result.modified
    assert _count(result.graph_module, ATEN_NEG) == 2


def test_duplicate_values_are_not_fused_when_one_is_mutated_downstream() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(SHAPE)
    first = graph.call_function(ATEN_CLONE, args=(x,))
    first.meta["val"] = torch.ones(SHAPE)
    second = graph.call_function(ATEN_CLONE, args=(x,))
    second.meta["val"] = torch.ones(SHAPE)
    mutation = graph.call_function(torch.ops.aten.add_.Tensor, args=(second, 1))
    mutation.meta["val"] = torch.full(SHAPE, 2.0)
    consumed = graph.call_function(ADD, args=(mutation, 1))
    consumed.meta["val"] = torch.full(SHAPE, 3.0)
    graph.output((first, consumed))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass()(graph_module)

    assert not result.modified
    assert _count(result.graph_module, ATEN_CLONE) == 2
    first_output, second_output = result.graph_module(torch.ones(SHAPE))
    torch.testing.assert_close(first_output, torch.ones(SHAPE))
    torch.testing.assert_close(second_output, torch.full(SHAPE, 3.0))


def test_getitem_is_not_an_effect_barrier() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(SHAPE)
    first = graph.call_function(ATEN_NEG, args=(x,))
    first.meta["val"] = -torch.ones(SHAPE)
    getitem = graph.call_function(operator.getitem, args=((x, x), 0))
    getitem.meta["val"] = torch.ones(SHAPE)
    consumed = graph.call_function(ADD, args=(getitem, 1))
    consumed.meta["val"] = torch.full(SHAPE, 2.0)
    second = graph.call_function(ATEN_NEG, args=(x,))
    second.meta["val"] = -torch.ones(SHAPE)
    graph.output((first, second, consumed))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass(may_alias_outputs=True)(graph_module)

    assert result.modified
    assert _count(result.graph_module, ATEN_NEG) == 1


def test_effect_classification_is_linear_in_graph_size() -> None:
    class CountingFuseDuplicateUsersPass(FuseDuplicateUsersPass):
        effect_checks = 0

        @classmethod
        def _has_observable_effect(cls, node):
            cls.effect_checks += 1
            return super()._has_observable_effect(node)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(SHAPE)
    duplicates = []
    for _ in range(20):
        duplicate = graph.call_function(RELU, args=(x,))
        duplicate.meta["val"] = torch.ones(SHAPE)
        duplicates.append(duplicate)
    stacked = graph.call_function(STACK, args=(duplicates, 0))
    stacked.meta["val"] = torch.ones((20, *SHAPE))
    graph.output(stacked)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    graph_node_count = len(list(graph.nodes))

    result = CountingFuseDuplicateUsersPass()(graph_module)

    assert result.modified
    assert CountingFuseDuplicateUsersPass.effect_checks == graph_node_count


def test_newly_returned_representative_does_not_absorb_second_output() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)
    duplicates = []
    for _ in range(3):
        duplicate = graph.call_function(RELU, args=(x,))
        duplicate.meta["val"] = torch.empty(SHAPE)
        duplicates.append(duplicate)
    interior = graph.call_function(ADD, args=(duplicates[0], x))
    interior.meta["val"] = torch.empty(SHAPE)
    graph.output((interior, duplicates[1], duplicates[2]))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass()(graph_module)

    assert result.modified
    assert _count(result.graph_module, RELU) == 2
    returned = result.graph_module.graph.output_node().args[0]
    assert returned[1] is not returned[2]


def test_aliasing_output_paths_are_not_collapsed() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)
    aliases = []
    for _ in range(2):
        duplicate = graph.call_function(RELU, args=(x,))
        duplicate.meta["val"] = torch.empty(SHAPE)
        alias = graph.call_function(ATEN_VIEW, args=(duplicate, SHAPE))
        alias.meta["val"] = torch.empty(SHAPE)
        aliases.append(alias)
    graph.output(tuple(aliases))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass(may_alias_outputs=True)(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RELU) == 2


def test_direct_outputs_with_aliasing_output_paths_are_not_collapsed() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)
    duplicates = []
    aliases = []
    for _ in range(2):
        duplicate = graph.call_function(RELU, args=(x,))
        duplicate.meta["val"] = torch.empty(SHAPE)
        alias = graph.call_function(ATEN_VIEW, args=(duplicate, SHAPE))
        alias.meta["val"] = torch.empty(SHAPE)
        duplicates.append(duplicate)
        aliases.append(alias)
    graph.output((duplicates[0], aliases[0], duplicates[1], aliases[1]))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = FuseDuplicateUsersPass(may_alias_outputs=True)(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RELU) == 2
