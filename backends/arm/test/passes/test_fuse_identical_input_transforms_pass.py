# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import cast, Sequence

import pytest
import torch
from executorch.backends.arm._passes.fuse_identical_input_transforms_pass import (
    FuseIdenticalInputTransformsPass,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult, ProxyValue
from torch.utils import _pytree as pytree


_ADD = exir_ops.edge.aten.add.Tensor
_CAT = exir_ops.edge.aten.cat.default
_VIEW = exir_ops.edge.aten.view_copy.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default


def _count_node(
    graph_module: torch.fx.GraphModule, target: torch.fx.node.Target
) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def _compute_nodes(graph_module: torch.fx.GraphModule) -> list[torch.fx.node.Target]:
    return [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]


def _validate_numerics(
    original: torch.fx.GraphModule,
    modified: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    original.eval()
    modified.eval()
    with torch.no_grad():
        original_output = original(*inputs)
        modified_output = modified(*inputs)

    flat_original, _ = pytree.tree_flatten(original_output)
    flat_modified, _ = pytree.tree_flatten(modified_output)
    for original_tensor, modified_tensor in zip(flat_original, flat_modified):
        torch.testing.assert_close(original_tensor, modified_tensor)


def _apply_pass(
    graph_module: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, PassResult]:
    original = copy.deepcopy(graph_module)
    result = cast(PassResult, FuseIdenticalInputTransformsPass().call(graph_module))
    return original, result


def _binary_transform_graph(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_transforms: Sequence[tuple[torch.fx.node.Target, Sequence[int]]],
    y_transforms: Sequence[tuple[torch.fx.node.Target, Sequence[int]]],
) -> torch.fx.GraphModule:
    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)

    x_node = _apply_transforms(builder, x, x_transforms)
    y_node = _apply_transforms(builder, y, y_transforms)
    add = builder.call_operator(op=_ADD, args=(x_node, y_node))
    builder.output([add])
    return builder.get_graph_module()


def _concat_transform_graph(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_transform: tuple[torch.fx.node.Target, Sequence[int]],
    y_transform: tuple[torch.fx.node.Target, Sequence[int]],
    dim: int,
) -> torch.fx.GraphModule:
    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)

    x_node = _apply_transforms(builder, x, [x_transform])
    y_node = _apply_transforms(builder, y, [y_transform])
    cat = builder.call_operator(op=_CAT, args=([x_node, y_node], dim))
    builder.output([cat])
    return builder.get_graph_module()


def _apply_transforms(
    builder: GraphBuilder,
    node: ProxyValue,
    transforms: Sequence[tuple[torch.fx.node.Target, Sequence[int]]],
) -> ProxyValue:
    transformed = node
    for op, arg in transforms:
        transformed = builder.call_operator(op=op, args=(transformed, list(arg)))
    return transformed


def test_fuse_identical_input_transforms_sinks_view() -> None:
    x_data = torch.randn(6)
    y_data = torch.randn(6)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [2, 3])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_permute() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 2, 1])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_cat_input_views() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    builder.output([cat])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    call_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    cat_node = next(node for node in call_nodes if node.target == _CAT)
    view = next(node for node in call_nodes if node.target == _VIEW)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_CAT, _VIEW]
    assert [input_node.name for input_node in cat_node.args[0]] == ["x", "y"]
    assert cat_node.args[1] == 2
    assert cat_node.meta["val"].shape == torch.Size((2, 3, 8))
    assert view.args == (cat_node, [6, 8])
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_cat_mixed_transforms() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _concat_transform_graph(
        x_data,
        y_data,
        (_VIEW, [2, 3, 4]),
        (_PERMUTE, [0, 1, 2]),
        0,
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_VIEW, _PERMUTE, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_cat_different_permutes() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 4, 3)
    graph_module = _concat_transform_graph(
        x_data,
        y_data,
        (_PERMUTE, [0, 1, 2]),
        (_PERMUTE, [0, 2, 1]),
        0,
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 2
    assert _compute_nodes(result.graph_module) == [_PERMUTE, _PERMUTE, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_shared_binary_input_transform() -> (
    None
):
    x_data = torch.randn(6)
    y_data = torch.randn(6)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [2, 3]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [2, 3]))
    add = builder.call_operator(op=_ADD, args=(x_view, y_view))
    builder.output([add, x_view])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_shared_cat_input_transform() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    builder.output([cat, x_view])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_different_views() -> None:
    x_data = torch.randn(6)
    y_data = torch.randn(6)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [1, 2, 3])],
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_different_permutes() -> None:
    x_data = torch.randn(2, 1, 3)
    y_data = torch.randn(2, 1, 3)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 1, 2])],
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 2
    assert _compute_nodes(result.graph_module) == [_PERMUTE, _PERMUTE, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_view_permute_chain() -> None:
    x_data = torch.randn(6, 4)
    y_data = torch.randn(6, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3, 4]), (_PERMUTE, [0, 2, 1])],
        [(_VIEW, [2, 3, 4]), (_PERMUTE, [0, 2, 1])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW, _PERMUTE]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_sinks_permute_view_chain() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(2, 3, 4)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1]), (_VIEW, [2, 2, 6])],
        [(_PERMUTE, [0, 2, 1]), (_VIEW, [2, 2, 6])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE, _VIEW]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


@pytest.mark.parametrize(
    "x_shape, y_shape, view_shape",
    [
        ((1, 6), (6,), (2, 3)),
        ((1, 1, 6), (1, 6), (3, 2)),
    ],
)
def test_fuse_identical_input_transforms_sinks_views_through_rank_broadcasts(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    view_shape: tuple[int, ...],
) -> None:
    x_data = torch.randn(x_shape)
    y_data = torch.randn(y_shape)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, view_shape)],
        [(_VIEW, view_shape)],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _VIEW) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _VIEW]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_fuse_identical_input_transforms_rejects_views_through_expanding_broadcast() -> (
    None
):
    x_data = torch.randn(1, 6)
    y_data = torch.randn(6, 1)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_VIEW, [2, 3])],
        [(_VIEW, [2, 3])],
    )

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _ADD]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ((1, 2, 3), (4, 2, 3)),
        ((4, 1, 3), (1, 2, 3)),
        ((4, 2, 1), (1, 1, 3)),
    ],
)
def test_fuse_identical_input_transforms_sinks_permutes_through_broadcasts(
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
) -> None:
    x_data = torch.randn(x_shape)
    y_data = torch.randn(y_shape)
    graph_module = _binary_transform_graph(
        x_data,
        y_data,
        [(_PERMUTE, [0, 2, 1])],
        [(_PERMUTE, [0, 2, 1])],
    )

    original, result = _apply_pass(graph_module)

    assert result.modified
    assert _count_node(result.graph_module, _PERMUTE) == 1
    assert _compute_nodes(result.graph_module) == [_ADD, _PERMUTE]
    _validate_numerics(original, result.graph_module, (x_data, y_data))


def test_cat_rejects_different_view_inputs() -> None:
    x_data = torch.randn(2, 3, 4)
    y_data = torch.randn(1, 6, 4)

    builder = GraphBuilder()
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    x_view = builder.call_operator(op=_VIEW, args=(x, [6, 4]))
    y_view = builder.call_operator(op=_VIEW, args=(y, [6, 4]))
    cat = builder.call_operator(op=_CAT, args=([x_view, y_view], 1))
    builder.output([cat])
    graph_module = builder.get_graph_module()

    original, result = _apply_pass(graph_module)

    assert not result.modified
    assert _count_node(result.graph_module, _VIEW) == 2
    assert _compute_nodes(result.graph_module) == [_VIEW, _VIEW, _CAT]
    _validate_numerics(original, result.graph_module, (x_data, y_data))
