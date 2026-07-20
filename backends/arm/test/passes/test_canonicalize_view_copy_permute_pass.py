# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import cast

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes import CanonicalizeViewCopyPermutePass
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils import _pytree as pytree


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr, compiler_min=min, compiler_max=max
    )
    return symint


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


def test_canonicalize_direct_permute_chain() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3, 4, 5)
    x = builder.placeholder("x", x_data)
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(x, [0, 2, 3, 1]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(p1, [0, 3, 1, 2]),
    )
    builder.output([p2])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 0
    )
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_pixel_shuffle_view_permute_chain() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(1, 2, 2, 8)
    x = builder.placeholder("x", x_data)
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(x, [0, 3, 1, 2]),
    )
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [1, 2, 2, 2, 2, 2]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [0, 1, 4, 2, 5, 3]),
    )
    v2 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p2, [1, 2, 4, 4]),
    )
    p3 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v2, [0, 2, 3, 1]),
    )
    builder.output([p3])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 2
    assert _compute_nodes(result.graph_module) == [
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view_copy.default,
    ]
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_direct_output_axis_permute() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 12, 5)
    x = builder.placeholder("x", x_data)
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(x, [2, 3, 4, 5]),
    )
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [2, 0, 1, 3]),
    )
    v2 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [4, 6, 5]),
    )
    builder.output([v2])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1

    compute_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert [node.target for node in compute_nodes] == [
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
    ]
    assert compute_nodes[0].args[1] == [6, 4, 5]
    assert compute_nodes[1].args[1] == [1, 0, 2]
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_reordered_view_with_grouped_permute() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3, 4)
    x = builder.placeholder("x", x_data)
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(x, [3, 2, 4]),
    )
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [2, 0, 1]),
    )
    v2 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [4, 6]),
    )
    builder.output([v2])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1

    compute_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert [node.target for node in compute_nodes] == [
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view_copy.default,
    ]
    assert compute_nodes[0].args[1] == [2, 0, 1]
    assert compute_nodes[1].args[1] == [4, 6]
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_moves_view_before_permute() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3, 4)
    x = builder.placeholder("x", x_data)
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(x, [2, 0, 1]),
    )
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [4, 6]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [1, 0]),
    )
    builder.output([p2])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 0
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1

    compute_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert [node.target for node in compute_nodes] == [
        exir_ops.edge.aten.view_copy.default,
    ]
    assert compute_nodes[0].args[1] == [6, 4]
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_moves_permute_before_view() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(1, 2, 10, 10)
    x = builder.placeholder("x", x_data)
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(x, [1, 2, 5, 2, 5, 2]),
    )
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [3, 5, 0, 1, 2, 4]),
    )
    v2 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [4, 2, 5, 5]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v2, [0, 2, 3, 1]),
    )
    builder.output([p2])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 2

    compute_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert [node.target for node in compute_nodes] == [
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view_copy.default,
    ]
    assert compute_nodes[0].args[1] == [1, 2, 5, 2, 5, 2]
    assert compute_nodes[1].args[1] == [3, 5, 0, 2, 4, 1]
    assert compute_nodes[2].args[1] == [4, 5, 5, 2]
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_follows_interleaved_chain_users() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(4, 2, 4)
    y_data = torch.randn(2, 3)
    x = builder.placeholder("x", x_data)
    y = builder.placeholder("y", y_data)
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(x, [1, 0, 2]),
    )
    unrelated = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(y, [3, 2]),
    )
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [1, 2, 4, 4]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [0, 2, 1, 3]),
    )
    builder.output([p2, unrelated])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 0
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 2

    compute_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert compute_nodes[0].target == exir_ops.edge.aten.view_copy.default
    assert compute_nodes[0].args[1] == [1, 4, 2, 4]
    _validate_numerics(gm_before, result.graph_module, (x_data, y_data))


def test_canonicalize_reordered_view_rejects_separating_permute() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3, 4)
    x = builder.placeholder("x", x_data)
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(x, [3, 2, 4]),
    )
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [0, 2, 1]),
    )
    builder.output([p1])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert not result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_symbolic_pixel_shuffle_view_permute_chain() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=2)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True) as mode:
        builder = GraphBuilder(fake_tensor_mode=mode)
        x = builder.placeholder("x", torch.empty(size=(batch, 2, 2, 8)))
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 3, 1, 2]),
        )
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(p1, [batch, 2, 2, 2, 2, 2]),
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(v1, [0, 1, 4, 2, 5, 3]),
        )
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(p2, [batch, 2, 4, 4]),
        )
        p3 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(v2, [0, 2, 3, 1]),
        )
        builder.output([p3])
        original = builder.get_graph_module()

        pass_instance = CanonicalizeViewCopyPermutePass()
        result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 2

    compute_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    assert compute_nodes[0].args[1] == [batch, 2, 2, 2, 2, 2]
    assert compute_nodes[1].args[1] == [0, 1, 4, 2, 5, 3]
    assert compute_nodes[2].args[1] == [batch, 4, 4, 2]


def test_canonicalize_symbolic_singleton_permute_stays_permute() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=2)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True) as mode:
        builder = GraphBuilder(fake_tensor_mode=mode)
        x = builder.placeholder("x", torch.empty(size=(batch, 1, 4)))
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [1, 0, 2]),
        )
        builder.output([p1])
        original = builder.get_graph_module()

        pass_instance = CanonicalizeViewCopyPermutePass()
        result = cast(PassResult, pass_instance.call(original))

    assert not result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 1
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 0


def test_canonicalize_view_permute_swap_uses_factor_order() -> None:
    x = torch.empty((2, 3, 4, 5, 6, 7, 8, 9), device="meta")
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    input_node.meta["val"] = x
    view = graph.call_function(
        exir_ops.edge.aten.view_copy.default,
        args=(input_node, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )
    view.meta["val"] = torch.empty((1, 2, 3, 4, 5, 6, 7, 8, 9), device="meta")
    permute = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        args=(view, [0, 8, 7, 6, 5, 4, 3, 2, 1]),
    )
    permute.meta["val"] = torch.empty((1, 9, 8, 7, 6, 5, 4, 3, 2), device="meta")

    assert CanonicalizeViewCopyPermutePass()._view_permute_swap(view, permute) == (
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 9, 8, 7, 6, 5, 4, 3, 2],
    )


def test_canonicalize_view_permute_swap_rejects_reordered_split_axis() -> None:
    x = torch.empty((4,), device="meta")
    graph = torch.fx.Graph()
    input_node = graph.placeholder("x")
    input_node.meta["val"] = x
    view = graph.call_function(
        exir_ops.edge.aten.view_copy.default,
        args=(input_node, [2, 2]),
    )
    view.meta["val"] = torch.empty((2, 2), device="meta")
    permute = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        args=(view, [1, 0]),
    )
    permute.meta["val"] = torch.empty((2, 2), device="meta")

    assert CanonicalizeViewCopyPermutePass()._view_permute_swap(view, permute) is None


def test_canonicalize_does_not_cross_multi_user_chain_node() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3, 4)
    x = builder.placeholder("x", x_data)
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(x, [0, 2, 1]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(p1, [0, 2, 1]),
    )
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p1, [2, 12]),
    )
    builder.output([p2, v1])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert not result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 2
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_unsupported_start_view_does_not_block_suffix() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3)
    x = builder.placeholder("x", x_data)
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(x, [3, 2]),
    )
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(v1, [1, 0]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(p1, [1, 0]),
    )
    builder.output([p2])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 0
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1
    _validate_numerics(gm_before, result.graph_module, (x_data,))


def test_canonicalize_unsupported_end_view_does_not_block_prefix() -> None:
    builder = GraphBuilder()
    x_data = torch.randn(2, 3)
    x = builder.placeholder("x", x_data)
    p1 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(x, [1, 0]),
    )
    p2 = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(p1, [1, 0]),
    )
    v1 = builder.call_operator(
        op=exir_ops.edge.aten.view_copy.default,
        args=(p2, [3, 2]),
    )
    builder.output([v1])
    original = builder.get_graph_module()
    gm_before = copy.deepcopy(original)

    pass_instance = CanonicalizeViewCopyPermutePass()
    result = cast(PassResult, pass_instance.call(original))

    assert result.modified
    assert (
        _count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default) == 0
    )
    assert _count_node(result.graph_module, exir_ops.edge.aten.view_copy.default) == 1
    _validate_numerics(gm_before, result.graph_module, (x_data,))
