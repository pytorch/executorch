# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import cast

import torch
from executorch.backends.arm._passes import FuseConsecutiveConcatsPass
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata
from torch.fx import GraphModule, Node


_CAT = exir_ops.edge.aten.cat.default


def _nested_concat_graph(
    *,
    outer_dim: int = 1,
    outer_dim_as_kwarg: bool = False,
    include_inner_output: bool = False,
    input_qparams: dict[int, str] | None = None,
) -> tuple[GraphModule, Node, Node, Node, Node, Node]:
    builder = GraphBuilder()
    a = builder.placeholder("a", torch.randn(1, 2))
    b = builder.placeholder("b", torch.randn(1, 2))
    c = builder.placeholder(
        "c", torch.randn(1, 4) if outer_dim == 0 else torch.randn(1, 2)
    )
    inner = builder.call_operator(_CAT, ([a, b], 1))
    outer = builder.call_operator(
        _CAT,
        ([inner, c],) if outer_dim_as_kwarg else ([inner, c], outer_dim),
        kwargs={"dim": outer_dim} if outer_dim_as_kwarg else {},
        meta=NodeMetadata({"input_qparams": input_qparams or {}}),
    )
    builder.output([inner, outer] if include_inner_output else [outer])
    return (
        builder.get_graph_module(),
        a.node,
        b.node,
        c.node,
        inner.node,
        outer.node,
    )


def _cat_nodes(graph_module: GraphModule) -> list[Node]:
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == _CAT
    ]


def _validate_numerics(
    original: GraphModule, modified: GraphModule, inputs: tuple[torch.Tensor, ...]
) -> None:
    torch.testing.assert_close(original(*inputs), modified(*inputs))


def test_fuse_consecutive_concats_flattens_single_use_concat() -> None:
    graph_module, a, b, c, inner, outer = _nested_concat_graph(
        input_qparams={0: "qparams"}
    )
    inner.meta["input_qparams"] = {0: "qparams"}
    inner.meta["output_qparams"] = {0: "qparams"}
    outer.meta["output_qparams"] = {0: "qparams"}

    original = copy.deepcopy(graph_module)
    result = FuseConsecutiveConcatsPass().call(graph_module)

    assert result.modified
    assert _cat_nodes(graph_module) == [outer]
    assert list(cast(list[Node], outer.args[0])) == [a, b, c]
    assert outer.meta["input_qparams"] == {0: "qparams"}
    assert outer.meta["output_qparams"] == {0: "qparams"}
    _validate_numerics(
        original,
        graph_module,
        (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2)),
    )


def test_fuse_consecutive_concats_preserves_keyword_dim() -> None:
    graph_module, a, b, c, _, outer = _nested_concat_graph(outer_dim_as_kwarg=True)

    original = copy.deepcopy(graph_module)
    result = FuseConsecutiveConcatsPass().call(graph_module)

    assert result.modified
    assert _cat_nodes(graph_module) == [outer]
    assert list(cast(list[Node], outer.args[0])) == [a, b, c]
    assert outer.kwargs == {"dim": 1}
    _validate_numerics(
        original,
        graph_module,
        (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2)),
    )


def test_fuse_consecutive_concats_rejects_mismatched_qparams() -> None:
    graph_module, _, _, _, inner, outer = _nested_concat_graph()
    inner.meta["input_qparams"] = {0: "qparams"}
    inner.meta["output_qparams"] = {0: "qparams"}
    outer.meta["input_qparams"] = {0: "qparams"}
    outer.meta["output_qparams"] = {0: "different_qparams"}

    result = FuseConsecutiveConcatsPass().call(graph_module)

    assert not result.modified
    assert _cat_nodes(graph_module) == [inner, outer]


def test_fuse_consecutive_concats_rejects_multi_use_or_different_dim() -> None:
    graph_module, _, _, _, inner, outer = _nested_concat_graph(
        outer_dim=0,
        include_inner_output=True,
    )

    result = FuseConsecutiveConcatsPass().call(graph_module)

    assert not result.modified
    assert _cat_nodes(graph_module) == [inner, outer]
