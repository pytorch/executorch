# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from executorch.backends.arm._passes import (
    FuseConsecutiveConcatsPass,
    RewriteCatSlicePass,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata
from torch.fx import GraphModule, Node


_CAT = exir_ops.edge.aten.cat.default
_SLICE = exir_ops.edge.aten.slice_copy.Tensor


def _concat_slice_graph(
    input_channels: list[int],
    slice_ranges: list[tuple[int, int]],
    *,
    inserted_channels: int | None = None,
    input_qparams: dict[int, str] | None = None,
    include_step: bool = True,
) -> tuple[GraphModule, list[Node], Node]:
    builder = GraphBuilder()
    inputs = [
        builder.placeholder(f"input_{index}", torch.randn(1, channels, 8, 8))
        for index, channels in enumerate(input_channels)
    ]
    base_concat = builder.call_operator(_CAT, (inputs, 1))
    slices = [
        builder.call_operator(
            _SLICE,
            (
                (base_concat, 1, start, end, 1)
                if include_step
                else (base_concat, 1, start, end)
            ),
        )
        for start, end in slice_ranges
    ]
    if inserted_channels is None:
        output = slices[0]
    else:
        inserted = builder.placeholder(
            "inserted", torch.randn(1, inserted_channels, 8, 8)
        )
        output = builder.call_operator(
            _CAT,
            ([slices[0], inserted, *slices[1:]], 1),
            meta=NodeMetadata({"input_qparams": input_qparams or {}}),
        )
    builder.output([output])
    return builder.get_graph_module(), [input.node for input in inputs], output.node


def _run(graph_module: GraphModule) -> bool:
    slice_result = RewriteCatSlicePass().call(graph_module)
    concat_result = FuseConsecutiveConcatsPass().call(graph_module)
    return slice_result.modified or concat_result.modified


def _call_nodes(graph_module: GraphModule) -> list[Node]:
    return [node for node in graph_module.graph.nodes if node.op == "call_function"]


def test_rewrite_cat_slice_inserts_on_boundary() -> None:
    graph_module, inputs, output = _concat_slice_graph(
        [3, 4, 2],
        [(0, 7), (7, 9)],
        inserted_channels=1,
        input_qparams={0: "left", 1: "inserted", 2: "right"},
    )

    assert _run(graph_module)
    call_nodes = _call_nodes(graph_module)
    assert [node.target for node in call_nodes] == [_CAT, _CAT]
    output_inputs = cast(list[Node], output.args[0])
    nested_concat = output_inputs[0]
    assert list(cast(list[Node], nested_concat.args[0])) == [
        inputs[0],
        inputs[1],
    ]
    assert output_inputs == [
        nested_concat,
        next(node for node in graph_module.graph.nodes if node.name == "inserted"),
        inputs[2],
    ]
    assert output.meta["input_qparams"] == {
        0: "left",
        1: "inserted",
        2: "right",
    }


def test_rewrite_cat_slice_rejects_multi_user_op_growth() -> None:
    graph_module, inputs, output = _concat_slice_graph(
        [3, 4, 2], [(0, 5), (5, 9)], inserted_channels=1
    )

    assert not _run(graph_module)
    assert output in _call_nodes(graph_module)


def test_rewrite_cat_slice_rewrites_partial_coverage() -> None:
    graph_module, inputs, output = _concat_slice_graph(
        [3, 4], [(0, 3)], inserted_channels=1
    )

    assert _run(graph_module)
    assert [node.target for node in _call_nodes(graph_module)] == [_CAT]
    assert list(cast(list[Node], output.args[0])) == [
        inputs[0],
        next(node for node in graph_module.graph.nodes if node.name == "inserted"),
    ]


def test_rewrite_cat_slice_rejects_noncontiguous_multi_user_op_growth() -> None:
    graph_module, inputs, output = _concat_slice_graph(
        [3, 4, 2], [(0, 5), (6, 9)], inserted_channels=1
    )

    assert not _run(graph_module)
    assert output in _call_nodes(graph_module)


def test_rewrite_cat_slice_rejects_empty_slice() -> None:
    graph_module, _, output = _concat_slice_graph([0, 0], [(0, 0)])

    assert not _run(graph_module)
    assert output in _call_nodes(graph_module)


def test_rewrite_cat_slice_rejects_shared_concat() -> None:
    builder = GraphBuilder()
    a = builder.placeholder("a", torch.randn(1, 3, 8, 8))
    b = builder.placeholder("b", torch.randn(1, 4, 8, 8))
    source = builder.call_operator(_CAT, ([a, b], 1))
    sliced = builder.call_operator(_SLICE, (source, 1, 0, 3, 1))
    builder.output([source, sliced])
    graph_module = builder.get_graph_module()

    assert not RewriteCatSlicePass().call(graph_module).modified


def test_rewrite_cat_slice_rejects_three_piece_single_user_rewrite() -> None:
    graph_module, _, output = _concat_slice_graph([3, 3, 3], [(1, 8)])

    assert not RewriteCatSlicePass().call(graph_module).modified
    assert output in _call_nodes(graph_module)


def test_rewrite_cat_slice_rewrites_quantized_full_coverage() -> None:
    builder = GraphBuilder()
    a = builder.placeholder("a", torch.randn(1, 3, 8, 8))
    b = builder.placeholder("b", torch.randn(1, 4, 8, 8))
    source = builder.call_operator(_CAT, ([a, b], 1))
    source.node.meta["input_qparams"] = {0: "cat_qparams"}
    sliced = builder.call_operator(_SLICE, (source, 1, 0, 7, 1))
    builder.output([sliced])
    graph_module = builder.get_graph_module()

    assert RewriteCatSlicePass().call(graph_module).modified
    assert [node.target for node in _call_nodes(graph_module)] == [_CAT]


def test_rewrite_cat_slice_accepts_implicit_step() -> None:
    graph_module, inputs, output = _concat_slice_graph(
        [3, 4],
        [(0, 3), (3, 9223372036854775807)],
        inserted_channels=1,
        include_step=False,
    )

    assert _run(graph_module)
    assert [node.target for node in _call_nodes(graph_module)] == [_CAT]
    assert list(cast(list[Node], output.args[0])) == [
        inputs[0],
        next(node for node in graph_module.graph.nodes if node.name == "inserted"),
        inputs[1],
    ]


def test_rewrite_cat_slice_preserves_source_input_qparams() -> None:
    builder = GraphBuilder()
    a = builder.placeholder("a", torch.randn(1, 3, 8, 8))
    b = builder.placeholder("b", torch.randn(1, 4, 8, 8))
    source = builder.call_operator(_CAT, ([a, b], 1))
    source.node.meta["input_qparams"] = {0: "cat_qparams"}
    sliced = builder.call_operator(_SLICE, (source, 1, 0, 5, 1))
    sliced.node.meta["input_qparams"] = {0: "source_qparams"}
    sliced.node.meta["output_qparams"] = {0: "slice_qparams"}
    builder.output([sliced])
    graph_module = builder.get_graph_module()

    assert RewriteCatSlicePass().call(graph_module).modified

    slice_nodes = [node for node in _call_nodes(graph_module) if node.target == _SLICE]
    fused_cat = next(node for node in _call_nodes(graph_module) if node.target == _CAT)
    assert [node.args for node in slice_nodes] == [(b.node, 1, 0, 2, 1)]
    assert [node.meta["input_qparams"] for node in slice_nodes] == [{0: "cat_qparams"}]
    assert all(
        node.meta["output_qparams"] == {0: "slice_qparams"} for node in slice_nodes
    )
    assert fused_cat.meta["input_qparams"] == {0: "cat_qparams"}
    assert fused_cat.meta["output_qparams"] == {0: "slice_qparams"}
