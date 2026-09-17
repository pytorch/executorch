# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, Callable, cast

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes.resolve_view_copy_inferred_dim_pass import (
    ResolveViewCopyInferredDimPass,
)
from executorch.backends.arm._passes.symbolic_to_tosa_shape_pass import (
    SymbolicToTosaShapesPass,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, PassResult, ProxyValue
from torch._export.utils import _get_shape_env_from_gm
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Dim, export
from torch.fx.experimental.symbolic_shapes import ShapeEnv


class _RecordingMaterializer:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any], NodeMetadata]] = []

    def materialize_shape_op(self, target, args, kwargs, meta):
        self.calls.append((target, args, kwargs, meta))
        return target


def _run_symbolic_shape_pass(graph_module: torch.fx.GraphModule) -> PassResult:
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.1+FP+shape")):
        result = SymbolicToTosaShapesPass()(graph_module)
    assert result is not None
    return result


def _targets(graph_module: torch.fx.GraphModule) -> list[Any]:
    return [node.target for node in graph_module.graph.nodes]


def _single_node_with_target(
    graph_module: torch.fx.GraphModule,
    target: Any,
) -> torch.fx.Node:
    return next(node for node in graph_module.graph.nodes if node.target == target)


def _nodes_with_target(
    graph_module: torch.fx.GraphModule,
    target: Any,
) -> list[torch.fx.Node]:
    return [node for node in graph_module.graph.nodes if node.target == target]


def _symbolic_binary_op(
    builder: GraphBuilder,
    op: Callable[..., Any],
    args: tuple[Any, Any],
    value: int,
) -> ProxyValue:
    return builder.call_operator(op, args, meta=NodeMetadata({"val": value}))


def _build_view_graph(
    shape_builder: Callable[[GraphBuilder, ProxyValue], list[Any]],
    output_shape: tuple[int, ...],
) -> torch.fx.GraphModule:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    shape = shape_builder(builder, x)
    view = builder.call_operator(
        torch.ops.aten.view.default,
        (x, shape),
        meta=NodeMetadata({"val": torch.empty(output_shape)}),
    )
    builder.output([view])
    return builder.get_graph_module()


def _sym_size(
    builder: GraphBuilder,
    x: ProxyValue,
    dim: int,
    value: int,
) -> ProxyValue:
    return builder.call_operator(
        torch.ops.aten.sym_size.int,
        (x, dim),
        meta=NodeMetadata({"val": value}),
    )


def _shape_proxy(builder: GraphBuilder, value: int) -> ProxyValue:
    return builder.call_operator(
        exir_ops.backend.tosa.CONST_SHAPE.default,
        ([value],),
        meta=NodeMetadata(
            {
                "val": [value],
                TosaSpecialDtype.meta_key(): TosaSpecialDtype.SHAPE,
            }
        ),
    )


def test_symbolic_to_tosa_shapes_rewrites_sym_size_to_dim() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    sym_size = _sym_size(builder, x, 1, 18)
    builder.output([sym_size])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    dim_node = _single_node_with_target(graph_module, exir_ops.backend.tosa.DIM.default)

    assert getattr(dim_node.args[0], "target", None) == "x"
    assert dim_node.kwargs == {"axis": 1}
    assert torch.ops.aten.sym_size.int not in {
        node.target for node in graph_module.graph.nodes
    }
    assert result.modified is True
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_marks_dim_as_shape_dtype() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    sym_size = _sym_size(builder, x, 0, 2)
    builder.output([sym_size])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    dim_node = _single_node_with_target(
        result.graph_module, exir_ops.backend.tosa.DIM.default
    )

    assert dim_node.meta[TosaSpecialDtype.meta_key()] == TosaSpecialDtype.SHAPE
    result.graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_leaves_non_sym_size_ops_unchanged() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    add = builder.call_operator(
        torch.ops.aten.add.Tensor,
        (x, x),
        meta=NodeMetadata({"val": torch.empty(2, 18)}),
    )
    builder.output([add])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module

    _single_node_with_target(graph_module, torch.ops.aten.add.Tensor)
    assert exir_ops.backend.tosa.DIM.default not in {
        node.target for node in graph_module.graph.nodes
    }


def test_symbolic_to_tosa_shapes_rewrites_sym_size_and_symbolic_list() -> None:
    def shape(builder: GraphBuilder, x: ProxyValue) -> list[Any]:
        return [_sym_size(builder, x, 0, 2), 18]

    result = _run_symbolic_shape_pass(_build_view_graph(shape, (2, 18)))
    graph_module = result.graph_module
    targets = _targets(graph_module)
    view_node = _single_node_with_target(graph_module, torch.ops.aten.view.default)

    dim_node = _single_node_with_target(graph_module, exir_ops.backend.tosa.DIM.default)
    const_node = _single_node_with_target(
        graph_module, exir_ops.backend.tosa.CONST_SHAPE.default
    )

    assert dim_node.kwargs == {"axis": 0}
    assert const_node.args == ([18],)
    assert exir_ops.backend.tosa.CONCAT_SHAPE.default in targets
    assert torch.ops.aten.sym_size.int not in targets
    assert (
        getattr(view_node.args[1], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_materializes_raw_symint_list_arg() -> None:
    shape_env = ShapeEnv()
    width = shape_env.create_symintnode(sympy.Symbol("width"), hint=14)
    assert isinstance(width, torch.SymInt)
    shape_env.constrain_symbol_range(width.node.expr, compiler_min=1, compiler_max=16)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(size=(1, width * 16, 1, 1))
        reshape = graph.call_function(
            exir_ops.backend.tosa.RESHAPE.default,
            args=(x, [1, width * 16, 1, 1]),
        )
        reshape.meta["val"] = torch.empty(size=(1, width * 16, 1, 1))
        graph.output(reshape)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
        ):
            result = SymbolicToTosaShapesPass()(graph_module)

    assert result is not None
    graph_module = result.graph_module
    reshape_node = _single_node_with_target(
        graph_module, exir_ops.backend.tosa.RESHAPE.default
    )
    targets = _targets(graph_module)

    assert result.modified
    assert torch.ops.aten.sym_size.int not in targets
    assert exir_ops.backend.tosa.DIM.default in targets
    assert (
        getattr(reshape_node.args[1], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    assert not any(isinstance(arg, torch.SymInt) for arg in reshape_node.args)
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_rewrites_symbolic_expression() -> None:
    def shape(builder: GraphBuilder, x: ProxyValue) -> list[Any]:
        dim_0 = _sym_size(builder, x, 0, 2)
        dim_1 = _sym_size(builder, x, 1, 18)
        product = _symbolic_binary_op(builder, operator.mul, (dim_0, dim_1), 36)
        return [product]

    result = _run_symbolic_shape_pass(_build_view_graph(shape, (36,)))
    graph_module = result.graph_module
    targets = _targets(graph_module)
    view_node = _single_node_with_target(graph_module, torch.ops.aten.view.default)

    assert targets.count(exir_ops.backend.tosa.DIM.default) == 2
    assert exir_ops.backend.tosa.MUL_SHAPE.default in targets
    assert torch.ops.aten.sym_size.int not in targets
    assert (
        getattr(view_node.args[1], "target", None)
        == exir_ops.backend.tosa.MUL_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_resolves_view_copy_inferred_dim() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    sym_size = _sym_size(builder, x, 0, 2)
    view = builder.call_operator(
        exir_ops.edge.aten.view_copy.default,
        (x, [sym_size, 3, -1]),
        meta=NodeMetadata({"val": torch.empty(2, 3, 6)}),
    )
    builder.output([view])

    resolve_result = ResolveViewCopyInferredDimPass()(builder.get_graph_module())
    assert resolve_result is not None
    assert resolve_result.modified

    result = _run_symbolic_shape_pass(resolve_result.graph_module)
    graph_module = result.graph_module
    view_node = _single_node_with_target(
        graph_module, exir_ops.edge.aten.view_copy.default
    )
    const_shape_args = [
        node.args[0]
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CONST_SHAPE.default
    ]
    concat_node = _single_node_with_target(
        graph_module, exir_ops.backend.tosa.CONCAT_SHAPE.default
    )

    assert [-1] not in const_shape_args
    assert [3] in const_shape_args
    assert [6] in const_shape_args
    assert concat_node.meta["val"] == [2, 3, 6]
    assert (
        getattr(view_node.args[1], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    graph_module.graph.lint()


def test_resolve_view_copy_inferred_dim_materializes_dynamic_dim() -> None:
    class ViewModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.view_copy.default(x, [x.shape[0], -1])

    edge_model = to_edge(
        export(
            ViewModule(),
            (torch.randn(2, 3, 4),),
            dynamic_shapes={
                "x": {
                    0: Dim("batch", min=1, max=5),
                    1: Dim("height", min=2, max=6),
                    2: Dim("width", min=2, max=8),
                }
            },
        )
    )
    graph_module = edge_model.exported_program().graph_module
    shape_env = _get_shape_env_from_gm(graph_module)

    resolve_result = ResolveViewCopyInferredDimPass()(graph_module)
    assert resolve_result is not None
    assert resolve_result.modified

    view_node = _single_node_with_target(
        resolve_result.graph_module, exir_ops.edge.aten.view_copy.default
    )
    resolved_shape = cast(list[Any], view_node.args[1])
    assert -1 not in resolved_shape
    assert getattr(resolved_shape[1], "target", None) == operator.mul

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
    ):
        result = SymbolicToTosaShapesPass()(resolve_result.graph_module)
    assert result is not None

    graph_module = result.graph_module
    targets = _targets(graph_module)
    view_node = _single_node_with_target(
        graph_module, exir_ops.edge.aten.view_copy.default
    )

    assert torch.ops.aten.sym_size.int not in targets
    assert targets.count(exir_ops.backend.tosa.DIM.default) == 3
    assert exir_ops.backend.tosa.MUL_SHAPE.default in targets
    assert (
        getattr(view_node.args[1], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_keeps_view_copy_shape_without_inferred_dim() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(36))
    sym_size = _sym_size(builder, x, 0, 36)
    view = builder.call_operator(
        exir_ops.edge.aten.view_copy.default,
        (x, [sym_size, 1]),
        meta=NodeMetadata({"val": torch.empty(36, 1)}),
    )
    builder.output([view])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    view_node = _single_node_with_target(
        graph_module, exir_ops.edge.aten.view_copy.default
    )
    const_shape_nodes = _nodes_with_target(
        graph_module, exir_ops.backend.tosa.CONST_SHAPE.default
    )

    assert [node.args for node in const_shape_nodes] == [([1],)]
    assert (
        getattr(view_node.args[1], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_runs_for_shape_marked_list_without_sym_size() -> None:
    graph = torch.fx.Graph()
    shape = graph.call_function(exir_ops.backend.tosa.CONST_SHAPE.default, ([2],))
    shape.meta["val"] = [2]
    shape.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
    empty = graph.call_function(
        torch.ops.aten.empty.memory_format,
        ([shape],),
        {"device": torch.device("cpu"), "pin_memory": False},
    )
    empty.meta["val"] = torch.empty(2)
    graph.output(empty)

    result = _run_symbolic_shape_pass(torch.fx.GraphModule(torch.nn.Module(), graph))
    graph_module = result.graph_module
    empty_node = _single_node_with_target(
        graph_module,
        torch.ops.aten.empty.memory_format,
    )

    assert getattr(empty_node.args[0], "target", None) == (
        exir_ops.backend.tosa.CONST_SHAPE.default
    )
    graph_module.graph.lint()


@pytest.mark.parametrize(
    "symbolic_op,tosa_op",
    [
        (operator.add, exir_ops.backend.tosa.ADD_SHAPE.default),
        (operator.sub, exir_ops.backend.tosa.SUB_SHAPE.default),
        (operator.mul, exir_ops.backend.tosa.MUL_SHAPE.default),
        (operator.mod, exir_ops.backend.tosa.MOD_SHAPE.default),
        (operator.floordiv, exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default),
    ],
)
def test_symbolic_to_tosa_shapes_maps_symbolic_arithmetic_ops(
    symbolic_op,
    tosa_op,
) -> None:
    shape_pass = SymbolicToTosaShapesPass()
    materializer = _RecordingMaterializer()
    shape_pass.materializer = materializer
    builder = GraphBuilder()
    lhs = _shape_proxy(builder, 4)
    rhs = _shape_proxy(builder, 2)
    meta = NodeMetadata({"val": [4]})

    result = shape_pass.call_sym(symbolic_op, (lhs, rhs), meta)

    assert result == tosa_op
    assert materializer.calls == [(tosa_op, (lhs, rhs), {}, meta)]


def test_symbolic_to_tosa_shapes_rejects_unsupported_symbolic_op() -> None:
    shape_pass = SymbolicToTosaShapesPass()
    shape_pass.materializer = _RecordingMaterializer()
    builder = GraphBuilder()
    lhs = _shape_proxy(builder, 4)
    rhs = _shape_proxy(builder, 2)

    with pytest.raises(NotImplementedError, match="Symbolic op target"):
        shape_pass.call_sym(operator.truediv, (lhs, rhs), NodeMetadata({"val": [4]}))


def test_symbolic_to_tosa_shapes_rewrites_add_expression() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    dim_0 = _sym_size(builder, x, 0, 2)
    dim_1 = _sym_size(builder, x, 1, 18)
    add = _symbolic_binary_op(builder, operator.add, (dim_0, dim_1), 20)
    empty = builder.call_operator(
        torch.ops.aten.empty.memory_format,
        ([add],),
        {"device": torch.device("cpu"), "pin_memory": False},
        NodeMetadata({"val": torch.empty(20)}),
    )
    builder.output([empty])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    targets = _targets(graph_module)
    empty_node = _single_node_with_target(
        graph_module,
        torch.ops.aten.empty.memory_format,
    )

    assert targets.count(exir_ops.backend.tosa.DIM.default) == 2
    assert exir_ops.backend.tosa.ADD_SHAPE.default in targets
    assert torch.ops.aten.sym_size.int not in targets
    assert (
        getattr(empty_node.args[0], "target", None)
        == exir_ops.backend.tosa.ADD_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_materializes_raw_symint_mod_expression() -> None:
    shape_env = ShapeEnv()
    height = shape_env.create_symintnode(sympy.Symbol("height"), hint=14)
    assert isinstance(height, torch.SymInt)
    shape_env.constrain_symbol_range(height.node.expr, compiler_min=1, compiler_max=16)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(size=(1, height, 1, 1))
        dynamic_dim = (((height - ((height - 1) % 2)) - 1) // 2) + 1
        empty = graph.call_function(
            torch.ops.aten.empty.memory_format,
            args=([1, dynamic_dim, 1, 1],),
            kwargs={"device": torch.device("cpu"), "pin_memory": False},
        )
        empty.meta["val"] = torch.empty(size=(1, dynamic_dim, 1, 1))
        graph.output(empty)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env
        ):
            result = SymbolicToTosaShapesPass()(graph_module)

    assert result is not None
    graph_module = result.graph_module
    empty_node = _single_node_with_target(
        graph_module, torch.ops.aten.empty.memory_format
    )
    targets = _targets(graph_module)

    assert exir_ops.backend.tosa.MOD_SHAPE.default in targets
    assert exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default in targets
    assert (
        getattr(empty_node.args[0], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_concats_static_scalar_proxy() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    dim_1 = _sym_size(builder, x, 1, 18)
    sub = builder.call_operator(
        operator.sub,
        (2, 1),
        meta=NodeMetadata({"val": 1}),
    )
    empty = builder.call_operator(
        torch.ops.aten.empty.memory_format,
        ([sub, dim_1],),
        {"device": torch.device("cpu"), "pin_memory": False},
        NodeMetadata({"val": torch.empty(1, 18)}),
    )
    builder.output([empty])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    concat_node = _single_node_with_target(
        graph_module, exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    empty_node = _single_node_with_target(
        graph_module, torch.ops.aten.empty.memory_format
    )

    assert concat_node.meta["val"] == [1, 18]
    assert (
        getattr(empty_node.args[0], "target", None)
        == exir_ops.backend.tosa.CONCAT_SHAPE.default
    )
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_concats_scalar_symbolic_expression() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(2, 18))
    dim_0 = _sym_size(builder, x, 0, 2)
    dim_1 = _sym_size(builder, x, 1, 18)
    sub = _symbolic_binary_op(builder, operator.sub, (dim_0, 1), 1)
    empty = builder.call_operator(
        torch.ops.aten.empty.memory_format,
        ([sub, dim_1],),
        {"device": torch.device("cpu"), "pin_memory": False},
        NodeMetadata({"val": torch.empty(1, 18)}),
    )
    builder.output([empty])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    sub_node = _single_node_with_target(
        graph_module, exir_ops.backend.tosa.SUB_SHAPE.default
    )
    concat_node = _single_node_with_target(
        graph_module, exir_ops.backend.tosa.CONCAT_SHAPE.default
    )

    assert sub_node.meta["val"] == [1]
    assert concat_node.meta["val"] == [1, 18]
    graph_module.graph.lint()


def test_symbolic_to_tosa_shapes_rewrites_nested_symbolic_expression() -> None:
    builder = GraphBuilder()
    x = builder.placeholder("x", torch.randn(8, 3, 2))
    dim_0 = _sym_size(builder, x, 0, 8)
    dim_1 = _sym_size(builder, x, 1, 3)
    sub = _symbolic_binary_op(builder, operator.sub, (dim_0, 1), 7)
    add = _symbolic_binary_op(builder, operator.add, (sub, dim_1), 10)
    floordiv = _symbolic_binary_op(builder, operator.floordiv, (add, 2), 5)
    empty = builder.call_operator(
        torch.ops.aten.empty.memory_format,
        ([floordiv],),
        {"device": torch.device("cpu"), "pin_memory": False},
        NodeMetadata({"val": torch.empty(5)}),
    )
    builder.output([empty])

    result = _run_symbolic_shape_pass(builder.get_graph_module())
    graph_module = result.graph_module
    targets = _targets(graph_module)
    empty_node = _single_node_with_target(
        graph_module,
        torch.ops.aten.empty.memory_format,
    )

    assert targets.count(exir_ops.backend.tosa.DIM.default) == 2
    assert exir_ops.backend.tosa.SUB_SHAPE.default in targets
    assert exir_ops.backend.tosa.ADD_SHAPE.default in targets
    assert exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default in targets
    assert torch.ops.aten.sym_size.int not in targets
    assert (
        getattr(empty_node.args[0], "target", None)
        == exir_ops.backend.tosa.DIV_FLOOR_SHAPE.default
    )
    graph_module.graph.lint()
