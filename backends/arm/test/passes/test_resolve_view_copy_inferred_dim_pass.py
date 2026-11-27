# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import pytest

import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes.resolve_view_copy_inferred_dim_pass import (
    ResolveViewCopyInferredDimPass,
)
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import NodeMetadata, ProxyValue
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _make_symint(
    shape_env: ShapeEnv, symbol: str, hint: int, min: int = 1, max: int = 64
) -> torch.SymInt:
    symint = shape_env.create_symintnode(sympy.Symbol(symbol), hint=hint)
    assert isinstance(symint, torch.SymInt)
    shape_env.constrain_symbol_range(
        symint.node.expr, compiler_min=min, compiler_max=max
    )
    return symint


def _single_view_graph(
    op,
    input_value: torch.Tensor,
    shape: list[Any] | tuple[Any, ...],
    output_value: torch.Tensor,
    fake_tensor_mode: FakeTensorMode | None = None,
) -> tuple[torch.fx.GraphModule, ProxyValue, ProxyValue]:
    builder = GraphBuilder(fake_tensor_mode=fake_tensor_mode)
    x = builder.placeholder("x", input_value)
    view = builder.call_operator(
        op,
        (x, shape),
        meta=NodeMetadata({"val": output_value}),
    )
    builder.output([view])
    return builder.get_graph_module(), x, view


def test_resolve_view_copy_inferred_dim_replaces_static_dim() -> None:
    graph_module, _, view = _single_view_graph(
        torch.ops.aten.view.default,
        torch.empty(2, 12),
        [2, -1, 4],
        torch.empty(2, 3, 4),
    )

    result = ResolveViewCopyInferredDimPass().call(graph_module)

    assert result.modified
    assert view.node.args[1] == [2, 3, 4]
    assert torch.ops.aten.sym_size.int not in {
        node.target for node in graph_module.graph.nodes
    }
    graph_module.graph.lint()


def test_resolve_view_copy_inferred_dim_materializes_dynamic_dim() -> None:
    shape_env = ShapeEnv()
    batch = _make_symint(shape_env, "batch", hint=2)
    width = _make_symint(shape_env, "width", hint=3)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True) as mode:
        graph_module, x, view = _single_view_graph(
            exir_ops.edge.aten.view_copy.default,
            torch.empty(size=(batch, width, 4)),
            [batch, -1, 4],
            torch.empty(size=(batch, width, 4)),
            mode,
        )

        result = ResolveViewCopyInferredDimPass().call(graph_module)

    assert result.modified
    resolved_shape = cast(list[Any], view.node.args[1])
    assert resolved_shape[0] == batch
    assert isinstance(resolved_shape[1], torch.fx.Node)
    assert resolved_shape[1].target == torch.ops.aten.sym_size.int
    assert resolved_shape[1].args == (x.node, 1)
    assert resolved_shape[1].meta["val"].node.expr == width.node.expr
    assert resolved_shape[2] == 4
    assert -1 not in resolved_shape
    graph_module.graph.lint()


def test_resolve_view_copy_inferred_dim_preserves_tuple_shape() -> None:
    graph_module, _, view = _single_view_graph(
        exir_ops.edge.aten.view_copy.default,
        torch.empty(2, 12),
        (2, -1, 4),
        torch.empty(2, 3, 4),
    )

    result = ResolveViewCopyInferredDimPass().call(graph_module)

    assert result.modified
    assert view.node.args[1] == (2, 3, 4)
    graph_module.graph.lint()


def test_resolve_view_copy_inferred_dim_ignores_explicit_shape() -> None:
    graph_module, _, view = _single_view_graph(
        exir_ops.edge.aten.view_copy.default,
        torch.empty(2, 3, 4),
        [2, 3, 4],
        torch.empty(2, 3, 4),
    )

    result = ResolveViewCopyInferredDimPass().call(graph_module)

    assert not result.modified
    assert view.node.args[1] == [2, 3, 4]
    graph_module.graph.lint()


def test_resolve_view_copy_inferred_dim_rejects_multiple_inferred_dims() -> None:
    graph_module, _, view = _single_view_graph(
        exir_ops.edge.aten.view_copy.default,
        torch.empty(24),
        [2, -1, 4],
        torch.empty(2, 3, 4),
    )
    view.node.args = (view.node.args[0], [2, -1, -1])

    with pytest.raises(ValueError, match="more than one inferred dimension"):
        ResolveViewCopyInferredDimPass().call(graph_module)
