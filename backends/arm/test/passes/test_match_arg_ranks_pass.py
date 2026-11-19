# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import pytest
import sympy  # type: ignore[import-untyped]
import torch
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.tosa_lowering_utils import create_const
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def test_match_arg_ranks_handles_tosa_binary_ops() -> None:
    fake_mode = FakeTensorMode()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = fake_mode.from_tensor(torch.empty((1, 270, 480, 3, 2, 2)))
    y = graph.placeholder("y")
    y.meta["val"] = fake_mode.from_tensor(torch.empty((1, 1, 1, 1)))
    add = graph.call_function(exir_ops.backend.tosa.ADD.default, args=(x, y))
    add.meta["val"] = fake_mode.from_tensor(torch.empty((1, 270, 480, 3, 2, 2)))
    graph.output(add)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        result = MatchArgRanksPass(cast(ExportedProgram, None)).call(graph_module)
    call_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    view = next(
        node
        for node in call_nodes
        if node.target == exir_ops.edge.aten.view_copy.default
    )
    add = next(
        node for node in call_nodes if node.target == exir_ops.backend.tosa.ADD.default
    )

    assert result.modified
    assert view.all_input_nodes[0].target == "y"
    assert view.args[1] == [1, 1, 1, 1, 1, 1]
    assert add.all_input_nodes[0].target == "x"
    assert add.all_input_nodes[1] is view


@pytest.mark.parametrize("share_shift_with_data", (False, True))
def test_match_arg_ranks_preserves_mul_shift(share_shift_with_data):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    with FakeTensorMode():
        source.meta["val"] = torch.empty(2, 3, dtype=torch.int8)
    shift = create_const(graph, 0, torch.int8, source)
    scalar = (
        shift if share_shift_with_data else create_const(graph, 1, torch.int8, source)
    )
    mul = graph.call_function(
        exir_ops.backend.tosa.MUL.default, (scalar, source, shift)
    )
    mul.meta["val"] = source.meta["val"]
    graph.output(mul)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = MatchArgRanksPass(cast(ExportedProgram, None))(
            torch.fx.GraphModule({}, graph)
        )

    assert result is not None
    result.graph_module.graph.lint()
    mul = result.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.backend.tosa.MUL.default
    )[0]
    assert mul.args[0].meta["val"].shape == (1, 1)
    assert mul.args[1].meta["val"].shape == (2, 3)
    shift = mul.args[2]
    assert shift.target == exir_ops.backend.tosa.CONST.default
    assert shift.args == ([0], [1], torch.int8)
    assert shift.meta["val"].shape == (1,)


@pytest.mark.parametrize("use_tosa", [False, True])
def test_match_arg_ranks_preserves_consumer_provenance(use_tosa: bool) -> None:
    """Broadcast views retain their consumer's provenance, not their input's."""
    fake_mode = FakeTensorMode()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = fake_mode.from_tensor(torch.empty((2, 3)))
    y = graph.placeholder("y")
    y.meta["val"] = fake_mode.from_tensor(torch.empty((3,)))
    y.meta["nn_module_stack"] = {"producer": ("producer", torch.nn.Identity)}

    target = (
        exir_ops.backend.tosa.ADD.default if use_tosa else exir_ops.edge.aten.add.Tensor
    )
    binary = graph.call_function(target, args=(x, y))
    expected_module_stack = {"consumer": ("consumer", torch.nn.ReLU)}
    expected_source_stack = [("consumer", torch.nn.ReLU)]
    binary.meta.update(
        {
            "val": fake_mode.from_tensor(torch.empty((2, 3))),
            "nn_module_stack": expected_module_stack,
            "source_fn_stack": expected_source_stack,
            "stack_trace": "original consumer traceback",
            "debug_handle": 123,
            # The generated view must not inherit these operator qparams.
            "input_qparams": {0: "consumer input qparams"},
            "output_qparams": {0: "consumer output qparams"},
        }
    )
    graph.output(binary)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        rank_pass = MatchArgRanksPass(cast(ExportedProgram, None))
        result = rank_pass.call(graph_module)
        second_result = rank_pass.call(result.graph_module)

    assert result.modified
    assert not second_result.modified, "The pass must be idempotent."
    call_nodes = [
        node for node in result.graph_module.graph.nodes if node.op == "call_function"
    ]
    views = [
        node
        for node in call_nodes
        if node.target == exir_ops.edge.aten.view_copy.default
    ]
    assert len(views) == 1
    view = views[0]
    consumer = next(node for node in call_nodes if node.target == target)

    assert view.args[0].target == "y"
    assert view.args[1] == [1, 3]
    assert consumer.args[1] is view
    assert view.meta.get("nn_module_stack") == expected_module_stack
    assert view.meta["nn_module_stack"] != y.meta["nn_module_stack"]
    assert view.meta.get("source_fn_stack") == expected_source_stack
    assert "original consumer traceback" in view.meta.get("stack_trace", "")
    assert view.meta.get("debug_handle") == 123
    assert view.meta.get("input_qparams") == {}
    assert view.meta.get("output_qparams") == {}
    # Retracing must recalculate the view's shape, not retain ADD's shape.
    assert tuple(view.meta["val"].shape) == (1, 3)
    assert tuple(consumer.meta["val"].shape) == (2, 3)


def test_match_arg_ranks_materializes_dynamic_view_shape() -> None:
    shape_env = ShapeEnv()
    width = shape_env.create_symintnode(sympy.Symbol("width"), hint=14)
    assert isinstance(width, torch.SymInt)
    shape_env.constrain_symbol_range(width.node.expr, compiler_min=1, compiler_max=16)

    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(size=(1, width * 16, 1, 1))
        y = graph.placeholder("y")
        y.meta["val"] = torch.empty(size=(width * 16,))
        shift = graph.placeholder("shift")
        shift.meta["val"] = torch.zeros(
            size=[
                1,
            ],
            dtype=torch.int8,
        )
        mul = graph.call_function(exir_ops.backend.tosa.MUL.default, args=(x, y, shift))
        mul.meta["val"] = torch.empty(size=(1, width * 16, 1, 1))
        graph.output(mul)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        with TosaLoweringContext(
            TosaSpecification.create_from_string("TOSA-1.1+FP+shape")
        ):
            result = MatchArgRanksPass(cast(ExportedProgram, None)).call(graph_module)

    view = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.view_copy.default
    )
    materialized_shape = view.args[1]

    assert result.modified
    assert materialized_shape[:3] == [1, 1, 1]
    assert isinstance(materialized_shape[3], torch.fx.Node)
    assert not any(isinstance(dim, torch.SymInt) for dim in materialized_shape)
