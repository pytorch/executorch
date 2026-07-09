# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast

import torch
from executorch.backends.arm._passes.remove_permutes_around_elementwise_tosa_ops import (
    RemovePermutesAroundElementwiseTosaOps,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

TOSA_INT_SPEC = TosaSpecification.create_from_string("TOSA-1.0+INT")
TOSA_FP_SPEC = TosaSpecification.create_from_string("TOSA-1.0+FP")
PERMUTE_TARGET = exir_ops.edge.aten.permute_copy.default
RESCALE_TARGET = exir_ops.backend.tosa.RESCALE.default
MUL_TARGET = exir_ops.edge.aten.mul.Tensor
ADD_TARGET = exir_ops.edge.aten.add.Tensor
ERF_TARGET = exir_ops.edge.aten.erf.default


def _fake_exported_program() -> ExportedProgram:
    return cast(
        ExportedProgram,
        SimpleNamespace(
            graph_signature=SimpleNamespace(
                inputs_to_buffers={},
                inputs_to_lifted_tensor_constants={},
                inputs_to_parameters={},
            )
        ),
    )


def _count_nodes(graph_module: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target
    )


def test_remove_permutes_around_rescale_tosa_INT() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(1, 3, 4, 5)

    permute_in = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(x, [0, 2, 3, 1]),
    )
    rescale = graph.create_node(
        "call_function",
        RESCALE_TARGET,
        args=(permute_in, torch.int8, [1.0], 0, 0),
    )
    permute_out = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(rescale, [0, 3, 1, 2]),
    )
    graph.output(permute_out)

    graph_module = torch.fx.GraphModule({}, graph)

    with TosaLoweringContext(TOSA_INT_SPEC):
        result = RemovePermutesAroundElementwiseTosaOps(_fake_exported_program()).call(
            graph_module
        )

    assert result.modified
    assert _count_nodes(result.graph_module, PERMUTE_TARGET) == 0
    assert _count_nodes(result.graph_module, RESCALE_TARGET) == 1


def test_remove_permutes_around_gelu_with_folded_scalar_constants_tosa_FP() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(1, 2, 3, 4)

    scalar_constants = []
    for i in range(3):
        const = graph.placeholder(f"c_scalar_{i}")
        const.meta["val"] = torch.randn(1, 1, 1, 1)
        scalar_constants.append(const)

    permute_in = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(x, [0, 2, 3, 1]),
    )
    permute_in.meta["val"] = torch.randn(1, 3, 4, 2)
    mul_0 = graph.create_node(
        "call_function",
        MUL_TARGET,
        args=(permute_in, scalar_constants[0]),
    )
    mul_0.meta["val"] = torch.randn(1, 3, 4, 2)
    erf = graph.create_node("call_function", ERF_TARGET, args=(mul_0,))
    erf.meta["val"] = torch.randn(1, 3, 4, 2)
    add = graph.create_node(
        "call_function",
        ADD_TARGET,
        args=(erf, scalar_constants[1]),
    )
    add.meta["val"] = torch.randn(1, 3, 4, 2)
    mul_1 = graph.create_node(
        "call_function",
        MUL_TARGET,
        args=(add, scalar_constants[2]),
    )
    mul_1.meta["val"] = torch.randn(1, 3, 4, 2)
    mul_2 = graph.create_node(
        "call_function",
        MUL_TARGET,
        args=(permute_in, mul_1),
    )
    mul_2.meta["val"] = torch.randn(1, 3, 4, 2)
    permute_out = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(mul_2, [0, 3, 1, 2]),
    )
    permute_out.meta["val"] = torch.randn(1, 2, 3, 4)
    graph.output(permute_out)

    graph_module = torch.fx.GraphModule({}, graph)

    with TosaLoweringContext(TOSA_FP_SPEC):
        result = RemovePermutesAroundElementwiseTosaOps(_fake_exported_program()).call(
            graph_module
        )

    assert result.modified
    assert _count_nodes(result.graph_module, PERMUTE_TARGET) == 3
    assert _count_nodes(result.graph_module, ERF_TARGET) == 1


def test_remove_permutes_skips_stale_shared_boundary_subgraph_tosa_FP() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(1, 16, 16, 8)

    channel_const = graph.placeholder("p_layer_norm_weight")
    channel_const.meta["val"] = torch.randn(1, 1, 1, 8)

    permute_in = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(x, [0, 3, 1, 2]),
    )
    permute_in.meta["val"] = torch.randn(1, 8, 16, 16)
    first_mul = graph.create_node(
        "call_function",
        MUL_TARGET,
        args=(permute_in, permute_in),
    )
    first_mul.meta["val"] = torch.randn(1, 8, 16, 16)
    shared_permute = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(first_mul, [0, 2, 3, 1]),
    )
    shared_permute.meta["val"] = torch.randn(1, 16, 16, 8)
    second_mul = graph.create_node(
        "call_function",
        MUL_TARGET,
        args=(shared_permute, channel_const),
    )
    second_mul.meta["val"] = torch.randn(1, 16, 16, 8)
    permute_out = graph.create_node(
        "call_function",
        PERMUTE_TARGET,
        args=(second_mul, [0, 3, 1, 2]),
    )
    permute_out.meta["val"] = torch.randn(1, 8, 16, 16)
    graph.output(permute_out)

    graph_module = torch.fx.GraphModule({}, graph)

    with TosaLoweringContext(TOSA_FP_SPEC):
        result = RemovePermutesAroundElementwiseTosaOps(_fake_exported_program()).call(
            graph_module
        )

    assert result.modified
    assert _count_nodes(result.graph_module, PERMUTE_TARGET) == 1
    assert second_mul.args[1] is channel_const
