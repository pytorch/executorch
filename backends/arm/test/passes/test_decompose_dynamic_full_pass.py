# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import ArmPassManager, DecomposeDynamicFullPass
from executorch.backends.arm.test import common
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class _DynamicFull(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, 3.5)


class _DynamicIntegerFull(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, 3)


class _DynamicFullLike(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, 3.5)


class _StaticFull(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        return torch.full((2, 3), 3.5)


def _export_dynamic_full() -> torch.export.ExportedProgram:
    return torch.export.export(
        _DynamicFull().eval(),
        (torch.randn(2, 3, 4),),
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch", min=1, max=8),
                2: torch.export.Dim("height", min=1, max=16),
            }
        },
    )


def test_decompose_dynamic_full_to_scalar_full_and_repeat() -> None:
    exported_program = _export_dynamic_full()

    result = DecomposeDynamicFullPass()(exported_program.graph_module)
    assert result is not None

    full_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.full.default
    ]
    repeat_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.repeat.default
    ]

    assert len(full_nodes) == 1
    assert len(repeat_nodes) == 1
    assert full_nodes[0].args[0] == (1,)

    repeat_sizes = repeat_nodes[0].args[1]
    assert isinstance(repeat_sizes, list)
    assert len(repeat_sizes) == 3
    assert repeat_sizes[1] == 3
    assert getattr(repeat_sizes[0], "target", None) == torch.ops.aten.sym_size.int
    assert getattr(repeat_sizes[2], "target", None) == torch.ops.aten.sym_size.int

    result.graph_module.graph.lint()


def test_annotation_pipeline_converts_dynamic_integer_full_to_int32() -> None:
    exported_program = torch.export.export(
        _DynamicIntegerFull().eval(),
        (torch.randn(2, 3, 4),),
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch", min=1, max=8),
                2: torch.export.Dim("height", min=1, max=16),
            }
        },
    )

    graph_module = ArmPassManager(
        common.get_tosa_compile_spec("TOSA-1.0+INT")
    ).transform_for_annotation_pipeline(exported_program.graph_module)

    full_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.full.default
    ]
    repeat_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.repeat.default
    ]

    assert len(full_nodes) == 1
    assert len(repeat_nodes) == 1
    assert full_nodes[0].args[0] == (1,)
    assert full_nodes[0].kwargs["dtype"] == torch.int32
    assert full_nodes[0].meta["val"].dtype == torch.int32


def test_backend_pipeline_decomposes_dynamic_full_like() -> None:
    exported_program = torch.export.export(
        _DynamicFullLike().eval(),
        (torch.randn(2, 3, 4),),
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch", min=1, max=8),
                2: torch.export.Dim("height", min=1, max=16),
            }
        },
    )
    edge_program = to_edge(exported_program, compile_config=EdgeCompileConfig())
    graph_module = ArmPassManager(
        common.get_tosa_compile_spec("TOSA-1.0+FP")
    ).transform_to_backend_pipeline(
        edge_program.exported_program(),
        edge_program.exported_program().graph_module,
    )

    full_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == exir_ops.edge.aten.full.default
    ]
    full_like_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.full_like.default
    ]
    tile_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.TILE.default
    ]

    assert not full_nodes
    assert not full_like_nodes
    assert len(tile_nodes) == 1
    assert tile_nodes[0].args[1][1] == 3


def test_decompose_dynamic_full_leaves_static_full_unchanged() -> None:
    exported_program = torch.export.export(_StaticFull().eval(), ())

    result = DecomposeDynamicFullPass()(exported_program.graph_module)
    assert result is not None

    full_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.full.default
    ]
    repeat_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.repeat.default
    ]

    assert len(full_nodes) == 1
    assert full_nodes[0].args[0] == [2, 3]
    assert not repeat_nodes
