# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm._passes import (
    CanonicalizeViewCopyPermutePass,
    EnsureUniqueOutputNodesPass,
    ExirToTosaPass,
    FuseDuplicateUsersPass,
    InsertRescalePass,
    RemoveNoopPass,
    RewriteSlicePass,
)
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export, ExportedProgram
from torch.fx import Graph, GraphModule, Node


def _call(
    graph: Graph,
    target,
    args,
    val,
    kwargs=None,
) -> Node:
    node = graph.call_function(target, args, kwargs or {})
    node.meta["val"] = val
    return node


def _count_target(graph_module: GraphModule, target) -> int:
    return sum(node.target == target for node in graph_module.graph.nodes)


def _run_remove_noop(graph_module: GraphModule) -> GraphModule:
    spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
    with TosaLoweringContext(spec):
        result = RemoveNoopPass()(graph_module)
    assert result is not None
    return result.graph_module


def _tosa_data_layout_graph(
    target,
    shape_values: tuple[list[int], ...],
    output_shape: tuple[int, ...],
) -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2, 3)
    shape_nodes = [
        _call(
            graph,
            exir_ops.backend.tosa.CONST_SHAPE.default,
            (values,),
            values,
        )
        for values in shape_values
    ]
    kwargs = {"value": 0.0} if target == exir_ops.backend.tosa.PAD.default else {}
    result = _call(
        graph,
        target,
        (x, *shape_nodes),
        torch.ones(output_shape),
        kwargs,
    )
    graph.output((result,))
    return GraphModule(torch.nn.Module(), graph)


@pytest.mark.parametrize(
    "target",
    [
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.concatenate.default,
    ],
)
def test_remove_single_input_concat(target):
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2, 3)
    concat = _call(graph, target, ([x], 0), torch.ones(2, 3))
    graph.output((concat,))

    result = _run_remove_noop(GraphModule(torch.nn.Module(), graph))

    assert _count_target(result, target) == 0
    assert result.graph.output_node().args[0][0].op == "placeholder"


def test_keep_multi_input_concat():
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2, 3)
    concat = _call(
        graph,
        exir_ops.edge.aten.cat.default,
        ([x, x], 0),
        torch.ones(4, 3),
    )
    graph.output((concat,))

    result = _run_remove_noop(GraphModule(torch.nn.Module(), graph))

    assert _count_target(result, exir_ops.edge.aten.cat.default) == 1


def test_remove_full_slice_and_unused_shape_constants():
    graph_module = _tosa_data_layout_graph(
        exir_ops.backend.tosa.SLICE.default,
        ([0, 0], [2, 3]),
        (2, 3),
    )

    result = _run_remove_noop(graph_module)

    assert _count_target(result, exir_ops.backend.tosa.SLICE.default) == 0
    assert _count_target(result, exir_ops.backend.tosa.CONST_SHAPE.default) == 0
    assert result.graph.output_node().args[0][0].op == "placeholder"


def test_keep_partial_slice():
    graph_module = _tosa_data_layout_graph(
        exir_ops.backend.tosa.SLICE.default,
        ([1, 0], [1, 3]),
        (1, 3),
    )

    result = _run_remove_noop(graph_module)

    assert _count_target(result, exir_ops.backend.tosa.SLICE.default) == 1


def test_keep_slice_with_nonconstant_shape_operands():
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2, 3)
    starts = graph.placeholder("starts")
    starts.meta["val"] = [0, 0]
    sizes = graph.placeholder("sizes")
    sizes.meta["val"] = [2, 3]
    sliced = _call(
        graph,
        exir_ops.backend.tosa.SLICE.default,
        (x, starts, sizes),
        torch.ones(2, 3),
    )
    graph.output((sliced,))

    result = _run_remove_noop(GraphModule(torch.nn.Module(), graph))

    assert _count_target(result, exir_ops.backend.tosa.SLICE.default) == 1


def test_keep_full_slice_with_symbolic_shape():
    class FullSlice(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.slice.Tensor(x, 0, 0, torch.iinfo(torch.int64).max, 1)

    exported_program = export(
        FullSlice(),
        (torch.ones(4, 3),),
        dynamic_shapes={
            "x": {0: torch.export.Dim("batch", min=2, max=8)},
        },
        strict=True,
    )
    edge_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
    with TosaLoweringContext(spec):
        rewritten = RewriteSlicePass()(edge_program.graph_module).graph_module

    slice_nodes = [
        node
        for node in rewritten.graph.nodes
        if node.target == exir_ops.backend.tosa.SLICE.default
    ]
    assert len(slice_nodes) == 1
    assert isinstance(slice_nodes[0].args[2][0], torch.SymInt)

    result = _run_remove_noop(rewritten)

    assert _count_target(result, exir_ops.backend.tosa.SLICE.default) == 1


def test_remove_zero_pad_and_unused_shape_constant():
    graph_module = _tosa_data_layout_graph(
        exir_ops.backend.tosa.PAD.default,
        ([0, 0, 0, 0],),
        (2, 3),
    )

    result = _run_remove_noop(graph_module)

    assert _count_target(result, exir_ops.backend.tosa.PAD.default) == 0
    assert _count_target(result, exir_ops.backend.tosa.CONST_SHAPE.default) == 0


def test_keep_nonzero_pad():
    graph_module = _tosa_data_layout_graph(
        exir_ops.backend.tosa.PAD.default,
        ([0, 0, 1, 0],),
        (2, 4),
    )

    result = _run_remove_noop(graph_module)

    assert _count_target(result, exir_ops.backend.tosa.PAD.default) == 1


@pytest.mark.parametrize("padding", [[], [0], [0, 0]])
def test_reject_malformed_zero_pad(padding: list[int]):
    graph_module = _tosa_data_layout_graph(
        exir_ops.backend.tosa.PAD.default,
        (padding,),
        (2, 3),
    )

    with pytest.raises(TosaValueError, match="Padding length"):
        _run_remove_noop(graph_module)


def test_noop_removal_exposes_transpose_cleanup():
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2, 3)
    first_permute = _call(
        graph,
        exir_ops.edge.aten.permute_copy.default,
        (x, [1, 0]),
        torch.ones(3, 2),
    )
    padding = _call(
        graph,
        exir_ops.backend.tosa.CONST_SHAPE.default,
        ([0, 0, 0, 0],),
        [0, 0, 0, 0],
    )
    padded = _call(
        graph,
        exir_ops.backend.tosa.PAD.default,
        (first_permute, padding),
        torch.ones(3, 2),
        {"value": 0.0},
    )
    second_permute = _call(
        graph,
        exir_ops.edge.aten.permute_copy.default,
        (padded, [1, 0]),
        torch.ones(2, 3),
    )
    graph.output((second_permute,))

    graph_module = _run_remove_noop(GraphModule(torch.nn.Module(), graph))
    result = CanonicalizeViewCopyPermutePass()(graph_module).graph_module

    assert _count_target(result, exir_ops.backend.tosa.PAD.default) == 0
    assert _count_target(result, exir_ops.edge.aten.permute_copy.default) == 0


def test_noop_removal_exposes_duplicate_user_cleanup():
    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2, 3)
    y = graph.placeholder("y")
    y.meta["val"] = torch.ones(2, 3)
    padding = _call(
        graph,
        exir_ops.backend.tosa.CONST_SHAPE.default,
        ([0, 0, 0, 0],),
        [0, 0, 0, 0],
    )
    padded = _call(
        graph,
        exir_ops.backend.tosa.PAD.default,
        (x, padding),
        torch.ones(2, 3),
        {"value": 0.0},
    )
    first_add = _call(
        graph,
        exir_ops.edge.aten.add.Tensor,
        (padded, y),
        torch.ones(2, 3),
    )
    second_add = _call(
        graph,
        exir_ops.edge.aten.add.Tensor,
        (x, y),
        torch.ones(2, 3),
    )
    graph.output((first_add, second_add))

    graph_module = _run_remove_noop(GraphModule(torch.nn.Module(), graph))
    graph_module = FuseDuplicateUsersPass()(graph_module).graph_module
    result = EnsureUniqueOutputNodesPass()(graph_module).graph_module

    assert _count_target(result, exir_ops.edge.aten.add.Tensor) == 1
    outputs = result.graph.output_node().args[0]
    assert outputs[0] is not outputs[1]


class _PipelineInspectionPassManager(ArmPassManager):
    def _transform(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ) -> GraphModule:
        return graph_module


class _DataLayoutNoopPipelineModule(torch.nn.Module):
    def forward(self, x, y):
        selected = torch.select(x, 1, 0)
        permuted = x.permute(0, 2, 1)
        permuted = torch.nn.functional.pad(permuted, (0, 0, 0, 0, 0, 0))
        permuted = permuted.permute(0, 2, 1)
        padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0))
        return selected, permuted, padded + y, x + y


def test_data_layout_noop_cleanup_backend_pipeline():
    exported_program = export(
        _DataLayoutNoopPipelineModule(),
        (torch.ones(2, 1, 3), torch.ones(2, 1, 3)),
        strict=True,
    )
    edge_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()

    assert (
        _count_target(edge_program.graph_module, exir_ops.edge.aten.select_copy.int)
        == 1
    )
    assert (
        _count_target(
            edge_program.graph_module,
            exir_ops.edge.aten.constant_pad_nd.default,
        )
        == 2
    )
    assert (
        _count_target(
            edge_program.graph_module, exir_ops.edge.aten.permute_copy.default
        )
        == 2
    )
    assert _count_target(edge_program.graph_module, exir_ops.edge.aten.add.Tensor) == 2

    graph_module = ArmPassManager(
        TosaCompileSpec("TOSA-1.0+FP")
    ).transform_to_backend_pipeline(edge_program, edge_program.graph_module)

    assert _count_target(graph_module, exir_ops.backend.tosa.PAD.default) == 0
    assert _count_target(graph_module, exir_ops.backend.tosa.SLICE.default) == 0
    assert _count_target(graph_module, exir_ops.edge.aten.permute_copy.default) == 0
    assert _count_target(graph_module, exir_ops.backend.tosa.TRANSPOSE.default) == 0

    add_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.ADD.default
    ]
    identity_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.IDENTITY.default
    ]
    assert len(add_nodes) == 1
    assert len(identity_nodes) == 2
    assert all(node.args[0] is add_nodes[0] for node in identity_nodes)

    outputs = graph_module.graph.output_node().args[0]
    assert len(outputs) == len(set(outputs))
    graph_module.graph.lint()


def test_data_layout_noop_cleanup_pipeline_order():
    exported_program = export(torch.nn.Identity(), (torch.ones(1),), strict=True)
    pass_manager = _PipelineInspectionPassManager(TosaCompileSpec("TOSA-1.0+FP"))

    pass_manager.transform_to_backend_pipeline(
        exported_program, exported_program.graph_module
    )

    pass_types = [type(pass_) for pass_ in pass_manager.passes]
    exir_to_tosa_index = pass_types.index(ExirToTosaPass)
    cleanup_indices = [
        index
        for index, pass_type in enumerate(pass_types)
        if pass_type is RemoveNoopPass
    ]
    pre_tosa_cleanup = max(
        index for index in cleanup_indices if index < exir_to_tosa_index
    )
    post_tosa_cleanup = min(
        index for index in cleanup_indices if index > exir_to_tosa_index
    )

    assert pass_types[pre_tosa_cleanup + 1] is CanonicalizeViewCopyPermutePass
    assert pass_types[post_tosa_cleanup + 1] is InsertRescalePass
    assert pass_types[post_tosa_cleanup + 2] is FuseDuplicateUsersPass
    assert pass_types[post_tosa_cleanup + 3] is EnsureUniqueOutputNodesPass
