# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import (
    PropagateViewCopyPermuteDownPass,
    PropagateViewCopyPermuteUpPass,
)

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

input_t = Tuple[torch.Tensor]

PERMUTE = exir_ops.edge.aten.permute_copy.default
VIEW = exir_ops.edge.aten.view_copy.default
ADD = exir_ops.edge.aten.add.Tensor
RELU = exir_ops.edge.aten.relu.default
NEG = exir_ops.edge.aten.neg.default
MM = exir_ops.edge.aten.mm.default
RESCALE = exir_ops.backend.tosa.RESCALE.default
CLONE = exir_ops.edge.dim_order_ops._clone_dim_order.default
TABLE = exir_ops.backend.tosa.TABLE.default
SCATTER = exir_ops.backend.tosa.SCATTER.default
CAT = exir_ops.edge.aten.cat.default
SLICE = exir_ops.edge.aten.slice_copy.Tensor
SUM = exir_ops.edge.aten.sum.dim_IntList
MEAN = exir_ops.edge.aten.mean.dim


def _assert_call_targets(
    predicate: Callable[[list[object]], None],
) -> Callable[[ExportedProgram], ExportedProgram]:
    def check_order(exported_program: ExportedProgram) -> ExportedProgram:
        targets = [
            node.target
            for node in exported_program.graph_module.graph.nodes
            if node.op == "call_function"
        ]
        predicate(targets)
        return exported_program

    return check_order


class DownwardPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).relu().neg()

    data = (torch.randn(1, 2, 3, 4),)


def test_propagate_permute_down_through_transparent_ops_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(PERMUTE) < targets.index(RELU) < targets.index(NEG)

    pipeline = PassPipeline[input_t](
        DownwardPermute(),
        DownwardPermute.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class DownwardBinaryPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1) + y.permute(0, 2, 3, 1)

    data = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))


class DownwardView(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(2, 12).relu().neg()

    data = (torch.randn(2, 3, 4),)


def test_propagate_view_down_through_transparent_ops_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(VIEW) < targets.index(RELU) < targets.index(NEG)

    pipeline = PassPipeline[input_t](
        DownwardView(),
        DownwardView.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class UpwardPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu().neg().permute(0, 2, 3, 1)

    data = (torch.randn(1, 2, 3, 4),)


def test_propagate_permute_up_through_transparent_ops_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(PERMUTE) < targets.index(RELU) < targets.index(NEG)

    pipeline = PassPipeline[input_t](
        UpwardPermute(),
        UpwardPermute.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class UpwardBinaryPermute(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x + y).permute(0, 2, 3, 1)

    data = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))


def test_propagate_permute_up_swaps_with_binary_transparent_op_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.count(PERMUTE) == 1
        assert targets.index(ADD) < targets.index(PERMUTE)

    pipeline = PassPipeline[Tuple[torch.Tensor, torch.Tensor]](
        UpwardBinaryPermute(),
        UpwardBinaryPermute.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class UpwardView(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu().neg().view(2, 12)

    data = (torch.randn(2, 3, 4),)


def test_propagate_view_up_through_transparent_ops_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(VIEW) < targets.index(RELU) < targets.index(NEG)

    pipeline = PassPipeline[input_t](
        UpwardView(),
        UpwardView.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class StopAtNonTransparent(torch.nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.permute(1, 0).mm(weight)

    data = (torch.randn(3, 2), torch.randn(3, 4))


def test_propagate_stops_at_non_transparent_ops_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(PERMUTE) < targets.index(MM)

    pipeline = PassPipeline[Tuple[torch.Tensor, torch.Tensor]](
        StopAtNonTransparent(),
        StopAtNonTransparent.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class StopAtBranch(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        return y.relu() + y.neg()

    data = (torch.randn(1, 2, 3, 4),)


def test_propagate_stops_at_branches_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(PERMUTE) < targets.index(RELU)
        assert targets.index(PERMUTE) < targets.index(NEG)

    pipeline = PassPipeline[input_t](
        StopAtBranch(),
        StopAtBranch.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


class StopAtSharedTransformInput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        return (y * y.sigmoid()).permute(0, 3, 1, 2)

    data = (torch.randn(1, 2, 3, 4),)


class StopAtParameter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 2, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.weight).permute(0, 2, 3, 1)

    data = (torch.randn(1, 2, 3, 4),)


def test_propagate_moves_before_parameter_tosa_FP() -> None:
    def predicate(targets: list[object]) -> None:
        assert targets.index(ADD) < targets.index(PERMUTE)

    pipeline = PassPipeline[input_t](
        StopAtParameter(),
        StopAtParameter.data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
        },
        pass_list=[PropagateViewCopyPermuteUpPass],
        pass_functions=[_assert_call_targets(predicate)],
    )
    pipeline.run()


def _run_pass_on_graph_module(
    graph: torch.fx.Graph,
    pass_cls: type[ArmPass] = PropagateViewCopyPermuteUpPass,
) -> torch.fx.GraphModule:
    graph.lint()
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    result = pass_cls().call(graph_module)
    return result.graph_module


def _run_pass_on_graph(
    graph: torch.fx.Graph,
    pass_cls: type[ArmPass] = PropagateViewCopyPermuteUpPass,
) -> list[object]:
    graph_module = _run_pass_on_graph_module(graph, pass_cls)
    return [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]


def test_is_swappable_rejects_unnormalized_keep_dim_operator() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    sum_node = graph.call_function(SUM, args=(x, [1], False))

    with pytest.raises(
        RuntimeError,
        match="expects keep_dim=True for reduction ops to simplify propagation logic, got",
    ):
        PropagateViewCopyPermuteUpPass().is_swappable(sum_node)


def test_down_pass_moves_permute_after_transparent_chain() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(relu,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(neg)

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.index(RELU) < targets.index(NEG) < targets.index(PERMUTE)


def test_down_pass_skips_propagation_for_u85_like_tosa_int_cf() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(relu,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(neg)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT+cf")):
        targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.index(PERMUTE) < targets.index(RELU) < targets.index(NEG)


def test_down_pass_still_canonicalizes_for_u85_like_tosa_int_cf() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3))
    first_permute = graph.call_function(PERMUTE, args=(x, [0, 2, 1]))
    first_permute.meta["val"] = torch.empty((1, 3, 2))
    second_permute = graph.call_function(PERMUTE, args=(first_permute, [0, 2, 1]))
    second_permute.meta["val"] = torch.empty((1, 2, 3))
    graph.output(second_permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT+cf")):
        targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert PERMUTE not in targets


def test_down_pass_moves_view_after_transparent_chain() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 3, 4))
    view = graph.call_function(VIEW, args=(x, [2, 12]))
    view.meta["val"] = torch.empty((2, 12))
    relu = graph.call_function(RELU, args=(view,))
    relu.meta["val"] = torch.empty((2, 12))
    neg = graph.call_function(NEG, args=(relu,))
    neg.meta["val"] = torch.empty((2, 12))
    graph.output(neg)

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.index(RELU) < targets.index(NEG) < targets.index(VIEW)


def test_down_pass_moves_permute_to_graph_output() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(relu,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(neg)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    nodes = list(graph_module.graph.nodes)
    output = next(node for node in nodes if node.op == "output")
    moved_permute = next(node for node in nodes if node.target == PERMUTE)
    moved_neg = next(node for node in nodes if node.target == NEG)

    assert output.args[0] is moved_permute
    assert moved_permute.args[0] is moved_neg
    assert nodes.index(moved_neg) < nodes.index(moved_permute) < nodes.index(output)


def test_down_pass_moves_permute_to_matching_output_branch() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    left = graph.call_function(RELU, args=(x,))
    left.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(relu,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output((left, neg))

    graph_module = torch.fx.GraphModule({}, graph)
    output = next(node for node in graph.nodes if node.op == "output")
    PropagateViewCopyPermuteDownPass()._move_node(permute, output, neg)
    graph.lint()
    graph_module.recompile()

    assert output.args[0] == (left, permute)
    assert relu.args[0] is x
    assert permute.args[0] is neg


def test_up_pass_moves_permute_to_graph_input() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    relu = graph.call_function(RELU, args=(x,))
    relu.meta["val"] = torch.empty((1, 2, 3, 4))
    neg = graph.call_function(NEG, args=(relu,))
    neg.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(neg, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(permute)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    nodes = list(graph_module.graph.nodes)
    x = next(node for node in nodes if node.op == "placeholder")
    moved_permute = next(node for node in nodes if node.target == PERMUTE)
    moved_relu = next(node for node in nodes if node.target == RELU)

    assert moved_permute.args[0] is x
    assert moved_relu.args[0] is moved_permute
    assert nodes.index(x) < nodes.index(moved_permute) < nodes.index(moved_relu)


def test_up_pass_fuses_duplicate_permutes_at_placeholder() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 4, 3, 3))
    left_slice = graph.call_function(SLICE, args=(x, 1, 0, 2))
    left_slice.meta["val"] = torch.empty((1, 2, 3, 3))
    right_slice = graph.call_function(SLICE, args=(x, 1, 2, 4))
    right_slice.meta["val"] = torch.empty((1, 2, 3, 3))
    left_permute = graph.call_function(PERMUTE, args=(left_slice, [0, 2, 3, 1]))
    left_permute.meta["val"] = torch.empty((1, 3, 3, 2))
    right_permute = graph.call_function(PERMUTE, args=(right_slice, [0, 2, 3, 1]))
    right_permute.meta["val"] = torch.empty((1, 3, 3, 2))
    graph.output((left_permute, right_permute))

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    permutes = [node for node in call_nodes if node.target == PERMUTE]
    slices = [node for node in call_nodes if node.target == SLICE]
    x = next(node for node in graph_module.graph.nodes if node.op == "placeholder")

    assert len(permutes) == 1
    assert len(slices) == 2
    assert permutes[0].args == (x, [0, 2, 3, 1])
    assert [slice_node.args for slice_node in slices] == [
        (permutes[0], 3, 0, 2),
        (permutes[0], 3, 2, 4),
    ]


def test_up_pass_refreshes_permute_meta_before_view_slice_swap() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((3, 2, 8, 16))
    slice_node = graph.call_function(SLICE, args=(x, 0, 0, 1))
    slice_node.meta["val"] = torch.empty((1, 2, 8, 16))
    permute = graph.call_function(PERMUTE, args=(slice_node, [0, 3, 1, 2]))
    permute.meta["val"] = torch.empty((1, 16, 2, 8))
    view = graph.call_function(VIEW, args=(permute, [1, 32, 8]))
    view.meta["val"] = torch.empty((1, 32, 8))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    permute = next(node for node in call_nodes if node.target == PERMUTE)
    view = next(node for node in call_nodes if node.target == VIEW)
    slice_node = next(node for node in call_nodes if node.target == SLICE)
    graph_input = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )

    assert permute.args == (graph_input, [0, 3, 1, 2])
    assert permute.meta["val"].shape == torch.Size((3, 16, 2, 8))
    assert view.args == (permute, [3, 32, 8])
    assert slice_node.args == (view, 0, 0, 1)


def test_up_pass_keeps_scatter_input_view_after_slice() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((3, 16, 2, 8))
    indices = graph.placeholder("indices")
    indices.meta["val"] = torch.empty((1, 4), dtype=torch.int32)
    data = graph.placeholder("data")
    data.meta["val"] = torch.empty((1, 4, 16))
    slice_node = graph.call_function(SLICE, args=(x, 0, 0, 1))
    slice_node.meta["val"] = torch.empty((1, 16, 2, 8))
    view = graph.call_function(VIEW, args=(slice_node, [1, 32, 8]))
    view.meta["val"] = torch.empty((1, 32, 8))
    scatter = graph.call_function(SCATTER, args=(view, indices, data))
    scatter.meta["val"] = torch.empty((1, 32, 8))
    graph.output(scatter)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    view = next(node for node in call_nodes if node.target == VIEW)
    slice_node = next(node for node in call_nodes if node.target == SLICE)
    scatter = next(node for node in call_nodes if node.target == SCATTER)

    assert view.args == (slice_node, [1, 32, 8])
    assert scatter.args[0] is view


def test_up_pass_hoists_matching_transform_chain_across_slice_fanout() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 4, 3, 3))
    left_slice = graph.call_function(SLICE, args=(x, 1, 0, 2))
    left_slice.meta["val"] = torch.empty((1, 2, 3, 3))
    right_slice = graph.call_function(SLICE, args=(x, 1, 2, 4))
    right_slice.meta["val"] = torch.empty((1, 2, 3, 3))
    left_view = graph.call_function(VIEW, args=(left_slice, [1, 2, 9]))
    left_view.meta["val"] = torch.empty((1, 2, 9))
    right_view = graph.call_function(VIEW, args=(right_slice, [1, 2, 9]))
    right_view.meta["val"] = torch.empty((1, 2, 9))
    left_permute = graph.call_function(PERMUTE, args=(left_view, [0, 2, 1]))
    left_permute.meta["val"] = torch.empty((1, 9, 2))
    right_permute = graph.call_function(PERMUTE, args=(right_view, [0, 2, 1]))
    right_permute.meta["val"] = torch.empty((1, 9, 2))
    graph.output((left_permute, right_permute))

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    views = [node for node in call_nodes if node.target == VIEW]
    permutes = [node for node in call_nodes if node.target == PERMUTE]
    slices = [node for node in call_nodes if node.target == SLICE]
    graph_input = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )

    assert len(views) == 1
    assert len(permutes) == 1
    assert len(slices) == 2
    assert views[0].args == (graph_input, [1, 4, 9])
    assert permutes[0].args == (views[0], [0, 2, 1])
    assert [slice_node.args for slice_node in slices] == [
        (permutes[0], 2, 0, 2),
        (permutes[0], 2, 2, 4),
    ]


def test_up_pass_hoists_unit_slice_views_with_different_args() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((5, 2, 3, 4, 6))
    left_slice = graph.call_function(SLICE, args=(x, 2, 0, 1))
    left_slice.meta["val"] = torch.empty((5, 2, 1, 4, 6))
    right_slice = graph.call_function(SLICE, args=(x, 2, 1, 2))
    right_slice.meta["val"] = torch.empty((5, 2, 1, 4, 6))
    left_view = graph.call_function(VIEW, args=(left_slice, [5, 2, 4, 6]))
    left_view.meta["val"] = torch.empty((5, 2, 4, 6))
    right_view = graph.call_function(VIEW, args=(right_slice, [5, 2, 24]))
    right_view.meta["val"] = torch.empty((5, 2, 24))
    graph.output((left_view, right_view))

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    views = [node for node in call_nodes if node.target == VIEW]
    slices = [node for node in call_nodes if node.target == SLICE]
    graph_input = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )

    assert [view.args for view in views] == [
        (graph_input, [5, 2, 12, 6]),
        (graph_input, [5, 2, 72]),
    ]
    assert [slice_node.args for slice_node in slices] == [
        (views[0], 2, 0, 4),
        (views[1], 2, 24, 48),
    ]


def test_up_pass_keeps_mismatched_transform_slice_fanout_split() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 4, 3, 3))
    left_slice = graph.call_function(SLICE, args=(x, 1, 0, 2))
    left_slice.meta["val"] = torch.empty((1, 2, 3, 3))
    right_slice = graph.call_function(SLICE, args=(x, 1, 2, 4))
    right_slice.meta["val"] = torch.empty((1, 2, 3, 3))
    left_view = graph.call_function(VIEW, args=(left_slice, [1, 2, 9]))
    left_view.meta["val"] = torch.empty((1, 2, 9))
    right_permute = graph.call_function(PERMUTE, args=(right_slice, [0, 2, 3, 1]))
    right_permute.meta["val"] = torch.empty((1, 3, 3, 2))
    graph.output((left_view, right_permute))

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    slices = [node for node in call_nodes if node.target == SLICE]

    assert len(slices) == 2
    assert slices[0].args[0] is not slices[1].args[0]


def test_down_pass_moves_matching_input_permutations_after_binary_op() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4))
    x_permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    x_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    y_permute = graph.call_function(PERMUTE, args=(y, [0, 2, 3, 1]))
    y_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    add = graph.call_function(ADD, args=(x_permute, y_permute))
    add.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(add)

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.count(PERMUTE) == 1
    assert targets.index(ADD) < targets.index(PERMUTE)


def test_down_pass_keeps_sunk_view_before_rank_reducing_permute() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 8, 1, 32))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((2, 8, 1, 32))
    x_view = graph.call_function(VIEW, args=(x, [2, 8, 32]))
    x_view.meta["val"] = torch.empty((2, 8, 32))
    y_view = graph.call_function(VIEW, args=(y, [2, 8, 32]))
    y_view.meta["val"] = torch.empty((2, 8, 32))
    add = graph.call_function(ADD, args=(x_view, y_view))
    add.meta["val"] = torch.empty((2, 8, 32))
    output_view = graph.call_function(VIEW, args=(add, [2, 8, 32]))
    output_view.meta["val"] = torch.empty((2, 8, 32))
    permute = graph.call_function(PERMUTE, args=(output_view, [0, 2, 1]))
    permute.meta["val"] = torch.empty((2, 32, 8))
    graph.output(permute)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    add = next(node for node in call_nodes if node.target == ADD)
    output_view = next(node for node in call_nodes if node.target == VIEW)
    permute = next(node for node in call_nodes if node.target == PERMUTE)

    assert targets.count(VIEW) == 1
    assert targets.index(ADD) < targets.index(VIEW) < targets.index(PERMUTE)
    assert add.meta["val"].shape == torch.Size((2, 8, 1, 32))
    assert output_view.args[0] is add
    assert permute.args[0] is output_view


def test_down_pass_canonicalizes_horizontally_fused_singleton_permute() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 1, 1, 1))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 1, 1, 1))
    x_permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    x_permute.meta["val"] = torch.empty((1, 1, 1, 1))
    y_permute = graph.call_function(PERMUTE, args=(y, [0, 2, 3, 1]))
    y_permute.meta["val"] = torch.empty((1, 1, 1, 1))
    add = graph.call_function(ADD, args=(x_permute, y_permute))
    add.meta["val"] = torch.empty((1, 1, 1, 1))
    graph.output(add)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    add = next(node for node in call_nodes if node.target == ADD)

    assert targets.count(PERMUTE) == 0
    assert targets.count(VIEW) == 0
    assert [input_node.name for input_node in add.all_input_nodes] == ["x", "y"]
    assert next(iter(add.users)).op == "output"


def test_down_pass_moves_matching_input_permutations_after_cat() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4))
    x_permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    x_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    y_permute = graph.call_function(PERMUTE, args=(y, [0, 2, 3, 1]))
    y_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    cat_node = graph.call_function(CAT, args=([x_permute, y_permute], 3))
    cat_node.meta["val"] = torch.empty((1, 3, 4, 4))
    graph.output(cat_node)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    cat_node = next(node for node in call_nodes if node.target == CAT)

    assert targets.count(PERMUTE) == 1
    assert targets.index(CAT) < targets.index(PERMUTE)
    assert cat_node.args[1] == 1


def test_down_pass_swaps_concat_with_matching_input_permutations() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4))
    x_permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    x_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    y_permute = graph.call_function(PERMUTE, args=(y, [0, 2, 3, 1]))
    y_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    cat_node = graph.call_function(CAT, args=([x_permute, y_permute], 3))
    cat_node.meta["val"] = torch.empty((1, 3, 4, 4))
    graph.output(cat_node)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    cat_node = next(node for node in call_nodes if node.target == CAT)
    permute = next(node for node in call_nodes if node.target == PERMUTE)

    assert targets.count(PERMUTE) == 1
    assert targets.index(CAT) < targets.index(PERMUTE)
    assert [input_node.name for input_node in cat_node.args[0]] == ["x", "y"]
    assert cat_node.args[1] == 1
    assert cat_node.meta["val"].shape == torch.Size((1, 4, 3, 4))
    assert permute.args == (cat_node, [0, 2, 3, 1])


def test_up_pass_moves_noop_input_permutations_before_cat() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 1, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 1, 3, 4))
    cat_node = graph.call_function(CAT, args=([x, y], 1))
    cat_node.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(cat_node, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(permute)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    cat_node = next(node for node in call_nodes if node.target == CAT)

    assert targets.count(PERMUTE) == 0
    assert targets.count(VIEW) == 2
    assert cat_node.args[1] == 3
    assert cat_node.meta["val"].shape == torch.Size((1, 3, 4, 2))
    assert all(input_node.target == VIEW for input_node in cat_node.args[0])


def test_up_pass_swaps_concat_with_noop_output_permutation() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 1, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 1, 3, 4))
    cat_node = graph.call_function(CAT, args=([x, y], 1))
    cat_node.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(cat_node, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(permute)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteUpPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    cat_node = next(node for node in call_nodes if node.target == CAT)

    assert targets.count(PERMUTE) == 0
    assert targets.count(VIEW) == 2
    assert cat_node.args[1] == 3
    assert cat_node.meta["val"].shape == torch.Size((1, 3, 4, 2))
    assert all(input_node.target == VIEW for input_node in cat_node.args[0])


def test_down_pass_keeps_shared_input_permutations_before_cat() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4))
    x_permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    x_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    y_permute = graph.call_function(PERMUTE, args=(y, [0, 2, 3, 1]))
    y_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(x_permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    cat_node = graph.call_function(CAT, args=([x_permute, y_permute], 3))
    cat_node.meta["val"] = torch.empty((1, 3, 4, 4))
    graph.output((cat_node, relu))

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    cat_node = next(node for node in call_nodes if node.target == CAT)

    assert targets.count(PERMUTE) == 2
    assert [input_node.target for input_node in cat_node.args[0]] == [
        PERMUTE,
        PERMUTE,
    ]
    assert cat_node.args[1] == 3


def test_down_pass_moves_permutation_after_reduction() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    sum_node = graph.call_function(SUM, args=(permute, [3], True))
    sum_node.meta["val"] = torch.empty((1, 3, 4, 1))
    graph.output(sum_node)

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    sum_node = next(node for node in call_nodes if node.target == SUM)
    transform = next(node for node in call_nodes if node.target in (PERMUTE, VIEW))

    assert targets.index(SUM) < targets.index(transform.target)
    assert sum_node.args[1] == [1]
    assert transform.meta["val"].shape == torch.Size((1, 3, 4, 1))


def test_down_pass_stops_when_fanout_does_not_converge() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(permute,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output((relu, neg))

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.count(PERMUTE) == 1
    assert targets.index(PERMUTE) < targets.index(RELU)
    assert targets.index(PERMUTE) < targets.index(NEG)


def test_down_pass_splits_permute_over_slice_fanout() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 4, 3, 3))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 3, 4))
    left_slice = graph.call_function(SLICE, args=(permute, 3, 0, 2))
    left_slice.meta["val"] = torch.empty((1, 3, 3, 2))
    right_slice = graph.call_function(SLICE, args=(permute, 3, 2, 4))
    right_slice.meta["val"] = torch.empty((1, 3, 3, 2))
    left_view = graph.call_function(VIEW, args=(left_slice, [1, 9, 2]))
    left_view.meta["val"] = torch.empty((1, 9, 2))
    right_view = graph.call_function(VIEW, args=(right_slice, [1, 9, 2]))
    right_view.meta["val"] = torch.empty((1, 9, 2))
    left_permute = graph.call_function(PERMUTE, args=(left_view, [0, 2, 1]))
    left_permute.meta["val"] = torch.empty((1, 2, 9))
    right_permute = graph.call_function(PERMUTE, args=(right_view, [0, 2, 1]))
    right_permute.meta["val"] = torch.empty((1, 2, 9))
    graph.output((left_permute, right_permute))

    graph_module = _run_pass_on_graph_module(graph, PropagateViewCopyPermuteDownPass)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    slices = [node for node in call_nodes if node.target == SLICE]
    permutes = [node for node in call_nodes if node.target == PERMUTE]
    graph_input = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )

    assert [slice_node.args for slice_node in slices] == [
        (graph_input, 1, 0, 2),
        (graph_input, 1, 2, 4),
    ]
    assert all(permute_node.args[0].target == SLICE for permute_node in permutes)


def test_down_pass_stops_when_fanout_branch_has_nontransparent_op() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 3))
    weight = graph.placeholder("weight")
    weight.meta["val"] = torch.empty((2, 2))
    permute = graph.call_function(PERMUTE, args=(x, [1, 0]))
    permute.meta["val"] = torch.empty((3, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((3, 2))
    mm = graph.call_function(MM, args=(permute, weight))
    mm.meta["val"] = torch.empty((3, 2))
    add = graph.call_function(ADD, args=(relu, mm))
    add.meta["val"] = torch.empty((3, 2))
    graph.output(add)

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.count(PERMUTE) == 1
    assert targets.index(PERMUTE) < targets.index(RELU)
    assert targets.index(PERMUTE) < targets.index(MM)


def test_down_pass_stops_when_convergence_has_untracked_input() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 3, 4, 2))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(permute,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(permute,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    cat_node = graph.call_function(CAT, args=([relu, neg, y], 3))
    cat_node.meta["val"] = torch.empty((1, 3, 4, 6))
    graph.output(cat_node)

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.count(PERMUTE) == 1
    assert targets.index(PERMUTE) < targets.index(RELU)
    assert targets.index(PERMUTE) < targets.index(NEG)


def test_down_pass_stops_view_before_cat_converging_fanout() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    view = graph.call_function(VIEW, args=(x, [1, 3, 4, 2]))
    view.meta["val"] = torch.empty((1, 3, 4, 2))
    relu = graph.call_function(RELU, args=(view,))
    relu.meta["val"] = torch.empty((1, 3, 4, 2))
    neg = graph.call_function(NEG, args=(view,))
    neg.meta["val"] = torch.empty((1, 3, 4, 2))
    cat_node = graph.call_function(CAT, args=([relu, neg], 3))
    cat_node.meta["val"] = torch.empty((1, 3, 4, 4))
    graph.output(cat_node)

    targets = _run_pass_on_graph(graph, PropagateViewCopyPermuteDownPass)

    assert targets.count(VIEW) == 1
    assert targets.index(VIEW) < targets.index(RELU)
    assert targets.index(VIEW) < targets.index(NEG)


def test_up_pass_fuses_equivalent_output_permutations_before_fan_out() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    relu = graph.call_function(RELU, args=(x,))
    relu.meta["val"] = torch.empty((1, 2, 3, 4))
    first_permute = graph.call_function(PERMUTE, args=(relu, [0, 2, 3, 1]))
    first_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    second_permute = graph.call_function(PERMUTE, args=(relu, [0, 2, 3, 1]))
    second_permute.meta["val"] = torch.empty((1, 3, 4, 2))
    add = graph.call_function(ADD, args=(first_permute, second_permute))
    add.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(add)
    graph.lint()
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    result = PropagateViewCopyPermuteUpPass().call(graph_module)
    targets = [
        node.target
        for node in result.graph_module.graph.nodes
        if node.op == "call_function"
    ]

    assert targets.count(PERMUTE) == 1
    assert targets.index(PERMUTE) < targets.index(RELU) < targets.index(ADD)


def test_propagate_moves_before_dtype_changing_rescale() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(x, torch.int8, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output(permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(PERMUTE) < targets.index(RESCALE)


def test_propagate_fuses_permute_view_around_table() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 3, 4), dtype=torch.int8)
    table = graph.placeholder("table")
    table.meta["val"] = torch.empty((256,), dtype=torch.int8)
    permute = graph.call_function(PERMUTE, args=(x, [1, 0, 2]))
    permute.meta["val"] = torch.empty((3, 2, 4), dtype=torch.int8)
    view = graph.call_function(VIEW, args=(permute, [3, 8]))
    view.meta["val"] = torch.empty((3, 8), dtype=torch.int8)
    table_node = graph.call_function(TABLE, args=(view, table))
    table_node.meta["val"] = torch.empty((3, 8), dtype=torch.int8)
    output_view = graph.call_function(VIEW, args=(table_node, [3, 2, 4]))
    output_view.meta["val"] = torch.empty((3, 2, 4), dtype=torch.int8)
    output_permute = graph.call_function(PERMUTE, args=(output_view, [1, 0, 2]))
    output_permute.meta["val"] = torch.empty((2, 3, 4), dtype=torch.int8)
    graph.output(output_permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        graph_module = _run_pass_on_graph_module(
            graph, PropagateViewCopyPermuteDownPass
        )
    targets = [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]

    assert targets == [TABLE]


def test_propagate_stops_at_per_channel_rescale() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 3, 4, 5), dtype=torch.int32)
    rescale = graph.call_function(
        RESCALE, args=(x, torch.int8, [1.0, 1.0, 1.0, 1.0, 1.0], 0, 0)
    )
    rescale.meta["val"] = torch.empty((2, 3, 4, 5), dtype=torch.int8)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 3, 1, 2]))
    permute.meta["val"] = torch.empty((2, 5, 3, 4), dtype=torch.int8)
    graph.output(permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(RESCALE) < targets.index(PERMUTE)


def test_propagate_stops_at_rescale_changing_special_dtype() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 1, 1, 15), dtype=torch.int32)
    x.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
    rescale = graph.call_function(RESCALE, args=(x, torch.int32, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 1, 1, 15), dtype=torch.int32)
    view = graph.call_function(VIEW, args=(rescale, [15]))
    view.meta["val"] = torch.empty((15,), dtype=torch.int32)
    graph.output(view)

    targets = _run_pass_on_graph(graph)

    assert targets.index(RESCALE) < targets.index(VIEW)


def test_propagate_up_stops_at_shared_rescale_producer() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((10, 80, 16), dtype=torch.int8)
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((10, 80, 16), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(x, torch.int32, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((10, 80, 16), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(rescale, [1, 0, 2]))
    permute.meta["val"] = torch.empty((80, 10, 16), dtype=torch.int32)
    add = graph.call_function(ADD, args=(rescale, y))
    add.meta["val"] = torch.empty((10, 80, 16), dtype=torch.int32)
    graph.output((permute, add))

    targets = _run_pass_on_graph(graph)

    assert targets.index(RESCALE) < targets.index(PERMUTE)


def test_propagate_up_stops_before_narrowing_rescale_fed_by_binary_op() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    add = graph.call_function(ADD, args=(x, y))
    add.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(add, torch.int8, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output(permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(RESCALE) < targets.index(PERMUTE)


def test_propagate_up_crosses_same_width_rescale_fed_by_binary_op() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    add = graph.call_function(ADD, args=(x, y))
    add.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(add, torch.int32, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    graph.output(permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(PERMUTE) < targets.index(RESCALE)


def test_propagate_up_stops_before_narrowing_rescale_from_shared_placeholder() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(x, torch.int8, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    other = graph.call_function(NEG, args=(x,))
    other.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output((permute, other))

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(RESCALE) < targets.index(PERMUTE)


def test_propagate_up_crosses_widening_rescale_fed_by_binary_op() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    add = graph.call_function(ADD, args=(x, y))
    add.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    rescale = graph.call_function(RESCALE, args=(add, torch.int32, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    graph.output(permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(PERMUTE) < targets.index(RESCALE)


def test_propagate_up_stops_before_narrowing_rescale_behind_unary() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    clone = graph.call_function(CLONE, args=(x,))
    clone.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    rescale = graph.call_function(RESCALE, args=(clone, torch.int8, [1.0], 0, 0))
    rescale.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int8)
    permute = graph.call_function(PERMUTE, args=(rescale, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int8)
    graph.output(permute)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        targets = _run_pass_on_graph(graph)

    assert targets.index(RESCALE) < targets.index(PERMUTE)


def test_propagate_moves_before_int48_special_dtype() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    x.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
    relu = graph.call_function(RELU, args=(x,))
    relu.meta["val"] = torch.empty((1, 2, 3, 4), dtype=torch.int32)
    relu.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
    permute = graph.call_function(PERMUTE, args=(relu, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2), dtype=torch.int32)
    permute.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
    graph.output(permute)

    targets = _run_pass_on_graph(graph)

    assert targets.index(PERMUTE) < targets.index(RELU)


def test_propagate_moves_output_view_before_sum_with_split_dim_remap() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((6, 4))
    sum_node = graph.call_function(SUM, args=(x, [0], True))
    sum_node.meta["val"] = torch.empty((1, 4))
    view = graph.call_function(VIEW, args=(sum_node, [1, 1, 4]))
    view.meta["val"] = torch.empty((1, 1, 4))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    sum_node = next(node for node in call_nodes if node.target == SUM)
    view = next(node for node in call_nodes if node.target == VIEW)

    assert targets.index(VIEW) < targets.index(SUM)
    assert sum_node.args[1] == [0, 1]
    assert view.args[1] == [6, 1, 4]


def test_propagate_updates_view_map_between_arg_updates() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((6, 4))
    slice_node = graph.call_function(SLICE, args=(x, 0, 0, 4))
    slice_node.meta["val"] = torch.empty((4, 4))
    sum_node = graph.call_function(SUM, args=(slice_node, [0], True))
    sum_node.meta["val"] = torch.empty((1, 4))
    view = graph.call_function(VIEW, args=(sum_node, [1, 1, 4]))
    view.meta["val"] = torch.empty((1, 1, 4))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    view = next(node for node in call_nodes if node.target == VIEW)
    slice_node = next(node for node in call_nodes if node.target == SLICE)
    sum_node = next(node for node in call_nodes if node.target == SUM)

    assert targets.index(VIEW) < targets.index(SLICE) < targets.index(SUM)
    assert view.args[1] == [6, 1, 4]
    assert slice_node.args[1] == 0
    assert sum_node.args[1] == [0, 1]


def test_propagate_moves_output_view_before_mean_with_split_dim_remap() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((6, 4))
    mean_node = graph.call_function(MEAN, args=(x, [0], True))
    mean_node.meta["val"] = torch.empty((1, 4))
    view = graph.call_function(VIEW, args=(mean_node, [1, 1, 4]))
    view.meta["val"] = torch.empty((1, 1, 4))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    mean_node = next(node for node in call_nodes if node.target == MEAN)
    view = next(node for node in call_nodes if node.target == VIEW)

    assert targets.index(VIEW) < targets.index(MEAN)
    assert mean_node.args[1] == [0, 1]
    assert view.args[1] == [6, 1, 4]


def test_propagate_keeps_reduction_squeeze_after_sum() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 50, 10, 20))
    sum_node = graph.call_function(SUM, args=(x, [-1], True))
    sum_node.meta["val"] = torch.empty((1, 50, 10, 1))
    view = graph.call_function(VIEW, args=(sum_node, [1, 50, 10]))
    view.meta["val"] = torch.empty((1, 50, 10))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    sum_node = next(node for node in call_nodes if node.target == SUM)
    view = next(node for node in call_nodes if node.target == VIEW)

    assert targets.index(SUM) < targets.index(VIEW)
    assert sum_node.args[1] == [-1]
    assert view.args[1] == [1, 50, 10]


def test_propagate_keeps_unit_slice_before_reordering_view() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 3, 1, 7))
    slice_node = graph.call_function(SLICE, args=(x, 3, 2, 3))
    slice_node.meta["val"] = torch.empty((1, 3, 1, 1))
    view = graph.call_function(VIEW, args=(slice_node, [1, 1, 1, 3]))
    view.meta["val"] = torch.empty((1, 1, 1, 3))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    slice_node = next(node for node in call_nodes if node.target == SLICE)
    view = next(node for node in call_nodes if node.target == VIEW)

    assert targets.index(SLICE) < targets.index(VIEW)
    assert slice_node.args[1:4] == (3, 2, 3)
    assert view.args == (slice_node, [1, 1, 1, 3])


def test_propagate_stops_when_downward_inputs_are_not_equivalent_transforms() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 3, 4, 2))
    permute = graph.call_function(PERMUTE, args=(x, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    add = graph.call_function(ADD, args=(permute, y))
    add.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(add)

    targets = _run_pass_on_graph(graph)

    assert targets.index(PERMUTE) < targets.index(ADD)


def test_propagate_stops_split_dim_view_at_slice() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((6, 4))
    slice_node = graph.call_function(SLICE, args=(x, 0, 0, 4))
    slice_node.meta["val"] = torch.empty((4, 4))
    view = graph.call_function(VIEW, args=(slice_node, [2, 2, 4]))
    view.meta["val"] = torch.empty((2, 2, 4))
    graph.output(view)

    targets = _run_pass_on_graph(graph)

    assert targets.index(SLICE) < targets.index(VIEW)


def test_propagate_stops_merged_trailing_dim_view_at_slice() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((25, 5, 13, 7))
    slice_node = graph.call_function(SLICE, args=(x, 1, 0, 2))
    slice_node.meta["val"] = torch.empty((25, 2, 13, 7))
    view = graph.call_function(VIEW, args=(slice_node, [1, 25, 182]))
    view.meta["val"] = torch.empty((1, 25, 182))
    graph.output(view)

    targets = _run_pass_on_graph(graph)

    assert targets.index(SLICE) < targets.index(VIEW)


def test_propagate_stops_split_dim_view_at_cat() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((3, 4))
    cat_node = graph.call_function(CAT, args=([x, y], 0))
    cat_node.meta["val"] = torch.empty((6, 4))
    view = graph.call_function(VIEW, args=(cat_node, [2, 3, 4]))
    view.meta["val"] = torch.empty((2, 3, 4))
    graph.output(view)

    targets = _run_pass_on_graph(graph)

    assert targets.index(CAT) < targets.index(VIEW)


def test_propagate_keeps_channel_unit_slice_before_reordering_view() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    slice_node = graph.call_function(SLICE, args=(x, 1, 0, 1))
    slice_node.meta["val"] = torch.empty((1, 1, 3, 4))
    view = graph.call_function(VIEW, args=(slice_node, [1, 3, 4, 1]))
    view.meta["val"] = torch.empty((1, 3, 4, 1))
    graph.output(view)

    graph_module = _run_pass_on_graph_module(graph)
    call_nodes = [
        node for node in graph_module.graph.nodes if node.op == "call_function"
    ]
    targets = [node.target for node in call_nodes]
    slice_node = next(node for node in call_nodes if node.target == SLICE)
    view = next(node for node in call_nodes if node.target == VIEW)

    assert targets.index(SLICE) < targets.index(VIEW)
    assert slice_node.args[1:4] == (1, 0, 1)
    assert view.args == (slice_node, [1, 3, 4, 1])


def test_propagate_up_stops_at_multiple_distinct_edge_nodes() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4))
    add = graph.call_function(ADD, args=(x, y))
    add.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(add, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(permute)

    targets = _run_pass_on_graph(graph)

    assert targets.count(PERMUTE) == 1
    assert targets.index(ADD) < targets.index(PERMUTE)


def test_propagate_up_moves_to_top_node_before_distinct_edge_nodes() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3, 4))
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((1, 2, 3, 4))
    add = graph.call_function(ADD, args=(x, y))
    add.meta["val"] = torch.empty((1, 2, 3, 4))
    relu = graph.call_function(RELU, args=(add,))
    relu.meta["val"] = torch.empty((1, 2, 3, 4))
    permute = graph.call_function(PERMUTE, args=(relu, [0, 2, 3, 1]))
    permute.meta["val"] = torch.empty((1, 3, 4, 2))
    graph.output(permute)

    targets = _run_pass_on_graph(graph)

    assert targets.count(PERMUTE) == 1
    assert targets.index(ADD) < targets.index(PERMUTE) < targets.index(RELU)


def test_propagate_stops_rank_changing_view_at_slice() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((1, 2, 3))
    slice_node = graph.call_function(SLICE, args=(x, 0, 0, 1))
    slice_node.meta["val"] = torch.empty((1, 2, 3))
    view = graph.call_function(VIEW, args=(slice_node, [2, 3]))
    view.meta["val"] = torch.empty((2, 3))
    graph.output(view)

    targets = _run_pass_on_graph(graph)

    assert targets.index(SLICE) < targets.index(VIEW)
