# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import torch
from executorch.backends.arm._passes.remove_safe_softmax_guard_pass import (
    RemoveSafeSoftmaxGuardPass,
)
from executorch.backends.arm.common.pipeline_config import (  # type: ignore[attr-defined]
    ArmPassPipelineConfig,
    SDPASafeSoftmaxGuardPolicy,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import (
    _get_tosa_operator_distribution,
    ArmTester,
)
from executorch.backends.test.harness.stages import StageType
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torch.fx import GraphModule, Node


class SDPA(torch.nn.Module):
    def __init__(self, attn_mask: torch.Tensor | None = None) -> None:
        super().__init__()
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=self.attn_mask,
        )


class RegularSoftmax(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=-1)


class SafeSoftmaxGuardWithDtypeOverride(torch.nn.Module):
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        softmax = torch.softmax(scores, dim=-1)
        pred = torch.all(scores == -torch.inf, dim=-1, keepdim=True)
        zeros = torch.full_like(softmax, 0, dtype=torch.float64)
        return torch.where(pred, zeros, softmax)


def _to_edge_graph(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    return to_edge(export(module, inputs, strict=True)).exported_program().graph_module


def _sdpa_inputs() -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(1, 3, 4, 5) for _ in range(3))


def _op_counts(graph_module: GraphModule) -> Counter:
    return Counter(
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    )


def _get_op_node(graph_module: GraphModule, target: object) -> Node:
    nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == target
    ]
    assert len(nodes) == 1
    return nodes[0]


def _assert_guard_not_removed(graph_module: GraphModule) -> None:
    result = RemoveSafeSoftmaxGuardPass()(graph_module)

    assert result is not None
    assert not result.modified
    assert _op_counts(result.graph_module)[exir_ops.edge.aten.where.self] == 1


def test_remove_safe_softmax_guard_pass_removes_exact_edge_pattern():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())

    result = RemoveSafeSoftmaxGuardPass()(graph_module)
    counts = _op_counts(result.graph_module)

    assert result.modified
    assert counts[exir_ops.edge.aten._softmax.default] == 1
    assert counts[exir_ops.edge.aten.where.self] == 0
    assert counts[exir_ops.edge.aten.eq.Scalar] == 0
    assert counts[exir_ops.edge.aten.logical_not.default] == 0
    assert counts[exir_ops.edge.aten.any.dim] == 0
    assert counts[exir_ops.edge.aten.full_like.default] == 0


def test_remove_safe_softmax_guard_pass_allows_shared_intermediate_users():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    output = next(node for node in graph_module.graph.nodes if node.op == "output")
    where = _get_op_node(graph_module, exir_ops.edge.aten.where.self)
    eq = _get_op_node(graph_module, exir_ops.edge.aten.eq.Scalar)
    softmax = _get_op_node(graph_module, exir_ops.edge.aten._softmax.default)
    output.args = ((where, eq, softmax),)

    result = RemoveSafeSoftmaxGuardPass()(graph_module)
    counts = _op_counts(result.graph_module)

    assert result.modified
    assert counts[exir_ops.edge.aten.where.self] == 0
    assert counts[exir_ops.edge.aten.eq.Scalar] == 1
    assert counts[exir_ops.edge.aten._softmax.default] == 1


def test_remove_safe_softmax_guard_pass_rejects_mismatched_dimensions():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    any_dim = _get_op_node(graph_module, exir_ops.edge.aten.any.dim)
    any_dim.args = (any_dim.args[0], -2, any_dim.args[2])

    _assert_guard_not_removed(graph_module)


def test_remove_safe_softmax_guard_pass_accepts_equivalent_dimensions():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    any_dim = _get_op_node(graph_module, exir_ops.edge.aten.any.dim)
    any_dim.args = (any_dim.args[0], 3, any_dim.args[2])

    result = RemoveSafeSoftmaxGuardPass()(graph_module)

    assert result is not None
    assert result.modified
    assert _op_counts(result.graph_module)[exir_ops.edge.aten.where.self] == 0


def test_remove_safe_softmax_guard_pass_rejects_mismatched_scores():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    eq = _get_op_node(graph_module, exir_ops.edge.aten.eq.Scalar)
    other_scores = next(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    eq.args = (other_scores, eq.args[1])

    _assert_guard_not_removed(graph_module)


def test_remove_safe_softmax_guard_pass_rejects_nonzero_fill_value():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    full_like = _get_op_node(graph_module, exir_ops.edge.aten.full_like.default)
    full_like.args = (full_like.args[0], 1)

    _assert_guard_not_removed(graph_module)


def test_remove_safe_softmax_guard_pass_rejects_output_dtype_change():
    graph_module = _to_edge_graph(
        SafeSoftmaxGuardWithDtypeOverride(), (torch.randn(1, 3, 4, 4),)
    )

    _assert_guard_not_removed(graph_module)


def test_remove_safe_softmax_guard_pass_rejects_non_neg_inf_constant():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    eq = _get_op_node(graph_module, exir_ops.edge.aten.eq.Scalar)
    eq.args = (eq.args[0], -1e9)

    _assert_guard_not_removed(graph_module)


def test_remove_safe_softmax_guard_pass_rejects_keepdim_false():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())
    any_dim = _get_op_node(graph_module, exir_ops.edge.aten.any.dim)
    any_dim.args = (any_dim.args[0], any_dim.args[1], False)

    _assert_guard_not_removed(graph_module)


def test_remove_safe_softmax_guard_pass_is_idempotent():
    graph_module = _to_edge_graph(SDPA(), _sdpa_inputs())

    first_result = RemoveSafeSoftmaxGuardPass()(graph_module)
    second_result = RemoveSafeSoftmaxGuardPass()(first_result.graph_module)

    assert first_result.modified
    assert not second_result.modified


def test_sdpa_safe_softmax_guard_preserve_keeps_guard_before_tosa_lowering():
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+FP")
    compile_spec.set_pass_pipeline_config(
        ArmPassPipelineConfig(
            sdpa_safe_softmax_guard=SDPASafeSoftmaxGuardPolicy.PRESERVE
        )
    )
    tester = ArmTester(SDPA(), _sdpa_inputs(), compile_spec)

    tester.export().to_edge_transform_and_lower()
    graph_module = (
        tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
        .exported_program()
        .graph_module
    )
    counts = dict(_get_tosa_operator_distribution(graph_module))

    assert counts["EQUAL"] == 1
    assert counts["LOGICAL_NOT"] == 2
    assert counts["REDUCE_ANY"] == 1
    assert counts["SELECT"] == 1


def test_sdpa_safe_softmax_guard_remove_when_proven_keeps_guard():
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+FP")
    compile_spec.set_pass_pipeline_config(
        ArmPassPipelineConfig(
            sdpa_safe_softmax_guard=(SDPASafeSoftmaxGuardPolicy.REMOVE_WHEN_PROVEN)
        )
    )
    tester = ArmTester(SDPA(), _sdpa_inputs(), compile_spec)

    tester.export().to_edge_transform_and_lower()
    graph_module = (
        tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
        .exported_program()
        .graph_module
    )
    counts = dict(_get_tosa_operator_distribution(graph_module))

    assert counts["EQUAL"] == 1
    assert counts["LOGICAL_NOT"] == 2
    assert counts["REDUCE_ANY"] == 1
    assert counts["SELECT"] == 1


def test_remove_safe_softmax_guard_pass_does_not_rewrite_regular_softmax():
    graph_module = _to_edge_graph(RegularSoftmax(), (torch.randn(2, 3),))

    result = RemoveSafeSoftmaxGuardPass()(graph_module)
    counts = _op_counts(result.graph_module)

    assert not result.modified
    assert counts[exir_ops.edge.aten._softmax.default] == 1
    assert counts[exir_ops.edge.aten.where.self] == 0


def test_sdpa_safe_softmax_guard_remove_removes_guard_before_tosa_lowering():
    compile_spec = common.get_tosa_compile_spec("TOSA-1.0+FP")
    compile_spec.set_pass_pipeline_config(
        ArmPassPipelineConfig(sdpa_safe_softmax_guard=SDPASafeSoftmaxGuardPolicy.REMOVE)
    )
    tester = ArmTester(SDPA(), _sdpa_inputs(), compile_spec)

    tester.export().to_edge_transform_and_lower()
    graph_module = (
        tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
        .exported_program()
        .graph_module
    )
    counts = dict(_get_tosa_operator_distribution(graph_module))

    assert counts.get("EQUAL", 0) == 0
    assert counts.get("LOGICAL_NOT", 0) == 0
    assert counts.get("REDUCE_ANY", 0) == 0
    assert counts.get("SELECT", 0) == 0
    assert counts["REDUCE_MAX"] == 1
    assert counts["EXP"] == 1
    assert counts["REDUCE_SUM"] == 1
    assert counts["RECIPROCAL"] == 1
