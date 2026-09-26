# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Callable, Dict, Tuple

import pytest
import torch

from executorch.backends.arm._passes import (
    ArmPassManager,
    ConvertInt64OutputOpsToInt32Pass,
    PrepareGatherIndicesPass,
)
from executorch.backends.arm._passes.prepare_gather_indices_pass import (
    is_safe_int32_to_int64_gather_boundary,
)
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.operator_support import DontPartition, DontPartitionName
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph, GraphModule

input_t1 = Tuple[torch.Tensor]  # Input x


#########################################
## Test [int32 | other types] -> int64 ##
#########################################


class CastingToInt64Model(torch.nn.Module):
    def __init__(self, target_dtype: torch.dtype) -> None:
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.target_dtype)


test_data_suite_convert: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
    "fp32_input": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float32), torch.int64),
    "fp16_input": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float16), torch.int64),
}

test_data_suite_remove: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
    "int32_input": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.int64,
    ),
}


TestDataFactory = Callable[[], Tuple[torch.Tensor, torch.dtype]]


@common.parametrize("test_data", test_data_suite_convert)
def test_convert_int64_output_ops_to_int32_tosa_FP_convert_casting(
    test_data: TestDataFactory,
) -> None:
    test_tensor, target_dtype = test_data()
    module = CastingToInt64Model(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op="torch.ops.aten.to.dtype",
        exir_op=[],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # As expected: RuntimeError: Int did not match Long
    pipeline.run()


@common.parametrize("test_data", test_data_suite_remove)
def test_convert_int64_output_ops_to_int32_tosa_FP_remove_casting(
    test_data: TestDataFactory,
) -> None:
    test_tensor, target_dtype = test_data()
    module = CastingToInt64Model(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 0}
    )  # Empty graph without nodes
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


##########################################################
## Test pre-decomposition int64 cast preservation       ##
##########################################################
@pytest.mark.parametrize(
    "compile_spec",
    [
        TosaCompileSpec("TOSA-1.0+FP"),
        VgfCompileSpec("TOSA-1.0+FP"),
        EthosUCompileSpec("ethos-u85-128"),
        EthosUCompileSpec("ethos-u55-128"),
    ],
    ids=["tosa", "vgf", "u85", "u55"],
)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.int32])
def test_pre_decomposition_preserves_int64_cast_output(compile_spec, input_dtype):
    module = CastingToInt64Model(torch.int64)
    values = [1.75, -2.25] if input_dtype == torch.float32 else [1, -2]
    test_input = torch.tensor(values, dtype=input_dtype)
    expected = module(test_input)
    exported_program = torch.export.export(module, (test_input,))

    result = ArmPassManager(compile_spec).transform_for_pre_decomposition_pipeline(
        exported_program
    )

    cast_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == torch.ops.aten.to.dtype
    ]
    assert len(cast_nodes) == 1
    assert cast_nodes[0].args[1] == torch.int64
    assert cast_nodes[0].meta["val"].dtype == torch.int64
    actual = result.graph_module(test_input)[0]
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected)


##########################################################
## Test argmax/argmin int64 output -> int32 cast       ##
##########################################################


@pytest.mark.parametrize(
    "arg_op, aten_op_str",
    [
        (torch.argmax, "torch.ops.aten.argmax.default"),
        (torch.argmin, "torch.ops.aten.argmin.default"),
    ],
    ids=["argmax", "argmin"],
)
def test_convert_int64_output_ops_to_int32_tosa_FP_insert_cast(arg_op, aten_op_str):
    class ArgOpModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (10 * arg_op(x, dim=-1) + 10) + 1.5

    pipeline = TosaPipelineFP[input_t1](
        ArgOpModel(),
        (torch.randint(0, 10, (2, 4, 6, 8)),),
        aten_op=[aten_op_str, "torch.ops.aten.mul.Tensor", "torch.ops.aten.add.Tensor"],
        exir_op=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
        ],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.run()


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
def test_arg_op_safe_edge_scalar_constant_is_cast_to_int32(arg_op):
    class SafeScalarArithmetic(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return arg_op(x, dim=1) * 10

    module = SafeScalarArithmetic()
    test_input = torch.randn(2, 8)
    exported_program = to_edge(
        torch.export.export(module, (test_input,)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    mul = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.mul.Tensor
    )
    assert mul.args[0].meta["val"].dtype == torch.int32
    assert mul.args[1].meta["val"].dtype == torch.int32

    actual = result.graph_module(torch.tensor(10), test_input)[0]
    expected = module(test_input)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_arg_op_unsafe_arithmetic_stays_int64(arg_op, use_edge_ops: bool):
    class UnsafeArithmetic(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            indices = arg_op(x, dim=1).unsqueeze(-1)
            return indices * indices, indices

    module = UnsafeArithmetic()
    test_input = torch.zeros(1, 50001)
    test_input[0, -1] = 1 if arg_op is torch.argmax else -1
    exported_program = torch.export.export(module, (test_input,))
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    mul_targets = {
        torch.ops.aten.mul.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    }
    relay_targets = {
        torch.ops.aten.unsqueeze.default,
        exir_ops.edge.aten.unsqueeze_copy.default,
    }
    mul = next(
        node for node in result.graph_module.graph.nodes if node.target in mul_targets
    )
    relay = next(
        node for node in result.graph_module.graph.nodes if node.target in relay_targets
    )
    assert mul.args[0].meta["val"].dtype == torch.int64
    assert mul.args[1].meta["val"].dtype == torch.int64
    assert relay.args[0].meta["val"].dtype == torch.int32

    expected = module(test_input)
    actual = result.graph_module(test_input)
    assert actual[0].item() == 2_500_000_000
    assert actual[1].dtype == torch.int64
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
def test_arg_op_direct_output_is_unchanged(arg_op):
    class DirectOutput(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return arg_op(x, dim=1)

    exported_program = torch.export.export(DirectOutput(), (torch.randn(2, 8),))
    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    assert not result.modified
    assert result.graph_module(torch.randn(2, 8))[0].dtype == torch.int64


def test_arg_op_ignores_unrelated_node_without_value_metadata():
    from torch._subclasses import FakeTensorMode

    graph = Graph()
    with FakeTensorMode():
        fake_input = torch.empty(2, 8)
        fake_output = torch.empty(2, dtype=torch.int64)
    x = graph.placeholder("x")
    x.meta["val"] = fake_input
    argmax = graph.call_function(torch.ops.aten.argmax.default, (x, 1))
    argmax.meta["val"] = fake_output
    graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        (x,),
        {"dtype": torch.int64},
    )
    graph.output(argmax)
    graph_module = GraphModule(torch.nn.Module(), graph)

    result = ConvertInt64OutputOpsToInt32Pass().call(graph_module)

    assert not result.modified


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
def test_arg_op_bounded_symbolic_dimension_remains_dynamic(arg_op):
    class SymbolicArgOp(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return arg_op(x, dim=1).unsqueeze(-1)

    module = SymbolicArgOp()
    example_input = torch.randn(2, 8)
    exported_program = torch.export.export(
        module,
        (example_input,),
        dynamic_shapes={"x": {1: torch.export.Dim("arg_dim", min=3, max=32)}},
    )

    result = ArmPassManager(
        TosaCompileSpec("TOSA-1.0+FP")
    ).transform_for_pre_decomposition_pipeline(exported_program)

    int32_casts = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == torch.ops.dim_order_ops._to_dim_order_copy.default
        and node.kwargs["dtype"] == torch.int32
    ]
    assert len(int32_casts) == 1

    runtime_input = torch.randn(2, 16)
    torch.testing.assert_close(result.module()(runtime_input), module(runtime_input))


@pytest.mark.parametrize(
    "arg_op",
    [torch.argmax, torch.argmin],
    ids=["argmax", "argmin"],
)
def test_arg_op_unbounded_symbolic_dimension_skips_conversion(arg_op):
    class SymbolicArgOp(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return arg_op(x, dim=1).unsqueeze(-1)

    module = SymbolicArgOp()
    example_input = torch.randn(2, 8)
    exported_program = torch.export.export(
        module,
        (example_input,),
        dynamic_shapes={"x": {1: torch.export.Dim("arg_dim", min=3)}},
    )

    result = ArmPassManager(
        TosaCompileSpec("TOSA-1.0+FP")
    ).transform_for_pre_decomposition_pipeline(exported_program)

    assert not any(
        node.target == torch.ops.dim_order_ops._to_dim_order_copy.default
        for node in result.graph_module.graph.nodes
    )

    runtime_input = torch.randn(2, 16)
    torch.testing.assert_close(result.module()(runtime_input), module(runtime_input))


@pytest.mark.parametrize(
    "arithmetic_op, target",
    [
        (torch.add, torch.ops.aten.add.Tensor),
        (torch.sub, torch.ops.aten.sub.Tensor),
    ],
    ids=["add", "sub"],
)
def test_arg_op_oversized_alpha_keeps_arithmetic_inputs_int64(arithmetic_op, target):
    class OversizedAlpha(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            indices = torch.argmax(x, dim=1)
            return arithmetic_op(
                indices,
                torch.full_like(indices, 0),
                alpha=2**31,
            )

    module = OversizedAlpha()
    test_input = torch.randn(2, 8)
    exported_program = torch.export.export(module, (test_input,))

    result = ArmPassManager(
        TosaCompileSpec("TOSA-1.0+FP")
    ).transform_for_pre_decomposition_pipeline(exported_program)

    arithmetic = next(
        node for node in result.graph_module.graph.nodes if node.target == target
    )
    assert all(node.meta["val"].dtype == torch.int64 for node in arithmetic.args[:2])
    torch.testing.assert_close(result.module()(test_input), module(test_input))


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_explicit_int64_full_like_does_not_propagate_int32_range(
    use_edge_ops: bool,
):
    class ExplicitInt64FullLike(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            indices = torch.argmax(x, dim=1)
            explicit_int64 = torch.full_like(indices, 1, dtype=torch.int64)
            return explicit_int64 + indices

    module = ExplicitInt64FullLike()
    test_input = torch.randn(2, 8)
    exported_program = torch.export.export(module, (test_input,))
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    full_like_targets = {
        torch.ops.aten.full_like.default,
        exir_ops.edge.aten.full_like.default,
    }
    full_like = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target in full_like_targets
    )
    assert not result.modified
    assert full_like.args[0].meta["val"].dtype == torch.int64
    assert not _cast_nodes(result.graph_module)
    torch.testing.assert_close(result.graph_module(test_input)[0], module(test_input))


##########################################################
## Test topk indices int64 output -> safe int32 paths     ##
##########################################################


class TopKValuesOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.topk(x, 3, dim=1).values + 1


class TopKIndicesOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.topk(x, 3, dim=1).indices


class TopKFullLikeFloat(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.topk(x, 3, dim=1).indices
        return torch.full_like(indices, 1.5, dtype=torch.float32)


class TopKMultipleIndexConsumers(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        values, indices = torch.topk(x, 3, dim=1)
        return values, indices.unsqueeze(-1), torch.remainder(indices, 2)


class TopKUnsafeArithmetic(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        indices = torch.topk(x, 1, dim=1).indices.unsqueeze(-1)
        return indices * indices, indices


class TopKMixedDtypeCat(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        indices = torch.topk(x, 3, dim=1).indices
        return torch.cat((indices, y), dim=1) + 1


class TopKGather(torch.nn.Module):
    def forward(self, scores: torch.Tensor, features: torch.Tensor):
        indices = torch.topk(scores, 3, dim=1).indices
        expanded = indices.unsqueeze(-1).expand(-1, -1, 2)
        return torch.gather(features, dim=1, index=expanded)


class TopKSymbolicSafeConsumer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.topk(x, 3, dim=1).indices
        return indices.unsqueeze(-1)


def _export_topk_graph(module: torch.nn.Module, use_edge_ops: bool) -> GraphModule:
    exported_program = torch.export.export(module, (torch.randn(2, 8),))
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()
    return exported_program.graph_module


def _export_symbolic_topk_graph(
    module: torch.nn.Module,
    dynamic_dim: type,
) -> GraphModule:
    return torch.export.export(
        module,
        (torch.randn(2, 8),),
        dynamic_shapes={"x": {1: dynamic_dim}},
    ).graph_module


def _cast_nodes(graph_module: GraphModule):
    cast_targets = {
        torch.ops.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target in cast_targets
    ]


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_topk_values_only_is_unchanged(use_edge_ops: bool):
    result = ConvertInt64OutputOpsToInt32Pass().call(
        _export_topk_graph(TopKValuesOnly(), use_edge_ops)
    )

    assert not result.modified


def test_live_supported_node_without_value_metadata_raises():
    graph = Graph()
    x = graph.placeholder("x")
    cast = graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        (x,),
        {"dtype": torch.int64},
    )
    graph.output(cast)
    graph_module = GraphModule(torch.nn.Module(), graph)

    with pytest.raises(KeyError, match="val"):
        ConvertInt64OutputOpsToInt32Pass().call(graph_module)


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_topk_indices_only_preserves_output_dtype(use_edge_ops: bool):
    result = ConvertInt64OutputOpsToInt32Pass().call(
        _export_topk_graph(TopKIndicesOnly(), use_edge_ops)
    )

    assert not result.modified
    assert not _cast_nodes(result.graph_module)
    assert result.graph_module(torch.randn(2, 8))[0].dtype == torch.int64


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_topk_full_like_explicit_float_dtype_is_terminal(use_edge_ops: bool):
    module = TopKFullLikeFloat()
    test_input = torch.randn(2, 8)
    result = ConvertInt64OutputOpsToInt32Pass().call(
        _export_topk_graph(module, use_edge_ops)
    )

    full_like_targets = {
        torch.ops.aten.full_like.default,
        exir_ops.edge.aten.full_like.default,
    }
    full_like = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target in full_like_targets
    )
    casts = _cast_nodes(result.graph_module)
    assert result.modified
    assert full_like.args[0].meta["val"].dtype == torch.int32
    assert full_like.meta["val"].dtype == torch.float32
    assert not any(node.kwargs["dtype"] == torch.int64 for node in casts)

    expected = module(test_input)
    actual = result.graph_module(test_input)[0]
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)


def test_topk_symbolic_indices_output_is_unchanged():
    graph_module = _export_symbolic_topk_graph(
        TopKIndicesOnly(),
        torch.export.Dim("topk_dim", min=3),
    )
    graph_before = str(graph_module.graph)

    result = ConvertInt64OutputOpsToInt32Pass().call(graph_module)

    assert not result.modified
    assert str(result.graph_module.graph) == graph_before
    assert not _cast_nodes(result.graph_module)


def test_topk_bounded_symbolic_dimension_uses_int32_path():
    graph_module = _export_symbolic_topk_graph(
        TopKSymbolicSafeConsumer(),
        torch.export.Dim("topk_dim", min=3, max=32),
    )

    result = ConvertInt64OutputOpsToInt32Pass().call(graph_module)

    int32_casts = [
        node
        for node in _cast_nodes(result.graph_module)
        if node.kwargs["dtype"] == torch.int32
    ]
    assert result.modified
    assert len(int32_casts) == 1


def test_topk_unbounded_symbolic_dimension_skips_conversion():
    graph_module = _export_symbolic_topk_graph(
        TopKSymbolicSafeConsumer(),
        torch.export.Dim("topk_dim", min=3),
    )
    graph_before = str(graph_module.graph)

    result = ConvertInt64OutputOpsToInt32Pass().call(graph_module)

    assert not result.modified
    assert str(result.graph_module.graph) == graph_before
    assert not _cast_nodes(result.graph_module)


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_topk_multiple_index_consumers_use_safe_int32_path(use_edge_ops: bool):
    module = TopKMultipleIndexConsumers()
    result = ConvertInt64OutputOpsToInt32Pass().call(
        _export_topk_graph(module, use_edge_ops)
    )

    casts = _cast_nodes(result.graph_module)
    int32_casts = [node for node in casts if node.kwargs["dtype"] == torch.int32]
    int64_casts = [node for node in casts if node.kwargs["dtype"] == torch.int64]
    assert len(int32_casts) == 1
    assert len(int64_casts) == 1
    assert len(int32_casts[0].users) == 1

    test_input = torch.randn(2, 8)
    expected = module(test_input)
    actual = result.graph_module(test_input)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        assert actual_output.dtype == expected_output.dtype
        torch.testing.assert_close(actual_output, expected_output)


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_topk_unsafe_arithmetic_stays_int64(use_edge_ops: bool):
    module = TopKUnsafeArithmetic()
    test_input = torch.zeros(1, 50001)
    test_input[0, -1] = 1
    exported_program = torch.export.export(module, (test_input,))
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    mul_targets = {
        torch.ops.aten.mul.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    }
    mul = next(
        node for node in result.graph_module.graph.nodes if node.target in mul_targets
    )
    assert mul.args[0].meta["val"].dtype == torch.int64
    assert mul.args[1].meta["val"].dtype == torch.int64

    expected = module(test_input)
    actual = result.graph_module(test_input)
    assert actual[0].item() == 2_500_000_000
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_topk_mixed_dtype_cat_stays_int64(use_edge_ops: bool):
    module = TopKMixedDtypeCat()
    x = torch.randn(2, 8)
    y = torch.full((2, 1), torch.iinfo(torch.int32).max, dtype=torch.int32)
    exported_program = torch.export.export(module, (x, y))
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    assert not result.modified
    assert not _cast_nodes(result.graph_module)
    actual = exported_program.module()(x, y)
    assert actual.dtype == torch.int64
    assert actual[:, -1].tolist() == [torch.iinfo(torch.int32).max + 1] * 2
    torch.testing.assert_close(actual, module(x, y))


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
@pytest.mark.parametrize(
    "feature_channels",
    [2, 3],
    ids=["supported", "unsupported_trailing_dim"],
)
def test_topk_gather_index_dtype_at_portable_boundary(
    use_edge_ops: bool,
    feature_channels: int,
):
    module = TopKGather()
    inputs = (torch.randn(2, 8), torch.randn(2, 8, feature_channels))
    exported_program = torch.export.export(module, inputs)
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = ConvertInt64OutputOpsToInt32Pass().call(exported_program.graph_module)

    gather_targets = {
        torch.ops.aten.gather.default,
        exir_ops.edge.aten.gather.default,
    }
    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target in gather_targets
    )
    assert gather.args[2].meta["val"].dtype == torch.int64
    boundary = gather.args[2]
    assert boundary in _cast_nodes(result.graph_module)
    assert boundary.args[0].meta["val"].dtype == torch.int32

    expected = module(*inputs)
    actual = result.graph_module(*inputs)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
@pytest.mark.parametrize(
    "feature_channels",
    [2, 3],
    ids=["supported", "unsupported_trailing_dim"],
)
def test_prepare_gather_index_dtype(
    use_edge_ops: bool,
    feature_channels: int,
):
    module = TopKGather()
    inputs = (torch.randn(2, 8), torch.randn(2, 8, feature_channels))
    exported_program = torch.export.export(module, inputs)
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    bounded_result = ConvertInt64OutputOpsToInt32Pass().call(
        exported_program.graph_module
    )
    result = PrepareGatherIndicesPass(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ).call(bounded_result.graph_module)

    gather_targets = {
        torch.ops.aten.gather.default,
        exir_ops.edge.aten.gather.default,
    }
    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target in gather_targets
    )
    expected_index_dtype = (
        torch.int32 if use_edge_ops and feature_channels == 2 else torch.int64
    )
    assert gather.args[2].meta["val"].dtype == expected_index_dtype
    assert result.modified == (expected_index_dtype == torch.int32)

    expected = module(*inputs)
    actual = result.graph_module(*inputs)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "values_dtype,tosa_spec,supported",
    [
        (torch.int32, "TOSA-1.0+INT", True),
        (torch.int32, "TOSA-1.0+FP", False),
        (torch.float32, "TOSA-1.0+FP", True),
        (torch.float32, "TOSA-1.0+INT", True),
        (torch.bfloat16, "TOSA-1.0+FP+bf16", True),
        (torch.bfloat16, "TOSA-1.0+FP", False),
        (torch.float8_e4m3fn, "TOSA-1.0+FP+fp8e4m3", True),
        (torch.float8_e4m3fn, "TOSA-1.0+FP", False),
        (torch.float8_e5m2, "TOSA-1.0+FP+fp8e5m2", True),
        (torch.float8_e5m2, "TOSA-1.0+FP", False),
    ],
    ids=[
        "int32-int",
        "int32-fp",
        "fp32-fp",
        "fp32-int",
        "bf16-enabled",
        "bf16-disabled",
        "fp8e4m3-enabled",
        "fp8e4m3-disabled",
        "fp8e5m2-enabled",
        "fp8e5m2-disabled",
    ],
)
def test_prepare_gather_index_dtype_for_tosa_capabilities(
    values_dtype: torch.dtype,
    tosa_spec: str,
    supported: bool,
):
    inputs = (
        torch.randn(2, 8),
        torch.randn(2, 8, 2).to(values_dtype),
    )
    exported_program = to_edge(
        torch.export.export(TopKGather(), inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    graph_module = (
        ConvertInt64OutputOpsToInt32Pass()
        .call(exported_program.graph_module)
        .graph_module
    )
    gather = next(
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    )
    boundary = gather.args[2]
    assert boundary.meta["val"].dtype == torch.int64

    result = PrepareGatherIndicesPass(
        TosaSpecification.create_from_string(tosa_spec)
    ).call(graph_module)

    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    )
    assert result.modified == supported
    assert gather.args[2].meta["val"].dtype == (
        torch.int32 if supported else torch.int64
    )
    assert (boundary not in result.graph_module.graph.nodes) == supported


def test_symbolic_gather_shape_mismatch_rejects_boundary():
    scores_batch = torch.export.Dim("scores_batch", min=1, max=8)
    features_batch = torch.export.Dim("features_batch", min=1, max=8)
    exported_program = torch.export.export(
        TopKGather(),
        (
            torch.randn(2, 8),
            torch.randn(3, 8, 2),
        ),
        dynamic_shapes={
            "scores": {0: scores_batch},
            "features": {0: features_batch},
        },
    )
    exported_program = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    graph_module = (
        ConvertInt64OutputOpsToInt32Pass()
        .call(exported_program.graph_module)
        .graph_module
    )

    gather = next(
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    )
    boundary = gather.args[2]
    assert isinstance(gather.args[0].meta["val"].shape[0], torch.SymInt)
    assert isinstance(boundary.meta["val"].shape[0], torch.SymInt)
    assert not is_safe_int32_to_int64_gather_boundary(boundary)


@pytest.mark.parametrize("use_edge_ops", [False, True], ids=["aten", "edge"])
def test_prepare_scalar_gather_is_unchanged(use_edge_ops: bool):
    class ScalarGather(torch.nn.Module):
        def forward(self, values: torch.Tensor, indices: torch.Tensor):
            return torch.gather(values, 0, indices)

    module = ScalarGather()
    inputs = (torch.tensor(1.0), torch.tensor(0))
    exported_program = torch.export.export(module, inputs)
    if use_edge_ops:
        exported_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).exported_program()

    result = PrepareGatherIndicesPass(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ).call(exported_program.graph_module)

    assert not result.modified
    gather_targets = {
        torch.ops.aten.gather.default,
        exir_ops.edge.aten.gather.default,
    }
    assert any(
        node.target in gather_targets for node in result.graph_module.graph.nodes
    )
    torch.testing.assert_close(result.graph_module(*inputs)[0], module(*inputs))


def test_prepare_gather_does_not_narrow_arbitrary_int64_indices():
    class DirectGather(torch.nn.Module):
        def forward(self, values: torch.Tensor, indices: torch.Tensor):
            return torch.gather(values, 1, indices)

    module = DirectGather()
    inputs = (
        torch.randn(2, 8, 2),
        torch.randint(0, 8, (2, 3, 2), dtype=torch.int64),
    )
    exported_program = torch.export.export(module, inputs)

    result = PrepareGatherIndicesPass(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ).call(exported_program.graph_module)

    assert not result.modified
    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == torch.ops.aten.gather.default
    )
    assert gather.args[2].meta["val"].dtype == torch.int64
    torch.testing.assert_close(result.graph_module(*inputs)[0], module(*inputs))


def test_prepare_gather_preserves_layout_changing_boundary():
    class LayoutChangingGather(torch.nn.Module):
        def forward(self, values: torch.Tensor, indices: torch.Tensor):
            indices = indices.to(
                dtype=torch.int64, memory_format=torch.contiguous_format
            )
            return torch.gather(values, 1, indices)

    module = LayoutChangingGather()
    inputs = (
        torch.randn(2, 8, 2),
        torch.randint(0, 8, (2, 2, 3), dtype=torch.int32).transpose(1, 2),
    )
    exported_program = to_edge(
        torch.export.export(module, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    boundary = next(
        node
        for node in exported_program.graph.nodes
        if node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    )
    source = boundary.args[0]
    assert source.meta["val"].dim_order() != boundary.meta["val"].dim_order()

    result = PrepareGatherIndicesPass(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ).call(exported_program.graph_module)

    assert not result.modified
    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    )
    assert gather.args[2] is boundary
    torch.testing.assert_close(result.graph_module(*inputs)[0], module(*inputs))


##############################################################
## Test gather preparation for Arm target capabilities       ##
##############################################################
@pytest.mark.parametrize(
    "compile_spec",
    [
        TosaCompileSpec("TOSA-1.0+FP"),
        VgfCompileSpec("TOSA-1.0+FP"),
        EthosUCompileSpec("ethos-u85-128"),
        EthosUCompileSpec("ethos-u55-128"),
    ],
    ids=["tosa", "vgf", "u85", "u55"],
)
def test_pre_decomposition_preserves_gather_index_dtype(compile_spec):
    module = TopKGather()
    inputs = (torch.randn(2, 8), torch.randn(2, 8, 2))
    exported_program = torch.export.export(module, inputs)

    result = ArmPassManager(compile_spec).transform_for_pre_decomposition_pipeline(
        exported_program
    )

    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == torch.ops.aten.gather.default
    )
    assert gather.args[2].meta["val"].dtype == torch.int64
    boundary = gather.args[2]
    assert boundary in _cast_nodes(result.graph_module)
    assert boundary.args[0].meta["val"].dtype == torch.int32


##############################################################
def test_rejected_gather_keeps_int64_indices():
    module = TopKGather()
    inputs = (torch.randn(2, 8), torch.randn(2, 8, 2))
    compile_spec = TosaCompileSpec("TOSA-1.0+FP")
    partitioner = TOSAPartitioner(
        compile_spec,
        additional_checks=[DontPartition(exir_ops.edge.aten.gather.default)],
    )

    edge_manager = to_edge_transform_and_lower(
        torch.export.export(module, inputs),
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    gather = next(
        node
        for node in edge_manager.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    )
    assert gather.args[2].meta["val"].dtype == torch.int64


def test_rejected_gather_boundary_keeps_gather_portable():
    module = TopKGather()
    inputs = (torch.randn(2, 8), torch.randn(2, 8, 2))
    compile_spec = TosaCompileSpec("TOSA-1.0+FP")
    partitioner = TOSAPartitioner(
        compile_spec,
        additional_checks=[
            DontPartition(exir_ops.edge.dim_order_ops._to_dim_order_copy.default)
        ],
    )

    edge_manager = to_edge_transform_and_lower(
        torch.export.export(module, inputs),
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    gather = next(
        node
        for node in edge_manager.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    )
    boundary = gather.args[2]
    assert boundary.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    assert boundary.meta["val"].dtype == torch.int64


def test_supported_gather_is_delegated():
    module = TopKGather()
    inputs = (torch.randn(2, 8), torch.randn(2, 8, 2))
    compile_spec = TosaCompileSpec("TOSA-1.0+FP")

    edge_manager = to_edge_transform_and_lower(
        torch.export.export(module, inputs),
        partitioner=[TOSAPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    assert all(
        node.target != exir_ops.edge.aten.gather.default
        for node in edge_manager.exported_program().graph.nodes
    )


def test_shared_gather_boundary_with_rejected_gather_stays_portable():
    class SharedGatherBoundary(torch.nn.Module):
        def forward(self, scores, first_features, second_features):
            scores = scores + 1
            indices = torch.topk(scores, 3, dim=1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, 2)
            first = torch.gather(first_features, 1, indices) + 1
            second = torch.gather(second_features, 1, indices) + 1
            return (
                first,
                second,
            )

    inputs = (
        torch.randn(2, 8),
        torch.randn(2, 8, 2),
        torch.randn(2, 8, 2),
    )
    edge_manager = to_edge(
        torch.export.export(SharedGatherBoundary(), inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).transform([ConvertInt64OutputOpsToInt32Pass()])
    exported_program = edge_manager.exported_program()
    graph_module = exported_program.graph_module

    gathers = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.gather.default
    ]
    assert len(gathers) == 2
    boundary = gathers[0].args[2]
    assert boundary is gathers[1].args[2]
    assert boundary.meta["val"].dtype == torch.int64

    rejected_gather = gathers[1]
    check = DontPartitionName(rejected_gather.name)
    result = TOSAPartitioner(
        TosaCompileSpec("TOSA-1.0+FP"), additional_checks=[check]
    ).partition(exported_program)

    assert rejected_gather in check.rejected_nodes()
    assert all("delegation_tag" not in gather.meta for gather in gathers)
    assert "delegation_tag" not in boundary.meta
    assert all(gather.args[2] is boundary for gather in gathers)
    assert boundary.meta["val"].dtype == torch.int64

    returned_tags = set(result.partition_tags)
    active_tags = {
        node.meta["delegation_tag"]
        for node in result.tagged_exported_program.graph.nodes
        if "delegation_tag" in node.meta
    }
    assert returned_tags
    assert active_tags == returned_tags


##############################################################
## Test on_overflow range check for bounded index sources   ##
##############################################################

_OVERFLOW_DIM = torch.iinfo(torch.int32).max + 1


def _make_argmax_graph_large_dim() -> GraphModule:
    """Construct a minimal graph with an argmax over a dimension > INT32_MAX.

    Uses FakeTensorMode so no memory is allocated for the large dimension.

    """
    from torch._subclasses import FakeTensorMode

    graph = Graph()
    with FakeTensorMode():
        fake_input = torch.empty(_OVERFLOW_DIM, dtype=torch.float32)
        fake_output = torch.empty((), dtype=torch.int64)
    x = graph.placeholder("x")
    x.meta["val"] = fake_input
    out = graph.call_function(torch.ops.aten.argmax.default, (x, 0))
    out.meta["val"] = fake_output
    graph.output(out)
    return GraphModule(torch.nn.Module(), graph)


def _make_gather_graph_large_dim() -> GraphModule:
    """Construct a gather whose indexable dimension exceeds INT32_MAX."""
    from torch._subclasses import FakeTensorMode

    graph = Graph()
    with FakeTensorMode():
        fake_values = torch.empty((1, _OVERFLOW_DIM), dtype=torch.float32)
        fake_int32_indices = torch.empty((1, 1), dtype=torch.int32)
        fake_indices = torch.empty((1, 1), dtype=torch.int64)
        fake_output = torch.empty((1, 1), dtype=torch.float32)
    values = graph.placeholder("values")
    values.meta["val"] = fake_values
    int32_indices = graph.placeholder("indices")
    int32_indices.meta["val"] = fake_int32_indices
    indices = graph.call_function(
        torch.ops.dim_order_ops._to_dim_order_copy.default,
        (int32_indices,),
        {"dtype": torch.int64},
    )
    indices.meta["val"] = fake_indices
    gather = graph.call_function(torch.ops.aten.gather.default, (values, 1, indices))
    gather.meta["val"] = fake_output
    graph.output(gather)
    return GraphModule(torch.nn.Module(), graph)


def test_prepare_gather_skips_large_indexable_dimension():
    graph_module = _make_gather_graph_large_dim()
    result = PrepareGatherIndicesPass(
        TosaSpecification.create_from_string("TOSA-1.0+FP")
    ).call(graph_module)

    assert not result.modified
    gather = next(
        node
        for node in result.graph_module.graph.nodes
        if node.target == torch.ops.aten.gather.default
    )
    assert gather.args[2].meta["val"].dtype == torch.int64


def _make_topk_graph_large_dim() -> GraphModule:
    """Construct a topk graph indexing a dimension larger than INT32_MAX."""
    from torch._subclasses import FakeTensorMode

    graph = Graph()
    with FakeTensorMode():
        fake_input = torch.empty((1, _OVERFLOW_DIM), dtype=torch.float32)
        fake_values = torch.empty((1, 1), dtype=torch.float32)
        fake_indices = torch.empty((1, 1), dtype=torch.int64)
    x = graph.placeholder("x")
    x.meta["val"] = fake_input
    topk = graph.call_function(torch.ops.aten.topk.default, (x, 1, 1))
    topk.meta["val"] = (fake_values, fake_indices)
    indices = graph.call_function(operator.getitem, (topk, 1))
    indices.meta["val"] = fake_indices
    graph.output(indices)
    return GraphModule(torch.nn.Module(), graph)


def test_on_overflow_raise():
    gm = _make_argmax_graph_large_dim()
    with pytest.raises(RuntimeError, match="cannot be safely cast to int32"):
        ConvertInt64OutputOpsToInt32Pass(on_overflow="raise").call(gm)


def test_topk_on_overflow_raise():
    gm = _make_topk_graph_large_dim()
    with pytest.raises(RuntimeError, match="cannot be safely cast to int32"):
        ConvertInt64OutputOpsToInt32Pass(on_overflow="raise").call(gm)


def test_topk_on_overflow_warn(caplog):
    import logging

    gm = _make_topk_graph_large_dim()
    with caplog.at_level(logging.WARNING):
        result = ConvertInt64OutputOpsToInt32Pass(on_overflow="warn").call(gm)
    assert not result.modified
    assert caplog.messages == [
        "aten.topk.default indexes a dimension with more than 2147483647 "
        "elements; the int64 index cannot be safely cast to int32."
    ]


def test_topk_on_overflow_skip():
    gm = _make_topk_graph_large_dim()
    result = ConvertInt64OutputOpsToInt32Pass(on_overflow="skip").call(gm)
    assert not result.modified


def test_on_overflow_warn(caplog):
    import logging

    gm = _make_argmax_graph_large_dim()
    with caplog.at_level(logging.WARNING):
        result = ConvertInt64OutputOpsToInt32Pass(on_overflow="warn").call(gm)
    assert not result.modified
    assert "cannot be safely cast to int32" in caplog.text


def test_on_overflow_skip():
    gm = _make_argmax_graph_large_dim()
    result = ConvertInt64OutputOpsToInt32Pass(on_overflow="skip").call(gm)
    assert not result.modified


def test_on_overflow_invalid():
    with pytest.raises(ValueError, match="on_overflow must be"):
        ConvertInt64OutputOpsToInt32Pass(on_overflow="blah")
