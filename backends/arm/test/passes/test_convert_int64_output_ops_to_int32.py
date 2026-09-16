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
)
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.exir import EdgeCompileConfig, to_edge
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
