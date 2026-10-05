# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import pytest
import torch
from executorch.backends.arm._passes.decompose_topk_pass import get_static_topk_config
from executorch.backends.arm.operator_support.index_tensor_support import (
    IndexTensorSupported,
)
from executorch.backends.arm.operator_support.topk_support import TopKSupported
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    CheckFPComparisonInputs,
    CheckKnownUnsupportedTOSASemantics,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _fake_tensor(shape: tuple[int, ...], dtype: torch.dtype = torch.float32):
    with FakeTensorMode() as mode:
        return mode.from_tensor(torch.empty(shape, dtype=dtype))


def _placeholder(graph: torch.fx.Graph, name: str, shape, dtype=torch.float32):
    node = graph.placeholder(name)
    node.meta["val"] = _fake_tensor(shape, dtype)
    return node


def _checker() -> CheckKnownUnsupportedTOSASemantics:
    return CheckKnownUnsupportedTOSASemantics(WhyNoPartitionReporter())


def _fp_comparison_checker() -> CheckFPComparisonInputs:
    return CheckFPComparisonInputs(WhyNoPartitionReporter())


def _topk_node(shape=(2, 8), dtype=torch.float32, args=(2,), kwargs=None):
    graph = torch.fx.Graph()
    scores = _placeholder(graph, "scores", shape, dtype)
    return graph.call_function(
        exir_ops.edge.aten.topk.default, (scores, *args), kwargs or {}
    )


@pytest.mark.parametrize(
    "target",
    (
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.lt.Tensor,
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (torch.bool, torch.uint8, torch.int32, torch.int64),
)
def test_fp_comparison_rejects_unsupported_inputs(target, dtype) -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (3, 4), dtype)
    y = _placeholder(graph, "y", (3, 4), dtype)
    node = graph.call_function(target, (x, y))
    node.meta["val"] = _fake_tensor((3, 4), torch.bool)

    assert not _fp_comparison_checker().is_node_supported({}, node)


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.float32, torch.bfloat16, torch.int8, torch.int16),
)
def test_fp_comparison_accepts_supported_inputs(dtype) -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (3, 4), dtype)
    y = _placeholder(graph, "y", (3, 4), dtype)
    node = graph.call_function(exir_ops.edge.aten.eq.Tensor, (x, y))
    node.meta["val"] = _fake_tensor((3, 4), torch.bool)

    assert _fp_comparison_checker().is_node_supported({}, node)


def test_rejects_argmax_without_int32_cast_user() -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (3, 4))
    node = graph.call_function(exir_ops.edge.aten.argmax.default, (x, 1, False))
    node.meta["val"] = _fake_tensor((3,), torch.int64)

    assert not _checker().is_node_supported({}, node)


def test_accepts_argmax_with_int32_cast_user() -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (3, 4))
    node = graph.call_function(exir_ops.edge.aten.argmax.default, (x, 1, False))
    node.meta["val"] = _fake_tensor((3,), torch.int64)
    cast = graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        (node,),
        {"dtype": torch.int32},
    )
    cast.meta["val"] = _fake_tensor((3,), torch.int32)

    assert _checker().is_node_supported({}, node)


def test_accepts_argmax_with_all_users_casting_to_int32() -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (3, 4))
    node = graph.call_function(exir_ops.edge.aten.argmax.default, (x, 1, False))
    node.meta["val"] = _fake_tensor((3,), torch.int64)
    cast = graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        (node,),
        {"dtype": torch.int32},
    )
    cast.meta["val"] = _fake_tensor((3,), torch.int32)
    cast_2 = graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        (node,),
        {"dtype": torch.int32},
    )
    cast_2.meta["val"] = _fake_tensor((3,), torch.int32)

    assert _checker().is_node_supported({}, node)


def test_rejects_argmax_with_mixed_int32_cast_and_raw_user() -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (3, 4))
    node = graph.call_function(exir_ops.edge.aten.argmax.default, (x, 1, False))
    node.meta["val"] = _fake_tensor((3,), torch.int64)
    cast = graph.call_function(
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        (node,),
        {"dtype": torch.int32},
    )
    cast.meta["val"] = _fake_tensor((3,), torch.int32)
    raw_user = graph.call_function(torch.ops.aten.clone.default, (node,))
    raw_user.meta["val"] = _fake_tensor((3,), torch.int64)

    assert not _checker().is_node_supported({}, node)


@pytest.mark.parametrize("dtype", (torch.bool, torch.uint8))
def test_rejects_index_tensor_mask(dtype: torch.dtype) -> None:
    graph = torch.fx.Graph()
    x = _placeholder(graph, "x", (5, 2, 3))
    index = _placeholder(graph, "index", (5,), dtype)
    node = graph.call_function(exir_ops.edge.aten.index.Tensor, (x, [index]))
    node.meta["val"] = _fake_tensor((2, 2, 3))

    checker = IndexTensorSupported(
        TosaSpecification.create_from_string("TOSA-1.0+INT+u55"),
        WhyNoPartitionReporter(),
    )

    assert not checker.is_node_supported({}, node)


@pytest.mark.parametrize(
    "dynamic_values, dynamic_index", [(True, False), (False, True)]
)
def test_rejects_index_tensor_data_dependent_shapes(
    dynamic_values: bool, dynamic_index: bool
) -> None:
    shape_env = ShapeEnv()
    count = shape_env.create_unbacked_symint()
    shape_env.constrain_symbol_range(count.node.expr, compiler_min=0, compiler_max=512)
    graph = torch.fx.Graph()
    values = graph.placeholder("values")
    index = graph.placeholder("index")
    with FakeTensorMode(shape_env=shape_env):
        values.meta["val"] = torch.empty((count if dynamic_values else 8, 4))
        index.meta["val"] = torch.empty(
            (count if dynamic_index else 3,), dtype=torch.int32
        )
        node = graph.call_function(exir_ops.edge.aten.index.Tensor, (values, [index]))
        node.meta["val"] = torch.empty((count if dynamic_index else 3, 4))

    reporter = WhyNoPartitionReporter()
    checker = IndexTensorSupported(
        TosaSpecification.create_from_string("TOSA-1.1+FP+INT+shape"), reporter
    )

    assert not checker.is_node_supported({}, node)
    assert "Symbolic value or index shapes" in reporter.get_table_report()
    assert not shape_env.guards


@pytest.mark.parametrize(
    "shape,dtype,args,kwargs,reason",
    [
        ((8,), torch.float32, (1,), {}, "rank-2"),
        ((0, 8), torch.float32, (1,), {}, "positive static"),
        ((2, 0), torch.float32, (1,), {}, "positive static"),
        ((1, 2147483648), torch.float32, (1,), {}, "int32"),
        ((2, 8), torch.float32, (0,), {}, "constant"),
        ((2, 2), torch.float32, (3,), {}, "constant"),
        ((2, 8), torch.float32, (True,), {}, "constant"),
        ((2, 8), torch.float32, (2, 0), {}, "dim"),
        ((2, 8), torch.float32, (2, 3), {}, "dim"),
        ((2, 8), torch.float32, (2, True), {}, "dim"),
        ((2, 8), torch.float32, (2, -1, False), {}, "largest"),
        ((2, 8), torch.float32, (2,), {"sorted": False}, "sorted"),
        ((2, 8), torch.bfloat16, (1,), {}, "FP16 or FP32"),
        ((2, 8), torch.float64, (1,), {}, "FP16 or FP32"),
        ((2, 8), torch.int32, (1,), {}, "FP16 or FP32"),
    ],
)
def test_topk_unsupported_metadata(shape, dtype, args, kwargs, reason) -> None:
    node = _topk_node(shape, dtype, args, kwargs)
    spec = TosaSpecification.create_from_string("TOSA-1.0+FP+INT+bf16")
    reporter = WhyNoPartitionReporter()
    assert not TopKSupported(spec, reporter).is_node_supported({}, node)
    assert reason in reporter.get_table_report()


def test_topk_rejects_runtime_k() -> None:
    node = _topk_node()
    with node.graph.inserting_before(node):
        k = node.graph.placeholder("k")
    node.args = (node.args[0], k)
    config, reason = get_static_topk_config(
        node, TosaSpecification.create_from_string("TOSA-1.0+FP+INT")
    )
    assert config is None
    assert reason is not None and "constant" in reason


@pytest.mark.parametrize("index", [-2, -1], ids=["values", "indices"])
def test_topk_rejects_negative_tuple_indices(index) -> None:
    topk = _topk_node()
    extraction = topk.graph.call_function(operator.getitem, (topk, index))
    topk.graph.output(extraction)
    spec = TosaSpecification.create_from_string("TOSA-1.0+FP+INT")
    reporter = WhyNoPartitionReporter()
    assert not TopKSupported(spec, reporter).is_node_supported({}, topk)
    assert "canonical" in reporter.get_table_report()


@pytest.mark.parametrize("dim", [-1, 1])
def test_topk_keyword_arguments(dim) -> None:
    node = _topk_node(args=(), kwargs={"k": 4, "dim": dim})
    config, reason = get_static_topk_config(
        node, TosaSpecification.create_from_string("TOSA-1.0+FP+INT")
    )
    assert config is not None and config.k == 4
    assert reason is None
