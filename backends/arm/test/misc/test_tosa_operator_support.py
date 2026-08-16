# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    CheckKnownUnsupportedTOSASemantics,
)
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def _fake_tensor(shape: tuple[int, ...], dtype: torch.dtype = torch.float32):
    with FakeTensorMode() as mode:
        return mode.from_tensor(torch.empty(shape, dtype=dtype))


def _placeholder(graph: torch.fx.Graph, name: str, shape, dtype=torch.float32):
    node = graph.placeholder(name)
    node.meta["val"] = _fake_tensor(shape, dtype)
    return node


def _checker() -> CheckKnownUnsupportedTOSASemantics:
    return CheckKnownUnsupportedTOSASemantics(WhyNoPartitionReporter())


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
