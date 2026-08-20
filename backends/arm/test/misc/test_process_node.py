# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
import tosa_serializer as ts
from executorch.backends.arm.process_node import _add_const, process_placeholder
from executorch.backends.arm.tosa.mapping import TosaArg, TosaSpecialDtype
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir import to_edge
from torch._export.utils import is_param
from tosa.TosaGraph import TosaGraph  # type: ignore[import-not-found, import-untyped]
from tosa_serializer.numpy_utils import pack_6bit_array


class Int32BiasModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.tensor([1, -2, 0x12345678], dtype=torch.int32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the int32 parameter live in the exported graph.
        return x + self.bias[0].to(torch.float32)


class CapturingTosaGraph:
    def __init__(self) -> None:
        self.shape = None
        self.dtype = None
        self.values = None
        self.name = None
        self.serialized_bytes = None

    def addConst(self, shape, dtype, values, name):
        self.shape = shape
        self.dtype = dtype
        self.values = np.asarray(values)
        self.name = name
        if dtype == ts.DType.INT48:
            self.serialized_bytes = self._serialize_int48(self.values)

    @staticmethod
    def _serialize_int48(values: np.ndarray) -> list[int]:
        # Simulate a consumer that expects each element pre-normalized to int64
        # before narrowing to little-endian signed 48-bit storage.
        if values.dtype == np.int64:
            packed: list[int] = []
            for value in values.reshape(-1):
                masked = int(value) & ((1 << 48) - 1)
                packed.extend([(masked >> (8 * i)) & 0xFF for i in range(6)])
            return packed

        # Existing buggy path: treat raw bytes as already packed int48 stream.
        raw = values.view(np.uint8).reshape(-1).tolist()
        remainder = len(raw) % 6
        if remainder:
            raw.extend([0] * (6 - remainder))
        return raw


def _expected_int48_bytes(values: torch.Tensor) -> list[int]:
    packed: list[int] = []
    for value in values.tolist():
        masked = int(value) & ((1 << 48) - 1)
        packed.extend([(masked >> (8 * i)) & 0xFF for i in range(6)])
    return packed


def test_process_placeholder_int48_normalizes_int32_const_values() -> None:
    module = Int32BiasModule().eval()
    exported_program = torch.export.export(module, (torch.randn(1),))
    edge_program = to_edge(exported_program).exported_program()

    param_node = next(
        node
        for node in edge_program.graph.nodes
        if node.op == "placeholder" and is_param(edge_program, node)
    )
    param_node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48

    tosa_graph = CapturingTosaGraph()
    process_placeholder(
        param_node,
        tosa_graph,
        edge_program,
        containing_graph_module=None,
        tosa_spec=TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
    )

    assert tosa_graph.dtype == ts.DType.INT48
    assert tosa_graph.values is not None
    assert tosa_graph.values.dtype == np.int64
    assert tosa_graph.serialized_bytes == _expected_int48_bytes(module.bias)


def test_add_const_fp4_in_packed_storage() -> None:
    packed_values = np.array([0xDE, 0xFE, 0x6D, 0x55], dtype=np.uint8).reshape(
        1,
        1,
        4,
    )
    tosa_arg = cast(
        TosaArg,
        SimpleNamespace(dtype=ts.DType.FP4E2M1, shape=(1, 1, 8)),
    )
    tosa_graph = ts.TosaSerializer()

    _add_const(tosa_graph, packed_values, tosa_arg, name="fp4_weight")

    graph = TosaGraph.GetRootAs(bytes(tosa_graph.serialize()), 0)
    block = graph.Regions(0).Blocks(0)
    tensors = {
        block.Tensors(index).Name().decode(): block.Tensors(index)
        for index in range(block.TensorsLength())
    }
    tensor = tensors["fp4_weight"]

    assert tensor.Type() == ts.DType.FP4E2M1
    assert [tensor.Shape(index) for index in range(tensor.ShapeLength())] == [1, 1, 8]
    assert [tensor.Data(index) for index in range(tensor.DataLength())] == [
        0xDE,
        0xFE,
        0x6D,
        0x55,
    ]


def _test_add_const_fp6_in_packed_storage(dtype: int) -> None:
    values = np.arange(32, dtype=np.uint8).reshape(1, 1, 32)

    tosa_arg = cast(
        TosaArg,
        SimpleNamespace(dtype=dtype, shape=(1, 1, 32)),
    )
    tosa_graph = ts.TosaSerializer()

    _add_const(tosa_graph, values, tosa_arg, name="fp6_weight")

    graph = TosaGraph.GetRootAs(bytes(tosa_graph.serialize()), 0)
    block = graph.Regions(0).Blocks(0)
    tensors = {
        block.Tensors(index).Name().decode(): block.Tensors(index)
        for index in range(block.TensorsLength())
    }
    tensor = tensors["fp6_weight"]

    assert tensor.Type() == dtype
    assert [tensor.Shape(index) for index in range(tensor.ShapeLength())] == [
        1,
        1,
        32,
    ]
    assert tensor.DataLength() == 24
    assert [tensor.Data(index) for index in range(tensor.DataLength())] == (
        pack_6bit_array(values).reshape(-1).tolist()
    )


def test_add_const_fp6e2m3_in_packed_storage() -> None:
    _test_add_const_fp6_in_packed_storage(ts.DType.FP6E2M3)


def test_add_const_fp6e3m2_in_packed_storage() -> None:
    _test_add_const_fp6_in_packed_storage(ts.DType.FP6E3M2)
