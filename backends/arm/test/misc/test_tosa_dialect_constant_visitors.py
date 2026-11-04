# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import numpy as np
import pytest
import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm._passes.tosa_lowering_utils import create_const
from executorch.backends.arm.operators.node_visitor import get_node_visitors
from executorch.backends.arm.process_node import process_call_function, process_inputs
from executorch.backends.arm.tosa.constant_pool import TosaSerializerWithConstantPool
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph
from tosa.TosaGraph import TosaGraph  # type: ignore[import-untyped]


@pytest.mark.parametrize(
    "target,dtype,values",
    [
        (exir_ops.backend.tosa.MUL.default, torch.float32, [0]),
        (exir_ops.backend.tosa.MUL.default, torch.int32, [17]),
        (exir_ops.backend.tosa.NEGATE.default, torch.float32, [0, 0]),
        (exir_ops.backend.tosa.NEGATE.default, torch.int8, [3, -4]),
        (exir_ops.backend.tosa.PAD.default, torch.float32, [2.5]),
        (exir_ops.backend.tosa.PAD.default, torch.int8, [7]),
    ],
)
def test_visitors_recreate_constants_without_serializing_dialect_const(
    target, dtype, values
):
    spec = TosaSpecification.create_from_string("TOSA-1.0+INT+FP")
    visitors = get_node_visitors(spec)
    assert "tosa.CONST.default" not in visitors
    serializer = TosaSerializerWithConstantPool(
        "", targetMajor=1, targetMinor=0, targetPatch=0, targetDraft=False
    )
    graph = Graph()
    tensor = graph.placeholder("input_tensor")
    with FakeTensorMode():
        tensor.meta["val"] = torch.empty(2, 3, dtype=dtype)
    constant_dtype = (
        torch.int8 if target == exir_ops.backend.tosa.MUL.default else dtype
    )
    constants = [create_const(graph, value, constant_dtype, tensor) for value in values]
    if target == exir_ops.backend.tosa.MUL.default:
        args = (tensor, tensor, constants[0])
    elif target == exir_ops.backend.tosa.NEGATE.default:
        args = (tensor, *constants)
    else:
        padding = graph.call_function(
            exir_ops.backend.tosa.CONST_SHAPE.default, ([0, 0, 1, 1],)
        )
        padding.meta["val"] = [0, 0, 1, 1]
        padding.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
        args = (tensor, padding, constants[0])
    operation = graph.call_function(target, args)
    operation.meta["val"] = tensor.meta["val"]
    if target == exir_ops.backend.tosa.NEGATE.default and dtype == torch.int8:
        operation.meta["input_qparams"] = {
            0: QuantArgs(0.5, values[0], -128, 127, dtype)
        }
        operation.meta["output_qparams"] = {
            0: QuantArgs(0.5, values[1], -128, 127, dtype)
        }
    graph.output(operation)

    process_inputs(tensor, serializer, spec)
    for node in graph.nodes:
        if node.op == "call_function":
            process_call_function(node, serializer, visitors, spec)

    block = TosaGraph.GetRootAs(serializer.serialize(), 0).Regions(0).Blocks(0)
    tensor_names = {
        block.Tensors(index).Name().decode() for index in range(block.TensorsLength())
    }
    assert all(constant.name not in tensor_names for constant in constants)
    declared_names = tensor_names | {
        block.Shapes(index).Name().decode() for index in range(block.ShapesLength())
    }
    for index in range(block.OperatorsLength()):
        serialized = block.Operators(index)
        assert all(
            serialized.Inputs(input_index).decode() in declared_names
            for input_index in range(serialized.InputsLength())
        )
    operation_inputs = block.Operators(block.OperatorsLength() - 1)
    constant_inputs_start = 1 if target == exir_ops.backend.tosa.NEGATE.default else 2
    numpy_dtype = {
        torch.float32: np.float32,
        torch.int8: np.int8,
        torch.int32: np.int32,
    }[constant_dtype]
    for index, expected in enumerate(values, start=constant_inputs_start):
        name = operation_inputs.Inputs(index).decode()
        serialized_tensor = next(
            block.Tensors(tensor_index)
            for tensor_index in range(block.TensorsLength())
            if block.Tensors(tensor_index).Name().decode() == name
        )
        assert np.frombuffer(
            serialized_tensor.DataAsNumpy(), dtype=numpy_dtype
        ).tolist() == [expected]
