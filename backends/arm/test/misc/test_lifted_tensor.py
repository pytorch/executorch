# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections.abc import Callable
from typing import Union

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)
from executorch.backends.test.harness.stages import StageType


LiftedTensorInputs = tuple[torch.Tensor, int]
LiftedTensorCase = tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    LiftedTensorInputs,
]
LiftedScalarTensorInputs = tuple[torch.Tensor, ...]
LiftedScalarTensorCase = tuple[
    Callable[[torch.Tensor, Union[float, int, torch.Tensor]], torch.Tensor],
    LiftedScalarTensorInputs,
    Union[float, int, torch.Tensor],
]


class LiftedTensor(torch.nn.Module):

    test_data: dict[str, LiftedTensorCase] = {
        # test_name: (operator, test_data, length)
        "add": (operator.add, (torch.randn(2, 2), 2)),
        "truediv": (operator.truediv, (torch.ones(2, 2), 2)),
        "mul": (operator.mul, (torch.randn(2, 2), 2)),
        "sub": (operator.sub, (torch.rand(2, 2), 2)),
    }

    def __init__(self, op: callable):  # type: ignore[valid-type]
        super().__init__()
        self.op = op
        self.lifted_tensor = torch.Tensor([[1, 2], [3, 4]])

    def forward(self, x: torch.Tensor, length) -> torch.Tensor:
        sliced = self.lifted_tensor[:, :length]
        return self.op(sliced, x)  # type: ignore[misc]


class LiftedScalarTensor(torch.nn.Module):
    test_data: dict[str, LiftedScalarTensorCase] = {
        # test_name: (operator, test_data)
        "add": (operator.add, (torch.randn(2, 2),), 1.0),
        "truediv": (operator.truediv, (torch.randn(4, 2),), 1.0),
        "mul": (operator.mul, (torch.randn(1, 2),), 2.0),
        "sub": (operator.sub, (torch.randn(3),), 1.0),
    }

    def __init__(self, op: callable, arg1: Union[int, float, torch.tensor]):  # type: ignore[valid-type]
        super().__init__()
        self.op = op
        self.arg1 = arg1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x, self.arg1)  # type: ignore[misc]


"""Tests the ArmPartitioner with a placeholder of type lifted tensor."""


@common.parametrize("test_data", LiftedTensor.test_data)
def test_partition_lifted_tensor_tosa_FP(test_data: LiftedTensorCase) -> None:
    op, inputs = test_data
    module = LiftedTensor(op)
    aten_ops: list[str] = []
    pipeline = TosaPipelineFP[LiftedTensorInputs](
        module,
        inputs,
        aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()
    signature = (
        pipeline.tester.stages[StageType.TO_EDGE]
        .artifact.exported_program()
        .graph_signature
    )
    assert len(signature.lifted_tensor_constants) > 0


@common.parametrize("test_data", LiftedTensor.test_data)
def test_partition_lifted_tensor_tosa_INT(test_data: LiftedTensorCase) -> None:
    op, inputs = test_data
    module = LiftedTensor(op)
    aten_ops: list[str] = []
    pipeline = TosaPipelineINT[LiftedTensorInputs](
        module,
        inputs,
        aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()
    signature = (
        pipeline.tester.stages[StageType.TO_EDGE]
        .artifact.exported_program()
        .graph_signature
    )
    assert len(signature.lifted_tensor_constants) == 0


@common.parametrize("test_data", LiftedScalarTensor.test_data)
def test_partition_lifted_scalar_tensor_tosa_FP(
    test_data: LiftedScalarTensorCase,
) -> None:
    op, tensor_inputs, scalar_arg = test_data
    module = LiftedScalarTensor(op, scalar_arg)
    aten_ops: list[str] = []
    pipeline = TosaPipelineFP[LiftedScalarTensorInputs](
        module,
        tensor_inputs,
        aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_data", LiftedScalarTensor.test_data)
def test_partition_lifted_scalar_tensor_tosa_INT(
    test_data: LiftedScalarTensorCase,
) -> None:
    op, tensor_inputs, scalar_arg = test_data
    module = LiftedScalarTensor(op, scalar_arg)
    aten_ops: list[str] = []
    pipeline = TosaPipelineINT[LiftedScalarTensorInputs](
        module,
        tensor_inputs,
        aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()
