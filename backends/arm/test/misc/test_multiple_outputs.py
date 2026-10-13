# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)
from executorch.backends.test.harness.stages import StageType


input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class MultipleOutputsModule(torch.nn.Module):
    inputs: dict[str, input_t1] = {
        "randn": (torch.randn(10, 4, 5), torch.randn(10, 4, 5)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (x * y, x.sum(dim=-1, keepdim=True))


@common.parametrize("test_data", MultipleOutputsModule.inputs)
def test_multiple_outputs_tosa_FP(test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineFP[input_t1](
        MultipleOutputsModule(), test_data, aten_ops, exir_ops
    )
    pipeline.run()


@common.parametrize("test_data", MultipleOutputsModule.inputs)
def test_multiple_outputs_tosa_INT(test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](
        MultipleOutputsModule(), test_data, aten_ops, exir_ops, qtol=1
    )
    pipeline.run()


@common.parametrize("test_data", MultipleOutputsModule.inputs)
@common.XfailIfNoCorstone300
def test_multiple_outputs_u55_INT(test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = EthosU55PipelineINT[input_t1](
        MultipleOutputsModule(), test_data, aten_ops, exir_ops, qtol=1
    )
    pipeline.run()


@common.parametrize("test_data", MultipleOutputsModule.inputs)
@common.XfailIfNoCorstone320
def test_multiple_outputs_u85_INT(test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = EthosU85PipelineINT[input_t1](
        MultipleOutputsModule(), test_data, aten_ops, exir_ops, qtol=1
    )
    pipeline.run()


class ConstantAndComputedOutputsModule(torch.nn.Module):
    first_table: torch.Tensor
    second_table: torch.Tensor
    inputs = (torch.arange(-64.0, 64.0, 8.0).reshape(8, 2),)

    def __init__(self):
        super().__init__()
        self.register_buffer("first_table", torch.tensor([[-16.0, 32.0], [5.0, 6.0]]))
        self.register_buffer("second_table", torch.tensor([[48.0, -64.0], [7.0, 8.0]]))

    def forward(self, x: torch.Tensor):
        first = self.first_table[:1]
        second = self.second_table[:1]
        a = x + first
        b = a + second
        return first, second, a, b, b[:4]


@common.parametrize("fold_quantize", {"folded": True, "unfolded": False})
def test_constant_and_computed_outputs_u85_INT_contract(
    tmp_path: Path, fold_quantize: bool
):
    pipeline = EthosU85PipelineINT[Tuple[torch.Tensor]](
        ConstantAndComputedOutputsModule(),
        ConstantAndComputedOutputsModule.inputs,
        aten_ops=[],
        run_on_fvp=False,
        custom_path=str(tmp_path),
        fold_quantize=fold_quantize,
    )
    pipeline.run()

    artifact = pipeline.tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    delegate = next(
        node
        for node in artifact.exported_program().graph.nodes
        if node.target == torch.ops.higher_order.executorch_call_delegate
    )
    outputs = delegate.meta["val"]
    assert all(output.dtype == torch.int8 for output in outputs)
    with np.load(tmp_path / "output" / "out_vela.npz") as data:
        actual_bytes = np.prod(data["output_shape"], axis=1) * data["output_elem_size"]
        expected_bytes = [output.numel() * output.element_size() for output in outputs]
        assert actual_bytes.tolist() == expected_bytes
        assert data["output_elem_size"].tolist() == [1] * len(outputs)


@common.parametrize("fold_quantize", {"folded": True, "unfolded": False})
@common.XfailIfNoCorstone320
def test_constant_and_computed_outputs_u85_INT(fold_quantize: bool):
    pipeline = EthosU85PipelineINT[Tuple[torch.Tensor]](
        ConstantAndComputedOutputsModule(),
        ConstantAndComputedOutputsModule.inputs,
        aten_ops=[],
        qtol=1,
        rtol=0,
        fold_quantize=fold_quantize,
    )
    pipeline.run()
