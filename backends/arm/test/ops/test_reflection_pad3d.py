# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import EthosU55PipelineINT


input_t1 = Tuple[torch.Tensor]

test_data_suite_u55 = {
    "rank4_symmetric": lambda: (torch.rand(2, 4, 4, 4), (1, 1, 1, 1, 1, 1)),
    "rank5_symmetric": lambda: (torch.rand(1, 2, 4, 4, 4), (1, 1, 1, 1, 1, 1)),
    "asymmetric": lambda: (torch.rand(1, 2, 5, 5, 5), (1, 2, 2, 1, 3, 1)),
    "maximum_legal": lambda: (torch.rand(1, 2, 4, 4, 4), (3, 3, 3, 3, 3, 3)),
    "batched": lambda: (torch.rand(2, 2, 4, 4, 4), (1, 1, 1, 1, 1, 1)),
}


class ReflectionPad3d(torch.nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.pad(x, self.padding, mode="reflect")


@common.parametrize("test_data", test_data_suite_u55)
@common.XfailIfNoCorstone300
def test_reflection_pad3d_u55_INT(test_data):
    data, padding = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        ReflectionPad3d(padding),
        (data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_reflection_pad3d_u55_INT_a16w8():
    pipeline = EthosU55PipelineINT[input_t1](
        ReflectionPad3d((1, 1, 1, 1, 1, 1)),
        (torch.rand(1, 2, 4, 4, 4),),
        aten_ops=[],
        exir_ops=[],
        a16w8_quantization=True,
    )
    pipeline.run()


def test_reflection_pad3d_u55_INT_symbolic_width_not_delegated():
    width = torch.export.Dim("width", min=3, max=8)
    tester = ArmTester(
        ReflectionPad3d((1, 1, 1, 1, 1, 1)),
        (torch.rand(1, 2, 4, 4, 5),),
        common.get_u55_compile_spec(),
        dynamic_shapes={"x": {4: width}},
    )
    tester.quantize().export().to_edge().partition()

    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }
    assert torch.ops.aten.scalar_tensor.default in targets
    assert torch.ops.higher_order.executorch_call_delegate not in targets
