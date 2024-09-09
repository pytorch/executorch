# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester


class LiftedTensor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lifted_tensor = torch.Tensor([[1, 2], [3, 4]])

    def forward(self, x: torch.Tensor, length) -> torch.Tensor:
        sliced = self.lifted_tensor[:, :length]
        return sliced + x


class TestLiftedTensor(unittest.TestCase):
    """Tests the ArmPartitioner with a placeholder of type lifted tensor."""

    def test_partition_lifted_tensor(self):
        tester = (
            ArmTester(
                LiftedTensor(),
                example_inputs=(torch.ones(2, 2), 2),
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .to_edge()
            .dump_artifact()
        )
        signature = tester.get_artifact().exported_program().graph_signature
        assert len(signature.lifted_tensor_constants) > 0
        tester.partition()
        tester.to_executorch()
        tester.run_method_and_compare_outputs((torch.ones(2, 2), 2))
