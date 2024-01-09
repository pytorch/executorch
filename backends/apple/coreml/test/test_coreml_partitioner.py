# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import unittest

import torch

import executorch.exir as exir
from executorch.exir.backend.backend_api import to_backend

from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner


class TestCoreMLPartitioner(unittest.TestCase):
    def test_partition_add_mul(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        exported_program = exir.capture(model, inputs, exir.CaptureConfig()).to_edge().exported_program

        assert [
            node.target.__name__ for node in exported_program.graph.nodes if node.op == "call_function"
        ] == [
            "aten.mm.default", "aten.add.Tensor", "aten.sub.Tensor", "aten.mm.default", "aten.add.Tensor"
        ]

        exported_to_coreml = to_backend(
            exported_program,
            CoreMLPartitioner(skip_ops_for_coreml_delegation=["aten.mm.default"]),
        )

        assert [
            node.target.__name__ for node in exported_to_coreml.graph.nodes if node.op == "call_function"
        ] == [
            "aten.mm.default", "executorch_call_delegate", "getitem", "aten.mm.default", "executorch_call_delegate", "getitem"
        ]


if __name__ == "__main__":
    test_runner = TestCoreMLPartitioner()
    test_runner.test_partition_add_mul()
