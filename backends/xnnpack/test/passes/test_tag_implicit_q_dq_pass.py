# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.tag_implicit_q_dq_pass import (
    TagImplicitQDqPass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.dialects._ops import ops as exir_ops


class TestTagImplicitQDq(unittest.TestCase):
    PassStage = RunPasses([DuplicateDequantNodePass, TagImplicitQDqPass])

    class QDqModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            qparams = [0.12345, 0, -127, 127, torch.int8]
            x = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                x, *qparams
            )
            x = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                x, *qparams
            )
            x = torch.add(x, x)
            x = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                x, *qparams
            )
            x = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                x, *qparams
            )
            x = torch.mul(x, x)
            x = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                x, *qparams
            )
            x = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                x, *qparams
            )
            x = torch.add(x, x)
            x = torch.mul(x, x)
            return x

    def test_tag_implicit_q_dq_test(self):
        inputs = (torch.randn(2, 3),)
        artifact = (
            Tester(self.QDqModule(), inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .run_method_and_compare_outputs()
            .get_artifact(Tester.stage_name(self.PassStage))
        )

        for node in artifact.exported_program().module().graph.nodes:
            print(
                f"{node}: {node.meta.get(TagImplicitQDqPass.IS_IMPLICIT_Q_DQ_TAG, False)}"
            )

        # The six tagged nodes are:
        # 1) The dq of the first add input
        # 2) The dq of the second add input
        # 3) The q of the add output
        # 4) The dq of the first mul input
        # 5) The dq of the second mul input
        # 6) The q of the mul output
        self.assertEqual(
            sum(
                node.meta.get(TagImplicitQDqPass.IS_IMPLICIT_Q_DQ_TAG, False)
                for node in artifact.exported_program().module().graph.nodes
            ),
            6,
        )
