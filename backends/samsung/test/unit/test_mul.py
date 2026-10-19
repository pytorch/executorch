# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock

import numpy as np
import torch
from executorch.backends.samsung.builders.op_mul import MulVisitor
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.program._fake_program import get_fake_program


class Scale(torch.nn.Module):
    def forward(self, x):
        return x * (32**-0.5)


class TestMulVisitor(unittest.TestCase):
    def test_partition_support_check_does_not_materialize_fake_constants(self):
        program = to_edge(
            torch.export.export(Scale(), (torch.randn(1, 2, 100, 100),)),
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        ).exported_program()
        program = get_fake_program(program)
        mul = next(
            node
            for node in program.graph.nodes
            if node.target == exir_ops.edge.aten.mul.Tensor
        )
        graph = Mock()
        graph.define_tensor.side_effect = range(3)
        self.assertTrue(MulVisitor(program).define_node(mul, graph, {}))
        self.assertIsNone(graph.define_tensor.call_args_list[1].args[4])

    def test_attention_scale_has_matching_operand_shapes(self):
        shape = (1, 2, 100, 100)
        program = to_edge(
            torch.export.export(Scale(), (torch.randn(shape),)),
            compile_config=EdgeCompileConfig(_skip_dim_order=True),
        ).exported_program()
        mul = next(
            node
            for node in program.graph.nodes
            if node.target == exir_ops.edge.aten.mul.Tensor
        )
        graph = Mock()
        graph.define_tensor.side_effect = range(3)
        self.assertTrue(MulVisitor(program).define_node(mul, graph, {}))
        constant = graph.define_tensor.call_args_list[1]
        self.assertEqual(constant.args[1], list(shape))
        np.testing.assert_allclose(constant.args[4], np.full(shape, 32**-0.5))
        self.assertEqual(graph.define_op.call_args.args[2], [0, 1])


if __name__ == "__main__":
    unittest.main()
