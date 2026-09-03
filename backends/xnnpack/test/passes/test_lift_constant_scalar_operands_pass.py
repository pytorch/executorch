# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack._passes.convert_to_sdpa import ConvertToSDPAPass
from executorch.backends.xnnpack._passes.lift_constant_scalar_operands_pass import (
    LiftConstantScalarOperandsPass,
)
from executorch.backends.xnnpack.utils.configs import (
    get_transform_passes,
    get_xnnpack_edge_compile_config,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind


class TestLiftConstantScalarOperandsPass(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class MulScalar(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.mul.Scalar(x, 0.5)

    class AddScalar(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.add.Scalar(x, 0.5)

    class SDPA(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

    def _to_edge_program_manager(self, module):
        return to_edge(
            torch.export.export(module, (torch.randn(2, 3),), strict=True),
            compile_config=get_xnnpack_edge_compile_config(skip_dim_order=True),
        )

    def _lift(self, exported_program):
        return XNNPACKPassManager(
            exported_program, passes=[LiftConstantScalarOperandsPass]
        ).transform()

    def test_lifts_mul_scalar_operand(self):
        exported_program = self._lift(
            self._to_edge_program_manager(self.MulScalar()).exported_program()
        )
        graph = exported_program.graph_module.graph

        self.assertFalse(
            any(node.target == exir_ops.edge.aten.mul.Scalar for node in graph.nodes)
        )
        self.assertTrue(
            any(node.target == exir_ops.edge.aten.mul.Tensor for node in graph.nodes)
        )
        self.assertFalse(any(node.op == "get_attr" for node in graph.nodes))

        constant_specs = [
            spec
            for spec in exported_program.graph_signature.input_specs
            if spec.kind == InputKind.CONSTANT_TENSOR
        ]
        self.assertEqual(len(constant_specs), 1)
        constant_spec = constant_specs[0]
        self.assertIn(constant_spec.target, exported_program.constants)

        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        self.assertEqual(placeholders[0].name, constant_spec.arg.name)
        mul_node = next(
            node for node in graph.nodes if node.target == exir_ops.edge.aten.mul.Tensor
        )
        self.assertIs(mul_node.args[1], placeholders[0])

    def test_is_idempotent(self):
        exported_program = self._lift(
            self._to_edge_program_manager(self.MulScalar()).exported_program()
        )
        exported_program = self._lift(exported_program)

        constant_specs = [
            spec
            for spec in exported_program.graph_signature.input_specs
            if spec.kind == InputKind.CONSTANT_TENSOR
        ]
        self.assertEqual(len(constant_specs), 1)
        self.assertEqual(len(exported_program.constants), 1)

    def test_keeps_unmapped_scalar_op(self):
        exported_program = self._lift(
            self._to_edge_program_manager(self.AddScalar()).exported_program()
        )
        graph = exported_program.graph_module.graph

        self.assertTrue(
            any(node.target == exir_ops.edge.aten.add.Scalar for node in graph.nodes)
        )
        self.assertFalse(exported_program.constants)

    def test_converts_sdpa_after_default_transform_passes(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        mask = torch.randn(8, 8)
        for use_default_transforms in (False, True):
            with self.subTest(use_default_transforms=use_default_transforms):
                edge = to_edge(
                    torch.export.export(self.SDPA(), (q, k, v, mask), strict=True),
                    compile_config=get_xnnpack_edge_compile_config(),
                )
                if use_default_transforms:
                    edge = edge.transform(get_transform_passes())
                exported_program = XNNPACKPassManager(
                    edge.exported_program(),
                    passes=[ConvertToSDPAPass, LiftConstantScalarOperandsPass],
                ).transform()

                graph = exported_program.graph_module.graph
                self.assertTrue(
                    any(
                        node.target
                        == exir_ops.edge.aten.scaled_dot_product_attention.default
                        for node in graph.nodes
                    )
                )
                self.assertFalse(
                    any(
                        node.target == exir_ops.edge.aten.bmm.default
                        for node in graph.nodes
                    )
                )
                self.assertFalse(
                    any(
                        node.target == exir_ops.edge.aten.mul.Scalar
                        for node in graph.nodes
                    )
                )
