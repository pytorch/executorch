# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.constant_folding import fold_constants
from executorch.exir.passes.constant_prop_pass import get_constant_placeholder_dict
from torch.export.graph_signature import InputKind


class TestConstantFolding(unittest.TestCase):
    def test_boundaries_signatures_and_unresolved_inputs(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4))
                self.mul_folded = torch.nn.Parameter(torch.randn(4))

            def forward(self, x):
                weight = self.weight * 2
                return x + weight + weight + self.mul_folded

        for kind in (InputKind.PARAMETER, InputKind.CONSTANT_TENSOR):
            with self.subTest(kind=kind):
                model = Model()
                inputs = (torch.randn(4),)
                program = torch.export.export(model, inputs)
                selected = {n for n in program.graph.nodes if n.op == "call_function"}
                known = get_constant_placeholder_dict(program)
                before = str(program.graph)
                self.assertFalse(fold_constants(program, selected, {}, kind))
                self.assertEqual(str(program.graph), before)
                self.assertTrue(fold_constants(program, selected, known, kind))
                program.validate()
                program.graph.lint()
                torch.testing.assert_close(program.module()(*inputs), model(*inputs))
                lifted = [
                    s
                    for s in program.graph_signature.input_specs
                    if s.arg.name.startswith("mul_folded_")
                ]
                self.assertEqual(len(lifted), 1)
                self.assertEqual(lifted[0].kind, kind)
                self.assertFalse(
                    fold_constants(
                        program, selected & set(program.graph.nodes), known, kind
                    )
                )
