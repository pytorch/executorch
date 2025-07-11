# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.xnnpack import get_xnnpack_recipe
from executorch.exir.schema import DelegateCall, Program
from executorch.export import export
from torch import nn
from torch.testing._internal.common_quantization import TestHelperModules


class TestXnnpackRecipes(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def check_fully_delegated(self, program: Program) -> None:
        instructions = program.execution_plan[0].chains[0].instructions
        assert instructions is not None
        self.assertEqual(len(instructions), 1)
        self.assertIsInstance(instructions[0].instr_args, DelegateCall)

    def test_basic_recipe(self) -> None:
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        session = export(
            model=m_eager,
            example_inputs=example_inputs,
            export_recipe=get_xnnpack_recipe("FP32_CPU_ACCELERATED_RECIPE"),
        )
        self.assertTrue(
            torch.allclose(
                session.run_method("forward", example_inputs[0])[0],
                m_eager(*example_inputs[0]),
            )
        )
        self.check_fully_delegated(session.get_executorch_program())

    def test_dynamic_quant_recipe(self) -> None:
        with torch.no_grad():
            m_eager = TestHelperModules.TwoLinearModule().eval()
            example_inputs = [(torch.randn(9, 8),)]
            session = export(
                model=m_eager,
                example_inputs=example_inputs,
                export_recipe=get_xnnpack_recipe(
                    "DYNAMIC_QUANT_CPU_ACCELERATED_RECIPE"
                ),
            )
            self.assertTrue(
                torch.allclose(
                    session.run_method("forward", example_inputs[0])[0],
                    m_eager(*example_inputs[0]),
                    atol=1e-1,
                )
            )
            self.check_fully_delegated(session.get_executorch_program())

    def test_8a4w_recipe(self) -> None:
        class SimpleLinearModel(nn.Module):
            def __init__(self) -> None:
                super(SimpleLinearModel, self).__init__()
                self.layer1 = nn.Linear(32, 2)

            def forward(self, x) -> torch.Tensor:
                x = self.layer1(x)
                return x

        model = SimpleLinearModel()
        example_inputs = [(torch.randn(1, 32),)]
        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=get_xnnpack_recipe(
                "8A4W_CPU_ACCELERATED_RECIPE", group_size=32
            ),
        )
        self.assertTrue(
            torch.allclose(
                session.run_method("forward", example_inputs[0])[0],
                model(*example_inputs[0]),
                atol=1e-1,
            )
        )
        self.check_fully_delegated(session.get_executorch_program())
