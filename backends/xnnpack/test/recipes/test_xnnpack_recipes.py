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
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.test.tester import Tester
from torchvision.models.segmentation import deeplabv3, deeplabv3_resnet50  # @manual


class TestXnnpackRecipes(unittest.TestCase):
    def setUp(self) -> None:
        torch._dynamo.reset()
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
            export_recipe=get_xnnpack_recipe("FP32_RECIPE"),
        )
        self.assertTrue(
            torch.allclose(
                session.run_method("forward", example_inputs[0])[0],
                m_eager(*example_inputs[0]),
                atol=1e-1,
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
                    "DYNAMIC_PER_CHANNEL_QUANT_RECIPE"
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
    
    def test_static_quant_recipe(self) -> None:
        with torch.no_grad():
            m_eager = TestHelperModules.TwoLinearModule().eval()
            example_inputs = [(torch.randn(9, 8),)]
            session = export(
                model=m_eager,
                example_inputs=example_inputs,
                export_recipe=get_xnnpack_recipe(
                    "STATIC_PER_CHANNEL_QUANT_RECIPE"
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
                "8A4W_ACCELERATED_RECIPE", group_size=32
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

    def test_mv3_model(self) -> None:
        mv3 = models.mobilenetv3.mobilenet_v3_small(pretrained=True)
        mv3 = mv3.eval()
        model_inputs = [(torch.randn(1, 3, 224, 224),)]
        self.assertTrue(hasattr(mv3, "forward"))
        dynamic_shapes =({2: torch.export.Dim("height", min=224, max=455), 3: torch.export.Dim("width", min=224, max=455)},)
        session = export(
            model=mv3,
            example_inputs=model_inputs,
            dynamic_shapes=dynamic_shapes,
            export_recipe=get_xnnpack_recipe(
                "STATIC_PER_CHANNEL_QUANT_RECIPE"
            ),
        )

        Tester._assert_outputs_equal(
            session.run_method("forward", model_inputs[0])[0], 
            mv3(*model_inputs[0]),
            atol=1e-3,
        )

    def test_mv2_model_with_static_quant_recipe(self) -> None:
        mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights)
        mv2 = mv2.eval()
        model_inputs = [(torch.randn(1, 3, 224, 224),)]
        self.assertTrue(hasattr(mv2, "forward"))
        dynamic_shapes =({2: torch.export.Dim("height", min=224, max=455), 3: torch.export.Dim("width", min=224, max=455)},)
        session = export(
            model=mv2,
            example_inputs=model_inputs,
            dynamic_shapes=dynamic_shapes,
            export_recipe=get_xnnpack_recipe(
                "STATIC_PER_CHANNEL_QUANT_RECIPE"
            ),
        )

        Tester._assert_outputs_equal(
            session.run_method("forward", model_inputs[0])[0], 
            mv2(*model_inputs[0]),
            atol=1e-3,
        )

    def test_dl3_with_recipe(self) -> None:
        class DL3Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = deeplabv3_resnet50(
                    weights=deeplabv3.DeepLabV3_ResNet50_Weights.DEFAULT
                )

            def forward(self, *args):
                return self.m(*args)["out"]
        
        dl3 = DL3Wrapper()
        dl3 = dl3.eval()
        model_inputs = [(torch.randn(1, 3, 224, 224),)]
        self.assertTrue(hasattr(dl3, "forward"))
        session = export(
            model=dl3,
            example_inputs=model_inputs,
            export_recipe=get_xnnpack_recipe(
                "STATIC_PER_CHANNEL_QUANT_RECIPE"
            ),
        )

        Tester._assert_outputs_equal(
            session.run_method("forward", model_inputs[0])[0], 
            dl3(*model_inputs[0]),
            atol=1e-3,
        )
    