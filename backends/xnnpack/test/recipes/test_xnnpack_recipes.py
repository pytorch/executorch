# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.xnnpack.recipes.xnnpack_recipe_provider import (
    XNNPACKRecipeProvider,
)
from executorch.backends.xnnpack.recipes.xnnpack_recipe_types import XNNPackRecipeType
from executorch.backends.xnnpack.test.tester import Tester
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.xnnpack import MODEL_NAME_TO_OPTIONS, QuantType
from executorch.exir.schema import DelegateCall, Program
from executorch.export import export, ExportRecipe, recipe_registry, StageType
from torch import nn
from torch.testing._internal.common_quantization import TestHelperModules
from torchao.quantization.utils import compute_error


class TestXnnpackRecipes(unittest.TestCase):
    def setUp(self) -> None:
        torch._dynamo.reset()
        super().setUp()
        recipe_registry.register_backend_recipe_provider(XNNPACKRecipeProvider())

    def tearDown(self) -> None:
        super().tearDown()

    def check_fully_delegated(self, program: Program) -> None:
        instructions = program.execution_plan[0].chains[0].instructions
        assert instructions is not None
        self.assertEqual(len(instructions), 1)
        self.assertIsInstance(instructions[0].instr_args, DelegateCall)

    # pyre-ignore
    def _compare_eager_quantized_model_outputs(
        self, session, example_inputs, atol: float
    ) -> None:
        """Utility to compare eager quantized model output with session output after xnnpack lowering"""
        torch_export_stage_output = session.get_stage_artifacts()[
            StageType.TORCH_EXPORT
        ]
        eager_quantized_model = torch_export_stage_output.data["forward"].module()
        output = session.run_method("forward", example_inputs[0])[0]
        expected = eager_quantized_model(*example_inputs[0])
        Tester._assert_outputs_equal(output, expected, atol=atol)

    def _compare_eager_unquantized_model_outputs(
        self, session, eager_unquantized_model, example_inputs, sqnr_threshold=20
    ):
        """Utility to compare eager unquantized model output with session output using SQNR"""
        quantized_output = session.run_method("forward", example_inputs[0])[0]
        original_output = eager_unquantized_model(*example_inputs[0])
        error = compute_error(original_output, quantized_output)
        print(f"{self._testMethodName} - SQNR: {error} dB")
        self.assertTrue(error > sqnr_threshold)

    def test_basic_recipe(self) -> None:
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        session = export(
            model=m_eager,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(XNNPackRecipeType.FP32),
        )
        self._compare_eager_quantized_model_outputs(session, example_inputs, 1e-3)
        self.check_fully_delegated(session.get_executorch_program())
        self._compare_eager_unquantized_model_outputs(session, m_eager, example_inputs)

    def test_int8_dynamic_quant_recipe(self) -> None:
        test_cases = [
            ExportRecipe.get_recipe(XNNPackRecipeType.PT2E_INT8_DYNAMIC_PER_CHANNEL),
        ]

        for export_recipe in test_cases:
            with self.subTest(export_recipe=export_recipe):
                with torch.no_grad():
                    m_eager = TestHelperModules.TwoLinearModule().eval()
                    example_inputs = [(torch.randn(9, 8),)]
                    session = export(
                        model=m_eager,
                        example_inputs=example_inputs,
                        export_recipe=export_recipe,
                    )
                    self._compare_eager_quantized_model_outputs(
                        session, example_inputs, 1e-1
                    )
                    self.check_fully_delegated(session.get_executorch_program())
                    self._compare_eager_unquantized_model_outputs(
                        session, m_eager, example_inputs
                    )

    def test_int8_static_quant_recipe(self) -> None:
        test_cases = [
            ExportRecipe.get_recipe(XNNPackRecipeType.PT2E_INT8_STATIC_PER_CHANNEL),
            ExportRecipe.get_recipe(XNNPackRecipeType.PT2E_INT8_STATIC_PER_TENSOR),
        ]

        for export_recipe in test_cases:
            with self.subTest(export_recipe=export_recipe):
                with torch.no_grad():
                    m_eager = TestHelperModules.TwoLinearModule().eval()
                    example_inputs = [(torch.randn(9, 8),)]
                    session = export(
                        model=m_eager,
                        example_inputs=example_inputs,
                        export_recipe=export_recipe,
                    )
                    self._compare_eager_quantized_model_outputs(
                        session, example_inputs, 1e-2
                    )
                    self.check_fully_delegated(session.get_executorch_program())
                    self._compare_eager_unquantized_model_outputs(
                        session, m_eager, example_inputs
                    )

    def test_8a4w_recipe(self) -> None:
        class SimpleLinearModel(nn.Module):
            def __init__(self) -> None:
                super(SimpleLinearModel, self).__init__()
                self.layer1 = nn.Linear(32, 2)

            def forward(self, x) -> torch.Tensor:
                x = self.layer1(x)
                return x

        test_cases = [
            ExportRecipe.get_recipe(
                XNNPackRecipeType.TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_CHANNEL,
            ),
            ExportRecipe.get_recipe(
                XNNPackRecipeType.TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR,
                group_size=8,
            ),
        ]

        for export_recipe in test_cases:
            with self.subTest(export_recipe=export_recipe):
                model = SimpleLinearModel().eval()
                example_inputs = [(torch.randn(1, 32),)]
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=export_recipe,
                )
                self.check_fully_delegated(session.get_executorch_program())
                self._compare_eager_quantized_model_outputs(
                    session, example_inputs, 1e-3
                )
                self._compare_eager_unquantized_model_outputs(
                    session, model, example_inputs, 14
                )

    def _get_recipe_for_quant_type(self, quant_type: QuantType) -> XNNPackRecipeType:
        # Map QuantType to corresponding recipe name.
        if quant_type == QuantType.STATIC_PER_CHANNEL:
            return XNNPackRecipeType.PT2E_INT8_STATIC_PER_CHANNEL
        elif quant_type == QuantType.DYNAMIC_PER_CHANNEL:
            return XNNPackRecipeType.PT2E_INT8_DYNAMIC_PER_CHANNEL
        elif quant_type == QuantType.STATIC_PER_TENSOR:
            return XNNPackRecipeType.PT2E_INT8_STATIC_PER_TENSOR
        elif quant_type == QuantType.NONE:
            return XNNPackRecipeType.FP32
        else:
            raise ValueError(f"Unsupported QuantType: {quant_type}")

    def _test_model_with_factory(self, model_name: str) -> None:
        if model_name not in MODEL_NAME_TO_MODEL:
            self.skipTest(f"Model {model_name} not found in MODEL_NAME_TO_MODEL")
            return

        if model_name not in MODEL_NAME_TO_OPTIONS:
            self.skipTest(f"Model {model_name} not found in MODEL_NAME_TO_OPTIONS")
            return

        # Create model using factory
        model, example_inputs, _example_kwarg_inputs, dynamic_shapes = (
            EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
        )
        model = model.eval()

        # Get the appropriate recipe based on quantization type
        options = MODEL_NAME_TO_OPTIONS[model_name]
        recipe_name = self._get_recipe_for_quant_type(options.quantization)

        # Export with recipe
        session = export(
            model=model,
            example_inputs=[example_inputs],
            export_recipe=ExportRecipe.get_recipe(recipe_name),
            dynamic_shapes=dynamic_shapes,
        )

        # Verify outputs match
        Tester._assert_outputs_equal(
            session.run_method("forward", example_inputs)[0],
            model(*example_inputs),
            atol=1e-3,
        )

    @unittest.skip("T187799178: Debugging Numerical Issues with Calibration")
    def test_all_models_with_recipes(self) -> None:
        models_to_test = [
            "linear",
            "add",
            "add_mul",
            "ic3",
            "mv2",
            "mv3",
            "resnet18",
            "resnet50",
            "vit",
            "w2l",
            "llama2",
        ]
        for model_name in models_to_test:
            with self.subTest(model=model_name):
                self._test_model_with_factory(model_name)

    def test_validate_recipe_kwargs_fp32(self) -> None:
        provider = XNNPACKRecipeProvider()

        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(XNNPackRecipeType.FP32, invalid_param=123)

        error_msg = str(cm.exception)
        self.assertIn("Recipe 'fp32' does not accept any parameters", error_msg)

    def test_validate_recipe_kwargs_int4_tensor_with_valid_group_size(
        self,
    ) -> None:
        provider = XNNPACKRecipeProvider()

        # Should not raise any exception
        recipe_w_default_group = provider.create_recipe(
            XNNPackRecipeType.TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR
        )
        self.assertIsNotNone(recipe_w_default_group)

        recipe = provider.create_recipe(
            XNNPackRecipeType.TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR,
            group_size=64,
        )
        self.assertIsNotNone(recipe)

    def test_validate_recipe_kwargs_int4_tensor_with_invalid_group_size(
        self,
    ) -> None:
        provider = XNNPACKRecipeProvider()

        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(
                XNNPackRecipeType.TORCHAO_INT8_DYNAMIC_ACT_INT4_WEIGHT_PER_TENSOR,
                group_size="32",  # String instead of int
            )

        error_msg = str(cm.exception)
        self.assertIn(
            "Parameter 'group_size' must be an integer, got str: 32", error_msg
        )
