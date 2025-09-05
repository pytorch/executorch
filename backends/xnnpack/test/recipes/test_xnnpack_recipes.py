# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import unittest
from typing import List, Optional, Tuple

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
from executorch.export import (
    export,
    ExportRecipe,
    ExportSession,
    recipe_registry,
    StageType,
)
from torch import nn, Tensor
from torch.testing import FileCheck
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

    def _compare_eager_quantized_model_outputs(
        self,
        # pyre-ignore[11]
        session: ExportSession,
        example_inputs: List[Tuple[Tensor]],
        atol: float,
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
        self,
        session: ExportSession,
        eager_unquantized_model: nn.Module,
        example_inputs: List[Tuple[Tensor]],
        sqnr_threshold: int = 20,
    ) -> None:
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

    def _get_recipe_for_quant_type(self, quant_type: QuantType) -> XNNPackRecipeType:
        # Map QuantType to corresponding recipe name.
        if quant_type == QuantType.STATIC_PER_CHANNEL:
            return XNNPackRecipeType.PT2E_INT8_STATIC_PER_CHANNEL
        elif quant_type == QuantType.DYNAMIC_PER_CHANNEL:
            return XNNPackRecipeType.PT2E_INT8_DYNAMIC_PER_CHANNEL
        elif quant_type == QuantType.STATIC_PER_TENSOR:
            return XNNPackRecipeType.PT2E_INT8_STATIC_PER_TENSOR
        return XNNPackRecipeType.FP32

    def _test_model_with_factory(
        self,
        model_name: str,
        tolerance: Optional[float] = None,
        sqnr_threshold: Optional[float] = None,
    ) -> None:
        logging.info(f"Testing model {model_name}")
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

        all_artifacts = session.get_stage_artifacts()
        quantized_model = all_artifacts[StageType.QUANTIZE].data["forward"]

        edge_program_manager = all_artifacts[StageType.TO_EDGE_TRANSFORM_AND_LOWER].data
        lowered_module = edge_program_manager.exported_program().module()

        # Check if model got lowered to xnnpack backend
        FileCheck().check("torch.ops.higher_order.executorch_call_delegate").run(
            lowered_module.code
        )

        if tolerance is not None:
            quantized_output = quantized_model(*example_inputs)
            lowered_output = lowered_module(*example_inputs)
            if model_name == "dl3":
                quantized_output = quantized_output["out"]
                lowered_output = lowered_output["out"]

            # lowering error
            try:
                Tester._assert_outputs_equal(
                    lowered_output, quantized_output, atol=tolerance, rtol=tolerance
                )
            except AssertionError as e:
                raise AssertionError(
                    f"Model '{model_name}' lowering error check failed with tolerance {tolerance}"
                ) from e
            logging.info(
                f"{self._testMethodName} - {model_name} - lowering error passed"
            )

        # verify sqnr between eager model and quantized model
        if sqnr_threshold is not None:
            original_output = model(*example_inputs)
            quantized_output = quantized_model(*example_inputs)
            # lowered_output = lowered_module(*example_inputs)
            if model_name == "dl3":
                original_output = original_output["out"]
                quantized_output = quantized_output["out"]
            error = compute_error(original_output, quantized_output)
            logging.info(f"{self._testMethodName} - {model_name} - SQNR: {error} dB")
            self.assertTrue(
                error > sqnr_threshold, f"Model '{model_name}' SQNR check failed"
            )

    def test_all_models_with_recipes(self) -> None:
        models_to_test = [
            # Tuple format: (model_name, error tolerance, minimum sqnr)
            ("linear", 1e-3, 20),
            ("add", 1e-3, 20),
            ("add_mul", 1e-3, 20),
            ("dl3", 1e-3, 20),
            ("ic3", None, None),
            ("ic4", 1e-3, 20),
            ("mv2", 1e-3, None),
            ("mv3", 1e-3, None),
            ("resnet18", 1e-3, 20),
            ("resnet50", 1e-3, 20),
            ("vit", 1e-1, 10),
            ("w2l", 1e-3, 20),
        ]
        try:
            for model_name, tolerance, sqnr in models_to_test:
                with self.subTest(model=model_name):
                    with torch.no_grad():
                        self._test_model_with_factory(model_name, tolerance, sqnr)
        finally:
            # Clean up dog.jpg file if it exists
            if os.path.exists("dog.jpg"):
                os.remove("dog.jpg")

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
