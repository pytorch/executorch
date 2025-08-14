# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


import unittest
from typing import List

import coremltools as ct

import torch
from executorch.backends.apple.coreml.recipes import (
    CoreMLRecipeProvider,
    CoreMLRecipeType,
)

from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.schema import DelegateCall, Program
from executorch.export import export, ExportRecipe, recipe_registry
from torch import nn
from torch.testing._internal.common_quantization import TestHelperModules


class TestCoreMLRecipes(unittest.TestCase):
    fp32_recipes: List[CoreMLRecipeType] = [
        CoreMLRecipeType.FP32,
    ]
    fp16_recipes: List[CoreMLRecipeType] = [
        CoreMLRecipeType.FP16,
    ]

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()
        self.provider = CoreMLRecipeProvider()
        # Register the provider for recipe registry tests
        recipe_registry.register_backend_recipe_provider(CoreMLRecipeProvider())

    def tearDown(self):
        super().tearDown()

    def check_fully_delegated(self, program: Program) -> None:
        instructions = program.execution_plan[0].chains[0].instructions
        assert instructions is not None
        self.assertEqual(len(instructions), 1)
        self.assertIsInstance(instructions[0].instr_args, DelegateCall)

    def test_all_fp32_recipes_with_simple_model(self):
        """Test all FP32 recipes with a simple linear model"""
        for recipe_type in self.fp32_recipes:
            with self.subTest(recipe=recipe_type.value):
                m_eager = TestHelperModules.TwoLinearModule().eval()
                example_inputs = [(torch.randn(9, 8),)]

                session = export(
                    model=m_eager,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type),
                )
                self.check_fully_delegated(session.get_executorch_program())

                # Verify outputs match
                if IS_VALID_TEST_RUNTIME:
                    self.assertTrue(
                        torch.allclose(
                            session.run_method("forward", example_inputs[0])[0],
                            m_eager(*example_inputs[0]),
                            atol=1e-3,
                        )
                    )

    def test_all_fp16_recipes_with_simple_model(self):
        """Test all FP16 recipes with a simple linear model"""

        for recipe_type in self.fp16_recipes:
            with self.subTest(recipe=recipe_type.value):
                m_eager = TestHelperModules.TwoLinearModule().eval()
                example_inputs = [(torch.randn(9, 8),)]

                session = export(
                    model=m_eager,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type),
                )

                self.check_fully_delegated(session.get_executorch_program())

                # Verify outputs match (slightly higher tolerance for FP16)
                if IS_VALID_TEST_RUNTIME:
                    self.assertTrue(
                        torch.allclose(
                            session.run_method("forward", example_inputs[0])[0],
                            m_eager(*example_inputs[0]),
                            atol=1e-3,
                        )
                    )

    def test_custom_simple_model(self):
        """Test with a custom simple model"""

        class CustomTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(20, 1)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        model = CustomTestModel().eval()
        example_inputs = [(torch.randn(1, 10),)]
        for recipe_type in self.fp32_recipes + self.fp16_recipes:
            with self.subTest(recipe=recipe_type.value):
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type),
                )
                session.print_delegation_info()
                self.check_fully_delegated(session.get_executorch_program())

                if IS_VALID_TEST_RUNTIME:
                    self.assertTrue(
                        torch.allclose(
                            session.run_method("forward", example_inputs[0])[0],
                            model(*example_inputs[0]),
                            atol=1e-3,
                        )
                    )

    def test_unsupported_recipe_type(self):
        """Test that unsupported recipe types return None"""
        from executorch.export import RecipeType

        class UnsupportedRecipeType(RecipeType):
            UNSUPPORTED = "unsupported"

            @classmethod
            def get_backend_name(cls) -> str:
                return "dummy"

        recipe = self.provider.create_recipe(UnsupportedRecipeType.UNSUPPORTED)
        self.assertIsNone(recipe)

    def test_recipe_registry_integration(self):
        """Test that recipes work with the global recipe registry"""
        for recipe_type in self.fp32_recipes + self.fp16_recipes:
            with self.subTest(recipe=recipe_type.value):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertIsNotNone(recipe)
                self.assertEqual(recipe.name, recipe_type.value)

    def test_invalid_recipe_kwargs(self):
        """Test detailed error messages for invalid kwargs"""
        provider = CoreMLRecipeProvider()

        # Test single invalid parameter
        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(CoreMLRecipeType.FP16, invalid_param=123)

        error_msg = str(cm.exception)
        self.assertIn("Unexpected parameters", error_msg)

        # Test multiple invalid parameters
        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(
                CoreMLRecipeType.FP32, param1="value1", param2="value2"
            )

        error_msg = str(cm.exception)
        self.assertIn("Unexpected parameters", error_msg)

        # Test mix of valid and invalid parameters
        with self.assertRaises(ValueError) as cm:
            provider.create_recipe(
                CoreMLRecipeType.FP32,
                minimum_deployment_target=ct.target.iOS16,  # valid
                invalid_param="invalid",  # invalid
            )

        error_msg = str(cm.exception)
        self.assertIn("Unexpected parameters", error_msg)

    def test_valid_kwargs(self):
        """Test valid kwargs"""
        recipe = self.provider.create_recipe(
            CoreMLRecipeType.FP32,
            minimum_deployment_target=ct.target.iOS16,
            compute_unit=ct.ComputeUnit.CPU_AND_GPU,
        )
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.name, "coreml_fp32")

        # Verify partitioners are properly configured
        partitioners = recipe.lowering_recipe.partitioners
        self.assertEqual(len(partitioners), 1, "Expected exactly one partitioner")

        # Verify delegation spec and compile specs
        delegation_spec = partitioners[0].delegation_spec
        self.assertIsNotNone(delegation_spec, "Delegation spec should not be None")

        compile_specs = delegation_spec.compile_specs
        self.assertIsNotNone(compile_specs, "Compile specs should not be None")

        spec_dict = {spec.key: spec.value for spec in compile_specs}

        # Assert that all expected specs are present with correct values
        self.assertIn(
            "min_deployment_target",
            spec_dict,
            "minimum_deployment_target should be in compile specs",
        )
        min_target_value = spec_dict["min_deployment_target"]
        if isinstance(min_target_value, bytes):
            min_target_value = min_target_value.decode("utf-8")
        self.assertEqual(
            str(min_target_value),
            str(ct.target.iOS16.value),
            "minimum_deployment_target should match the provided value",
        )

        self.assertIn(
            "compute_units", spec_dict, "compute_unit should be in compile specs"
        )
        compute_unit_value = spec_dict["compute_units"]
        if isinstance(compute_unit_value, bytes):
            compute_unit_value = compute_unit_value.decode("utf-8")
        self.assertEqual(
            str(compute_unit_value),
            ct.ComputeUnit.CPU_AND_GPU.name.lower(),
            "compute_unit should match the provided value",
        )
