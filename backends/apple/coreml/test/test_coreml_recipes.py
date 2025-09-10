# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


import unittest

import coremltools as ct
import torch

from executorch.backends.apple.coreml.recipes import (
    CoreMLRecipeProvider,
    CoreMLRecipeType,
)

from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.schema import DelegateCall
from executorch.export import export, ExportRecipe, recipe_registry, StageType

from torch import nn
from torch.testing._internal.common_quantization import TestHelperModules
from torchao.quantization.utils import compute_error


class TestCoreMLRecipes(unittest.TestCase):
    """Test suite for CoreML recipes focusing on quantization functionality"""

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()
        self.provider = CoreMLRecipeProvider()
        # Register the provider for recipe registry tests
        recipe_registry.register_backend_recipe_provider(CoreMLRecipeProvider())

    def tearDown(self):
        super().tearDown()

    def check_fully_delegated(self, session) -> None:
        """Helper to verify a program is fully delegated to CoreML"""
        session.print_delegation_info()
        program = session.get_executorch_program()
        instructions = program.execution_plan[0].chains[0].instructions
        assert instructions is not None
        self.assertEqual(len(instructions), 1)
        self.assertIsInstance(instructions[0].instr_args, DelegateCall)

    def _compare_eager_quantized_model_outputs(self, session, example_inputs, atol):
        """Utility to compare eager quantized model output with session output after coreml lowering"""
        if IS_VALID_TEST_RUNTIME:
            source_transform_output = session.get_stage_artifacts()[
                StageType.SOURCE_TRANSFORM
            ]
            eager_quantized_model = source_transform_output.data["forward"]
            output = session.run_method("forward", example_inputs[0])[0]
            expected = eager_quantized_model(*example_inputs[0])
            self.assertTrue(torch.allclose(output, expected, atol=atol))

    def _compare_eager_unquantized_model_outputs(
        self, session, eager_unquantized_model, example_inputs, sqnr_threshold=20
    ):
        """Utility to compare eager unquantized model output with session output using SQNR"""
        if IS_VALID_TEST_RUNTIME:
            quantized_output = session.run_method("forward", example_inputs[0])[0]
            original_output = eager_unquantized_model(*example_inputs[0])
            error = compute_error(original_output, quantized_output)
            print(f"SQNR: {error} dB")
            self.assertTrue(error > sqnr_threshold)

    def test_fp32_recipe(self):
        """Test FP32 recipe functionality"""
        model = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(CoreMLRecipeType.FP32),
        )
        self.check_fully_delegated(session)

        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-3)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_fp16_recipe(self):
        """Test FP16 recipe functionality"""
        model = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(CoreMLRecipeType.FP16),
        )
        self.check_fully_delegated(session)

        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-3)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_fp_recipes_with_custom_parameters(self):
        """Test FP recipes with custom deployment target and compute unit"""
        test_cases = [
            (CoreMLRecipeType.FP32, {"minimum_deployment_target": ct.target.iOS16}),
            (CoreMLRecipeType.FP16, {"compute_unit": ct.ComputeUnit.CPU_ONLY}),
        ]

        model = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]

        for recipe_type, kwargs in test_cases:
            with self.subTest(recipe=recipe_type.value, kwargs=kwargs):
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(recipe_type, **kwargs),
                )
                self.check_fully_delegated(session)

    def test_int4_weight_only_per_channel(self):
        """Test INT4 weight-only per-channel quantization"""
        model = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(
                CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL
            ),
        )
        self.check_fully_delegated(session)
        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-02)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_int4_weight_only_per_group(self):
        """Test INT4 weight-only per-group quantization with different group sizes"""

        class CustomTwoLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(32, 32)
                self.layer2 = nn.Linear(32, 8)

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = self.layer2(x)
                return x

        model = CustomTwoLinearModel().eval()
        example_inputs = [(torch.randn(1, 32),)]
        # Test with different group sizes
        for group_size in [8, 16, 32]:
            with self.subTest(group_size=group_size):
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(
                        CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP,
                        group_size=group_size,
                    ),
                )
                self.check_fully_delegated(session)

                self._compare_eager_quantized_model_outputs(
                    session, example_inputs, atol=1e-3
                )
                self._compare_eager_unquantized_model_outputs(
                    session, model, example_inputs
                )

    def test_int4_weight_only_per_group_validation(self):
        """Test INT4 per-group parameter validation"""
        # Test invalid group size type
        with self.assertRaises(ValueError) as cm:
            self.provider.create_recipe(
                CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP, group_size="32"
            )
        self.assertIn("must be an integer", str(cm.exception))

        # Test negative group size
        with self.assertRaises(ValueError) as cm:
            self.provider.create_recipe(
                CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP, group_size=-1
            )
        self.assertIn("must be positive", str(cm.exception))

    def test_int8_weight_only_per_channel(self):
        """Test INT8 weight-only per-channel quantization"""
        model = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(
                CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL
            ),
        )
        self.check_fully_delegated(session)

        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-2)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_int8_weight_only_per_group(self):
        """Test INT8 weight-only per-group quantization with different group sizes"""

        class SimpleLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(64, 2)

            def forward(self, x):
                return self.layer(x)

        model = SimpleLinearModel().eval()
        example_inputs = [(torch.randn(1, 64),)]

        # Test with different group sizes
        for group_size in [16, 32, 64]:
            with self.subTest(group_size=group_size):
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(
                        CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP,
                        group_size=group_size,
                    ),
                )
                self.check_fully_delegated(session)

                self._compare_eager_quantized_model_outputs(
                    session, example_inputs, atol=1e-2
                )
                self._compare_eager_unquantized_model_outputs(
                    session, model, example_inputs
                )

    def test_codebook_weight_only_recipe(self):
        """Test codebook quantization recipe"""

        class SimpleLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(32, 2)

            def forward(self, x):
                return self.layer(x)

        model = SimpleLinearModel().eval()
        example_inputs = [(torch.randn(1, 32),)]

        # Test different block sizes
        test_cases = [
            {"bits": 3, "block_size": [-1, 8]},
        ]

        for kwargs in test_cases:
            with self.subTest(kwargs=kwargs):
                session = export(
                    model=model,
                    example_inputs=example_inputs,
                    export_recipe=ExportRecipe.get_recipe(
                        CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY, **kwargs
                    ),
                )
                self.check_fully_delegated(session)

    def test_codebook_parameter_validation(self):
        """Test codebook parameter validation"""
        # Test invalid bits type
        with self.assertRaises(ValueError) as cm:
            self.provider.create_recipe(
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY, bits="3", block_size=[-1, 8]
            )
        self.assertIn("must be an integer", str(cm.exception))

        # Test bits out of range
        with self.assertRaises(ValueError) as cm:
            self.provider.create_recipe(
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY, bits=0, block_size=[-1, 8]
            )
        self.assertIn("must be between 1 and 8", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.provider.create_recipe(
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY, bits=9, block_size=[-1, 8]
            )
        self.assertIn("must be between 1 and 8", str(cm.exception))

        # Test invalid block_size type
        with self.assertRaises(ValueError) as cm:
            self.provider.create_recipe(
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY, bits=3, block_size="[-1, 16]"
            )
        self.assertIn("must be a list", str(cm.exception))

    def test_int8_static_quantization(self):
        """Test INT8 static quantization (weights + activations)"""

        class SimpleLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(32, 16)
                self.layer2 = nn.Linear(16, 2)

            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = self.layer2(x)
                return x

        model = SimpleLinearModel().eval()
        example_inputs = [(torch.randn(1, 32),)]

        recipe = ExportRecipe.get_recipe(
            CoreMLRecipeType.PT2E_INT8_STATIC, minimum_deployment_target=ct.target.iOS17
        )

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=recipe,
        )
        self.check_fully_delegated(session)

        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-3)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_int8_weight_only_pt2e(self):
        """Test PT2E-based INT8 weight-only quantization"""
        model = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(
                CoreMLRecipeType.PT2E_INT8_WEIGHT_ONLY
            ),
        )
        self.check_fully_delegated(session)

        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-2)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_int8_weight_only_pt2e_with_conv(self):
        """Test PT2E-based INT8 weight-only quantization with convolution layers"""

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(32, 10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = ConvModel().eval()
        example_inputs = [(torch.randn(1, 3, 32, 32),)]

        session = export(
            model=model,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(
                CoreMLRecipeType.PT2E_INT8_WEIGHT_ONLY
            ),
        )
        self.check_fully_delegated(session)

        self._compare_eager_quantized_model_outputs(session, example_inputs, atol=1e-2)
        self._compare_eager_unquantized_model_outputs(session, model, example_inputs)

    def test_filter_fn_comprehensive(self):
        """Comprehensive test for filter_fn parameter functionality"""

        def custom_filter(module, fqn):
            return isinstance(module, nn.Linear) and "target" in fqn

        # Test 1: TorchAO recipes accept filter_fn and default to None
        torchao_recipes = [
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP,
        ]

        for recipe_type in torchao_recipes:
            with self.subTest(f"{recipe_type.value}_default"):
                # Test default behavior (None)
                recipe = self.provider.create_recipe(recipe_type)
                config = recipe.quantization_recipe.ao_quantization_configs[0]
                self.assertIsNone(config.filter_fn)

            with self.subTest(f"{recipe_type.value}_custom"):
                # Test custom filter_fn
                recipe = self.provider.create_recipe(
                    recipe_type, filter_fn=custom_filter
                )
                config = recipe.quantization_recipe.ao_quantization_configs[0]
                self.assertEqual(config.filter_fn, custom_filter)

        # Test 2: Codebook recipe accepts filter_fn and has sensible default
        with self.subTest("codebook_default"):
            recipe = self.provider.create_recipe(
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY, bits=3, block_size=[-1, 16]
            )
            config = recipe.quantization_recipe.ao_quantization_configs[0]
            self.assertIsNotNone(config.filter_fn)

            # Test default filter targets Linear and Embedding layers
            linear_module = nn.Linear(10, 5)
            embedding_module = nn.Embedding(100, 10)
            conv_module = nn.Conv2d(3, 16, 3)

            self.assertTrue(config.filter_fn(linear_module, "linear"))
            self.assertTrue(config.filter_fn(embedding_module, "embedding"))
            self.assertFalse(config.filter_fn(conv_module, "conv"))

        with self.subTest("codebook_custom"):
            recipe = self.provider.create_recipe(
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY,
                filter_fn=custom_filter,
                bits=3,
                block_size=[-1, 16],
            )
            config = recipe.quantization_recipe.ao_quantization_configs[0]
            self.assertEqual(config.filter_fn, custom_filter)

    def test_quantization_recipe_structure(self):
        """Test that quantization recipes have proper structure"""
        quantization_recipes = [
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP,
            CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY,
        ]

        for recipe_type in quantization_recipes:
            with self.subTest(recipe=recipe_type.value):
                kwargs = (
                    {"bits": 3, "block_size": [-1, 16]}
                    if recipe_type == CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY
                    else {}
                )
                recipe = self.provider.create_recipe(recipe_type, **kwargs)
                self.assertIsNotNone(recipe)

                # Should have quantization recipe with ao_quantization_configs
                self.assertIsNotNone(recipe.quantization_recipe)
                self.assertIsNotNone(recipe.quantization_recipe.ao_quantization_configs)
                self.assertEqual(
                    len(recipe.quantization_recipe.ao_quantization_configs), 1
                )

                # Should have lowering recipe
                self.assertIsNotNone(recipe.lowering_recipe)
                self.assertIsNotNone(recipe.lowering_recipe.partitioners)

    def test_recipe_creation_with_defaults(self):
        """Test that recipes work with default parameters"""
        # Test that all recipes can be created without explicit parameters
        all_recipes = [
            CoreMLRecipeType.FP32,
            CoreMLRecipeType.FP16,
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP,  # should use default group_size=32
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL,
            CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP,  # should use default group_size=32
            CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY,  # should use default bits=3, block_size=[-1,16]
        ]

        for recipe_type in all_recipes:
            with self.subTest(recipe=recipe_type.value):
                kwargs = (
                    {"bits": 3, "block_size": [-1, 16]}
                    if recipe_type == CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY
                    else {}
                )
                recipe = self.provider.create_recipe(recipe_type, **kwargs)
                self.assertIsNotNone(recipe)
                self.assertEqual(recipe.name, recipe_type.value)

    def test_minimum_deployment_target_validation(self):
        """Test that minimum_deployment_target validation works correctly for quantization recipes"""
        test_cases = [
            (CoreMLRecipeType.PT2E_INT8_STATIC, ct.target.iOS17, {}),
            (CoreMLRecipeType.PT2E_INT8_WEIGHT_ONLY, ct.target.iOS17, {}),
            (
                CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_CHANNEL,
                ct.target.iOS18,
                {},
            ),
            (CoreMLRecipeType.TORCHAO_INT4_WEIGHT_ONLY_PER_GROUP, ct.target.iOS18, {}),
            (
                CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_CHANNEL,
                ct.target.iOS16,
                {},
            ),
            (CoreMLRecipeType.TORCHAO_INT8_WEIGHT_ONLY_PER_GROUP, ct.target.iOS18, {}),
            (
                CoreMLRecipeType.CODEBOOK_WEIGHT_ONLY,
                ct.target.iOS18,
                {"bits": 3, "block_size": [-1, 16]},
            ),
        ]

        for recipe_type, min_target, kwargs in test_cases:
            with self.subTest(recipe=recipe_type.value):

                # Test 1: Providing deployment target below minimum should raise ValueError
                too_low_target = ct.target.iOS15
                with self.assertRaises(ValueError) as cm:
                    self.provider.create_recipe(
                        recipe_type, minimum_deployment_target=too_low_target, **kwargs
                    )
                error_msg = str(cm.exception)
                self.assertIn(
                    f"minimum_deployment_target must be {str(min_target)} or higher",
                    error_msg,
                )

                # Test 2: Providing valid deployment target should work
                valid_recipe = self.provider.create_recipe(
                    recipe_type, minimum_deployment_target=min_target, **kwargs
                )
                self.assertIsNotNone(valid_recipe)

                # Test 3: Not providing deployment target should default to minimum
                default_recipe = self.provider.create_recipe(recipe_type, **kwargs)
                self.assertIsNotNone(default_recipe)

                # Test 4: Providing deployment target higher than minimum should work
                higher_target = (
                    ct.target.iOS18
                    if min_target == ct.target.iOS17
                    else ct.target.iOS18
                )
                higher_recipe = self.provider.create_recipe(
                    recipe_type, minimum_deployment_target=higher_target, **kwargs
                )
                self.assertIsNotNone(higher_recipe)
