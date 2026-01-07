# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import unittest
from typing import Any, Dict, List, Optional, Tuple

import torch
from executorch.backends.xnnpack.recipes.xnnpack_recipe_provider import (
    XNNPACKRecipeProvider,
)
from executorch.backends.xnnpack.test.tester import Tester
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir.schema import DelegateCall, Program
from executorch.export import (
    export,
    ExportRecipe,
    ExportSession,
    recipe_registry,
    StageType,
)
from executorch.export.utils import (
    is_fbcode,
    is_supported_platform_for_coreml_lowering,
    is_supported_platform_for_qnn_lowering,
)
from executorch.runtime import Runtime
from torch import nn, Tensor
from torch.testing import FileCheck
from torchao.quantization.utils import compute_error


class TestTargetRecipes(unittest.TestCase):
    """Test target recipes."""

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(4, 4)
            self.linear2 = torch.nn.Linear(4, 2)

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            a = self.linear1(x)
            b = a + y
            c = b - x
            result = self.linear2(c)
            return result

    def setUp(self) -> None:
        torch._dynamo.reset()
        super().setUp()
        recipe_registry.register_backend_recipe_provider(XNNPACKRecipeProvider())
        if is_supported_platform_for_coreml_lowering():
            from executorch.backends.apple.coreml.recipes import (  # pyre-ignore
                CoreMLRecipeProvider,
            )

            # pyre-ignore
            recipe_registry.register_backend_recipe_provider(CoreMLRecipeProvider())

        if is_fbcode() and is_supported_platform_for_qnn_lowering():
            from executorch.backends.qualcomm.recipes import (  # pyre-ignore
                QNNRecipeProvider,
            )

            # pyre-ignore
            recipe_registry.register_backend_recipe_provider(QNNRecipeProvider())
        self.model = TestTargetRecipes.Model()

    def tearDown(self) -> None:
        super().tearDown()

    def check_delegated(
        self, program: Program, expected_backends: Optional[List[str]] = None
    ) -> None:
        """Check if the program has been delegated to expected backends."""
        instructions = program.execution_plan[0].chains[0].instructions
        assert instructions is not None

        if expected_backends is None:
            # Just check that there's at least one delegate call
            self.assertGreater(len(instructions), 0)
            for instruction in instructions:
                self.assertIsInstance(instruction.instr_args, DelegateCall)
        else:
            # Check for specific backends
            delegates = program.execution_plan[0].delegates
            delegate_ids = [delegate.id for delegate in delegates]
            for expected_backend in expected_backends:
                self.assertIn(
                    expected_backend,
                    delegate_ids,
                    f"Expected backend {expected_backend} not found in delegates: {delegate_ids}",
                )

    def check_num_partitions(
        self, executorch_program: Program, expected_num_partitions: int
    ) -> None:
        """Check if the program has the expected number of partitions."""
        self.assertEqual(
            len(executorch_program.execution_plan[0].delegates),
            expected_num_partitions,
        )

    def _check_lowering_error(
        self,
        # pyre-ignore[11]
        session: ExportSession,
        example_inputs: List[Tuple[Tensor]],
        model_name: str,
        recipe_key: str,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> None:
        """Compare original model output with session output using tolerance."""
        quantized_model = session.get_stage_artifacts()[StageType.QUANTIZE].data[
            "forward"
        ]
        lowered_output = session.run_method("forward", *example_inputs)[0]
        quantized_output = quantized_model(*example_inputs[0])

        try:
            Tester._assert_outputs_equal(
                lowered_output, quantized_output, atol=atol, rtol=rtol
            )
            logging.info(
                f"Tolerance check passed for {model_name} with atol={atol}, rtol={rtol}"
            )
        except AssertionError as e:
            raise AssertionError(
                f"Model '{model_name}' Recipe: {recipe_key}, tolerance check failed"
            ) from e

    def _check_quantization_error(
        self,
        session: ExportSession,
        eager_model: nn.Module,
        example_inputs: List[Tuple[Tensor]],
        model_name: str,
        recipe_key: str,
        sqnr_threshold: float = 20.0,
    ) -> None:
        """Compare original model output with session output using SQNR."""
        eager_output = eager_model(*example_inputs[0])

        # get quantized model from session
        all_artifacts = session.get_stage_artifacts()
        quantized_model = all_artifacts[StageType.QUANTIZE].data["forward"]
        quantized_output = quantized_model(*example_inputs[0])

        error = compute_error(eager_output, quantized_output)
        logging.info(f"SQNR for {model_name}: {error} dB")
        self.assertTrue(
            error > sqnr_threshold,
            f"Model {model_name}, recipe: {recipe_key} SQNR check failed. Expected > {sqnr_threshold}, got {error}",
        )

    def _check_delegation_with_filecheck(self, session: ExportSession) -> None:
        """Check that the lowered module contains expected delegate calls."""
        all_artifacts = session.get_stage_artifacts()
        edge_program_manager = all_artifacts[StageType.TO_EDGE_TRANSFORM_AND_LOWER].data
        lowered_module = edge_program_manager.exported_program().module()

        # Check if model got lowered
        FileCheck().check("torch.ops.higher_order.executorch_call_delegate").run(
            lowered_module.code
        )

    # pyre-ignore
    @unittest.skipIf(
        not is_supported_platform_for_coreml_lowering(),
        "Skip test, coreml lowering not supported",
    )
    def test_ios_fp32_recipe_with_xnnpack_fallback(self) -> None:
        from executorch.export.target_recipes import get_ios_recipe

        # Linear ops skipped by coreml but handled by xnnpack
        model = self.model
        model.eval()

        example_inputs = [(torch.randn(2, 4), torch.randn(2, 4))]

        # Export using multi-backend target recipe with CoreML configured to skip linear operations
        recipe = get_ios_recipe(
            "ios-arm64-coreml-fp32",
            skip_ops_for_coreml_delegation=["aten.linear.default"],
        )

        # Export the model
        session = export(
            model=model, example_inputs=example_inputs, export_recipe=recipe
        )

        # Verify we can create executable
        executorch_program = session.get_executorch_program()
        # session.print_delegation_info()

        self.assertIsNotNone(
            executorch_program, "ExecuTorch program should not be None"
        )

        # Assert there is an execution plan
        self.assertTrue(len(executorch_program.execution_plan) == 1)

        # Check number of partitions created
        self.assertTrue(len(executorch_program.execution_plan[0].delegates) == 3)

        # First delegate backend is Xnnpack
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[0].id,
            "XnnpackBackend",
        )

        # Second delegate backend is CoreML
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[1].id,
            "CoreMLBackend",
        )

        # Third delegate backend is Xnnpack
        self.assertEqual(
            executorch_program.execution_plan[0].delegates[2].id,
            "XnnpackBackend",
        )

        et_runtime: Runtime = Runtime.get()
        backend_registry = et_runtime.backend_registry
        logging.info(
            f"backends registered: {et_runtime.backend_registry.registered_backend_names}"
        )
        if backend_registry.is_available(
            "CoreMLBackend"
        ) and backend_registry.is_available("XnnpackBackend"):
            logging.info("Running with CoreML and XNNPACK backends")
            et_output = session.run_method("forward", example_inputs[0])
            logging.info(f"et output {et_output}")

    def _test_model_with_target_recipes(
        self,
        model_name: str,
        recipe: ExportRecipe,
        expected_backend_name: str,
        eager_model: nn.Module,
        example_inputs: Tuple[Tensor],
        recipe_key: str,
        dynamic_shapes: Optional[Dict[str, Tuple[int, ...]]],
        atol: Optional[float] = 1e-1,
        rtol: Optional[float] = 1e-1,
        sqnr_threshold: Optional[int] = 20,
    ) -> None:
        """Test a model with a specific target recipe and expected backend."""
        logging.info(f"Testing model {model_name} with {expected_backend_name} backend")

        # Export with the provided recipe
        session = export(
            model=eager_model,
            example_inputs=[example_inputs],
            export_recipe=recipe,
            dynamic_shapes=dynamic_shapes,
        )
        logging.info(f"Exporting done for {model_name}-{recipe_key}")

        executorch_program = session.get_executorch_program()
        self.assertIsNotNone(
            executorch_program,
            f"ExecuTorch program should not be None for {expected_backend_name}",
        )

        # Check delegation for the expected backend
        self.check_delegated(executorch_program, [expected_backend_name])

        # Check number of partitions created
        self.check_num_partitions(executorch_program, 1)

        # Run the model if the backend is available
        et_runtime: Runtime = Runtime.get()
        backend_registry = et_runtime.backend_registry

        logging.info(
            f"backends registered: {et_runtime.backend_registry.registered_backend_names}"
        )

        if backend_registry.is_available(expected_backend_name):
            logging.info(f"Running with {expected_backend_name} backend")
            if atol is not None and rtol is not None:
                self._check_lowering_error(
                    session,
                    [example_inputs],
                    model_name,
                    recipe_key,
                    atol=atol,
                    rtol=rtol,
                )
                logging.info(
                    f"Accuracy checks passed for {model_name} with {expected_backend_name} with atol={atol}, rtol={rtol}"
                )

            # Test SQNR if specified
            if sqnr_threshold is not None:
                self._check_quantization_error(
                    session,
                    eager_model,
                    [example_inputs],
                    model_name,
                    recipe_key,
                    sqnr_threshold=sqnr_threshold,
                )

                logging.info(
                    f"SQNR check passed for {model_name} with {expected_backend_name} with sqnr={sqnr_threshold}"
                )

    @classmethod
    def _get_model_test_configs(
        cls,
    ) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float], Optional[int]]]]:
        """Get model-specific test configurations for different recipes."""
        # Format: {model_name: {target_recipe_name: (atol, rtol, sqnr_threshold)}}
        # If a model/recipe combination is present in this config, the model will be lowered for that recipe.
        # A value of `None` for any of atol, rtol, or sqnr_threshold means the corresponding accuracy check will be skipped after lowering.
        return {
            "linear": {
                "ios-arm64-coreml-fp16": (1e-3, 1e-3, 20),
                "ios-arm64-coreml-int8": (1e-2, 1e-2, 20),
                "android-arm64-snapdragon-fp16": (1e-3, 1e-3, None),
            },
            "add": {
                "ios-arm64-coreml-fp16": (1e-3, 1e-3, 20),
                "ios-arm64-coreml-int8": (1e-3, 1e-3, 20),
                "android-arm64-snapdragon-fp16": (1e-3, 1e-3, None),
            },
            "add_mul": {
                "ios-arm64-coreml-fp16": (1e-3, 1e-3, 20),
                "ios-arm64-coreml-int8": (1e-3, 1e-3, 20),
                "android-arm64-snapdragon-fp16": (1e-3, 1e-3, None),
            },
            "ic3": {
                "ios-arm64-coreml-fp16": (1e-1, 1.0, 20),
                "ios-arm64-coreml-int8": (None, None, None),
                "android-arm64-snapdragon-fp16": (5e-1, 1e-1, None),
            },
            "ic4": {
                "ios-arm64-coreml-fp16": (1e-1, 1e-1, 20),
                "ios-arm64-coreml-int8": (None, None, None),
                "android-arm64-snapdragon-fp16": (None, None, None),
            },
            "mv2": {
                "ios-arm64-coreml-fp16": (5e-2, 5e-2, 20),
                "ios-arm64-coreml-int8": (2e-1, 2e-1, 20),
                "android-arm64-snapdragon-fp16": (1e-2, 5e-2, None),
            },
            "mv3": {
                "ios-arm64-coreml-fp16": (2e-1, 2e-1, 20),
                "ios-arm64-coreml-int8": (None, None, None),
                "android-arm64-snapdragon-fp16": (None, None, None),
            },
            "resnet18": {
                "ios-arm64-coreml-fp16": (1e-1, 1e-1, 20),
                "ios-arm64-coreml-int8": (None, None, None),
                "android-arm64-snapdragon-fp16": (2e-1, 2e-1, None),
            },
            "resnet50": {
                "ios-arm64-coreml-fp16": (1e-2, 1e-2, 20),
                "ios-arm64-coreml-int8": (None, None, None),
                "android-arm64-snapdragon-fp16": (5e-1, 2e-1, None),
            },
            "vit": {
                "ios-arm64-coreml-fp16": (None, None, None),  # only lower
                "ios-arm64-coreml-int8": (None, None, None),  # only lower
                # Couldn't lower it to qnn
                # "android-arm64-snapdragon-fp16": (None, None, None),
            },
            "w2l": {
                "ios-arm64-coreml-fp16": (1e-2, 1e-2, 20),
                "ios-arm64-coreml-int8": (1e-1, 1e-1, 20),
                "android-arm64-snapdragon-fp16": (1e-2, 1e-2, None),
            },
        }

    @classmethod
    def _get_recipes(cls) -> Dict[str, Tuple[ExportRecipe, str]]:
        """Get available recipes with their configurations based on platform."""
        all_recipes: Dict[str, Tuple[ExportRecipe, str]] = {}

        # Add iOS recipes
        if is_supported_platform_for_coreml_lowering():
            from executorch.export.target_recipes import get_ios_recipe

            all_recipes["ios-arm64-coreml-fp16"] = (get_ios_recipe(), "CoreMLBackend")

            # ios-arm64-coreml-int8 requires CoreMLQuantizer which depends on
            # torch.ao.quantization.quantizer. This module has been migrated to
            # torchao and may not be available in all PyTorch versions.
            # TODO: https://github.com/pytorch/executorch/issues/16484
            # Update coremltools to use torchao.quantization.pt2e.quantizer
            # instead of the deprecated torch.ao.quantization.quantizer, then remove this try/except.
            try:
                all_recipes["ios-arm64-coreml-int8"] = (
                    get_ios_recipe("ios-arm64-coreml-int8"),
                    "CoreMLBackend",
                )
            except (ImportError, ModuleNotFoundError, ValueError) as e:
                logging.warning(
                    f"Skipping ios-arm64-coreml-int8 recipe (torch.ao.quantization.quantizer not available): {e}"
                )

        # Add android recipes
        if is_fbcode() and is_supported_platform_for_qnn_lowering():
            from executorch.export.target_recipes import get_android_recipe

            all_recipes["android-arm64-snapdragon-fp16"] = (
                get_android_recipe(),
                "QnnBackend",
            )

        return all_recipes

    def _run_model_with_recipe(
        self,
        model_name: str,
        recipe_key: str,
        eager_model: nn.Module,
        example_inputs: Tuple[Tensor],
        # pyre-ignore
        dynamic_shapes: Any,
    ) -> None:
        model_configs = self._get_model_test_configs()
        recipes = self._get_recipes()

        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not found in test configurations")

        if recipe_key not in recipes:
            raise ValueError(f"Recipe {recipe_key} not found in recipe configurations")

        recipe_tolerances = model_configs[model_name]

        if recipe_key not in recipe_tolerances:
            raise ValueError(f"Model {model_name} does not support recipe {recipe_key}")

        atol, rtol, sqnr_threshold = recipe_tolerances[recipe_key]
        recipe, expected_backend = recipes[recipe_key]

        with torch.no_grad():
            logging.info(f"Running model {model_name} with recipe {recipe_key}")
            self._test_model_with_target_recipes(
                model_name=model_name,
                recipe=recipe,
                expected_backend_name=expected_backend,
                eager_model=eager_model,
                example_inputs=example_inputs,
                dynamic_shapes=dynamic_shapes,
                recipe_key=recipe_key,
                atol=atol,
                rtol=rtol,
                sqnr_threshold=sqnr_threshold,
            )

    def _run_model_with_all_recipes(self, model_name: str) -> None:
        if model_name not in MODEL_NAME_TO_MODEL:
            self.skipTest(f"Model {model_name} not found in MODEL_NAME_TO_MODEL")
            return

        eager_model, example_inputs, _example_kwarg_inputs, dynamic_shapes = (
            EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
        )
        eager_model = eager_model.eval()

        recipes = self._get_recipes()
        model_configs = self._get_model_test_configs()

        try:
            # Pre-filter recipes to only those supported by the model
            supported_recipes = []
            for recipe_key in recipes.keys():
                if (
                    model_name in model_configs
                    and recipe_key in model_configs[model_name]
                ):
                    supported_recipes.append(recipe_key)

            if not supported_recipes:
                self.skipTest(f"Model {model_name} has no supported recipes")
                return

            for recipe_key in supported_recipes:
                with self.subTest(recipe=recipe_key):
                    self._run_model_with_recipe(
                        model_name,
                        recipe_key,
                        eager_model,
                        example_inputs,
                        dynamic_shapes,
                    )
        finally:
            # Clean up dog.jpg file if it exists
            if os.path.exists("dog.jpg"):
                os.remove("dog.jpg")

    def test_linear_model(self) -> None:
        """Test linear model with all applicable recipes."""
        self._run_model_with_all_recipes("linear")

    def test_add_model(self) -> None:
        """Test add model with all applicable recipes."""
        self._run_model_with_all_recipes("add")

    def test_add_mul_model(self) -> None:
        """Test add_mul model with all applicable recipes."""
        self._run_model_with_all_recipes("add_mul")

    def test_ic3_model(self) -> None:
        """Test ic3 model with all applicable recipes."""
        self._run_model_with_all_recipes("ic3")

    def test_ic4_model(self) -> None:
        """Test ic4 model with all applicable recipes."""
        self._run_model_with_all_recipes("ic4")

    def test_mv2_model(self) -> None:
        """Test mv2 model with all applicable recipes."""
        self._run_model_with_all_recipes("mv2")

    def test_mv3_model(self) -> None:
        """Test mv3 model with all applicable recipes."""
        self._run_model_with_all_recipes("mv3")

    def test_resnet18_model(self) -> None:
        """Test resnet18 model with all applicable recipes."""
        self._run_model_with_all_recipes("resnet18")

    def test_resnet50_model(self) -> None:
        """Test resnet50 model with all applicable recipes."""
        self._run_model_with_all_recipes("resnet50")

    def test_vit_model(self) -> None:
        """Test vit model with all applicable recipes."""
        self._run_model_with_all_recipes("vit")

    def test_w2l_model(self) -> None:
        """Test w2l model with all applicable recipes."""
        self._run_model_with_all_recipes("w2l")
