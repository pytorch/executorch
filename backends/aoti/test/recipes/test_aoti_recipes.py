import unittest

import torch

from executorch.backends.aoti.recipes.aoti_recipe_types import AOTIRecipeType
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.export import export, ExportRecipe, StageType
from torch.testing._internal.common_quantization import TestHelperModules


class TestAotiRecipes(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_basic_recipe(self) -> None:
        m_eager = TestHelperModules.TwoLinearModule().eval()
        example_inputs = [(torch.randn(9, 8),)]
        session = export(
            model=m_eager,
            example_inputs=example_inputs,
            export_recipe=ExportRecipe.get_recipe(AOTIRecipeType.FP32),
        )
        artifacts = session.get_stage_artifacts()
        aoti_artifacts = artifacts[StageType.AOTI_LOWERING]
        aoti_delegate_module = aoti_artifacts.data["forward"]
        with torch.inference_mode():
            eager_out = m_eager(*example_inputs[0])
            aoti_out = aoti_delegate_module(*example_inputs[0])

        self.assertTrue(torch.allclose(eager_out, aoti_out, atol=1e-3))

    def _test_model_with_factory(self, model_name: str) -> None:
        if model_name not in MODEL_NAME_TO_MODEL:
            self.skipTest(f"Model {model_name} not found in MODEL_NAME_TO_MODEL")
            return

        # Create model using factory
        model, example_inputs, _example_kwarg_inputs, dynamic_shapes = (
            EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[model_name])
        )
        model = model.eval()

        # Export with recipe
        session = export(
            model=model,
            example_inputs=[example_inputs],
            export_recipe=ExportRecipe.get_recipe(AOTIRecipeType.FP32),
            dynamic_shapes=dynamic_shapes,
        )

        artifacts = session.get_stage_artifacts()
        aoti_artifacts = artifacts[StageType.AOTI_LOWERING]
        aoti_delegate_module = aoti_artifacts.data["forward"]

        with torch.inference_mode():
            eager_out = model(*example_inputs)
            aoti_out = aoti_delegate_module(*example_inputs)

        self.assertTrue(torch.allclose(eager_out, aoti_out, atol=1e-3))

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
