# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, Optional, Sequence
from unittest.mock import Mock

import torch

from executorch.export.recipe import ExportRecipe, RecipeType
from executorch.export.recipe_provider import BackendRecipeProvider
from executorch.export.recipe_registry import recipe_registry


class TestRecipeType(RecipeType):
    FP32 = "fp32"
    INT8 = "int8"
    UNSUPPORTED = "unsupported"

    @classmethod
    def get_backend_name(cls) -> str:
        return "test_backend"


class AnotherTestRecipeType(RecipeType):
    DYNAMIC = "dynamic"

    @classmethod
    def get_backend_name(cls) -> str:
        return "another_backend"


class ConcreteBackendProvider(BackendRecipeProvider):
    def __init__(
        self, backend_name: str, supported_recipes: Sequence[RecipeType]
    ) -> None:
        self._backend_name = backend_name
        self._supported_recipes = supported_recipes
        self.last_kwargs: Optional[Dict[str, Any]] = None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def get_supported_recipes(self) -> Sequence[RecipeType]:
        return self._supported_recipes

    def create_recipe(
        self, recipe_type: RecipeType, **kwargs: Any
    ) -> Optional[ExportRecipe]:
        self.last_kwargs = kwargs
        if recipe_type in self._supported_recipes:
            return ExportRecipe(name=f"{self._backend_name}_{recipe_type.value}")
        return None


class TestExportRecipeGetRecipe(unittest.TestCase):

    def setUp(self) -> None:
        self.provider = ConcreteBackendProvider(
            "test_backend", [TestRecipeType.FP32, TestRecipeType.INT8]
        )
        recipe_registry.register_backend_recipe_provider(self.provider)

        self.another_provider = ConcreteBackendProvider(
            "another_backend", [AnotherTestRecipeType.DYNAMIC]
        )
        recipe_registry.register_backend_recipe_provider(self.another_provider)

    def tearDown(self) -> None:
        if recipe_registry._initialized:
            recipe_registry._providers.clear()

    def test_get_recipe_success(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_fp32")

    def test_get_recipe_unsupported_recipe_raises_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            ExportRecipe.get_recipe(TestRecipeType.UNSUPPORTED)

        error_message = str(context.exception)
        self.assertIn(
            "Recipe 'unsupported' not supported by 'test_backend'", error_message
        )
        self.assertIn("Supported: ['fp32', 'int8']", error_message)

    def test_get_recipe_unsupported_recipe_type_raises_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            # pyre-ignore[6]
            ExportRecipe.get_recipe("abc")

        error_message = str(context.exception)
        self.assertIn("Invalid recipe type:", error_message)

    def test_get_recipe_backend_name_extraction(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_fp32")

        result2 = ExportRecipe.get_recipe(AnotherTestRecipeType.DYNAMIC)
        self.assertIsNotNone(result2)
        self.assertEqual(result2.name, "another_backend_dynamic")

    def test_get_recipe_empty_kwargs(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32, **{})

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_fp32")

    def test_get_recipe_returns_correct_type(self) -> None:
        result = ExportRecipe.get_recipe(TestRecipeType.FP32)

        self.assertIsInstance(result, ExportRecipe)

    def test_get_recipe_with_kwargs_verification(self) -> None:
        """Test that kwargs are properly passed to recipe_registry.create_recipe"""
        kwargs = {"group_size": 32, "custom_kwarg": "val"}

        result = ExportRecipe.get_recipe(TestRecipeType.INT8, **kwargs)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_backend_int8")

        # Verify that the kwargs were passed to the backend provider's create_recipe method
        self.assertIsNotNone(self.provider.last_kwargs)
        self.assertEqual(self.provider.last_kwargs, kwargs)


class TestExportRecipeCombine(unittest.TestCase):
    def _lowering(self, **kwargs):
        from executorch.export.recipe import LoweringRecipe

        return LoweringRecipe(**kwargs)

    def test_combine_keeps_edge_compile_config_without_partitioners(self) -> None:
        # A recipe that lowers without delegating still needs its to_edge
        # config; it used to be dropped along with the whole LoweringRecipe.
        from executorch.exir import EdgeCompileConfig

        config = EdgeCompileConfig(_check_ir_validity=False)
        combined = ExportRecipe.combine(
            [
                ExportRecipe(
                    name="a", lowering_recipe=self._lowering(edge_compile_config=config)
                ),
                ExportRecipe(name="b"),
            ]
        )
        assert combined.lowering_recipe is not None
        self.assertFalse(
            combined.lowering_recipe.edge_compile_config._check_ir_validity
        )
        # Carried by copy, so mutating the combination cannot reach back into
        # the provider's own config -- including through the mutable lists,
        # which a shallow copy would still share.
        combined.lowering_recipe.edge_compile_config.preserve_ops.append(
            torch.ops.aten.linear.default
        )
        self.assertEqual(config.preserve_ops, [])

    def test_combine_rejects_conflicting_edge_compile_configs(self) -> None:
        # Picking one by position would decide the emitted graph by argument
        # order: preserve_ops that one backend requires would silently vanish.
        from executorch.exir import EdgeCompileConfig

        recipes = [
            ExportRecipe(
                name="a",
                lowering_recipe=self._lowering(
                    edge_compile_config=EdgeCompileConfig(
                        preserve_ops=[torch.ops.aten.linear.default]
                    )
                ),
            ),
            ExportRecipe(
                name="b",
                lowering_recipe=self._lowering(
                    edge_compile_config=EdgeCompileConfig(preserve_ops=[])
                ),
            ),
        ]
        for ordering in (recipes, list(reversed(recipes))):
            with self.assertRaisesRegex(ValueError, "edge_compile_configs") as cm:
                ExportRecipe.combine(ordering)
            # The message has to say which recipes disagree, and how.
            self.assertIn("a ", str(cm.exception))
            self.assertIn("b ", str(cm.exception))
            self.assertIn("aten.linear.default", str(cm.exception))

    def test_combine_accepts_distinct_but_equal_configs(self) -> None:
        # Every provider builds a fresh config object, so this -- not the
        # shared-object case -- is what a real multi-backend combination looks
        # like. Comparing by identity would reject all of them.
        from executorch.exir import EdgeCompileConfig

        combined = ExportRecipe.combine(
            [
                ExportRecipe(
                    name=n,
                    lowering_recipe=self._lowering(
                        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False)
                    ),
                )
                for n in ("a", "b")
            ]
        )
        assert combined.lowering_recipe is not None
        self.assertFalse(
            combined.lowering_recipe.edge_compile_config._check_ir_validity
        )

    def test_combine_ignores_preserve_ops_ordering(self) -> None:
        # preserve_ops says which ops to keep, not in what order.
        from executorch.exir import EdgeCompileConfig

        ops = [torch.ops.aten.linear.default, torch.ops.aten.silu.default]
        combined = ExportRecipe.combine(
            [
                ExportRecipe(
                    name="a",
                    lowering_recipe=self._lowering(
                        edge_compile_config=EdgeCompileConfig(preserve_ops=ops)
                    ),
                ),
                ExportRecipe(
                    name="b",
                    lowering_recipe=self._lowering(
                        edge_compile_config=EdgeCompileConfig(
                            preserve_ops=list(reversed(ops))
                        )
                    ),
                ),
            ]
        )
        assert combined.lowering_recipe is not None
        self.assertCountEqual(
            combined.lowering_recipe.edge_compile_config.preserve_ops, ops
        )

    def test_combine_without_lowering_inputs_produces_none(self) -> None:
        combined = ExportRecipe.combine(
            [ExportRecipe(name="a"), ExportRecipe(name="b")]
        )
        self.assertIsNone(combined.lowering_recipe)

    def test_combine_keeps_partitioners(self) -> None:
        from executorch.exir.backend.partitioner import Partitioner

        a, b = Mock(spec=Partitioner), Mock(spec=Partitioner)
        combined = ExportRecipe.combine(
            [
                ExportRecipe(
                    name="a", lowering_recipe=self._lowering(partitioners=[a])
                ),
                ExportRecipe(
                    name="b", lowering_recipe=self._lowering(partitioners=[b])
                ),
            ]
        )
        assert combined.lowering_recipe is not None
        self.assertEqual(combined.lowering_recipe.partitioners, [a, b])

    def test_combine_rejects_pipeline_stages(self) -> None:
        from executorch.export.types import StageType

        # Unnamed recipes fall back to their position; named ones are named.
        with self.assertRaisesRegex(ValueError, r"pipeline_stages.*recipes\[0\]"):
            ExportRecipe.combine(
                [
                    ExportRecipe(
                        pipeline_stages=[StageType.TO_EDGE_TRANSFORM_AND_LOWER]
                    ),
                    ExportRecipe(name="b"),
                ]
            )
        with self.assertRaisesRegex(ValueError, r"pipeline_stages.*'stagey'"):
            ExportRecipe.combine(
                [
                    ExportRecipe(
                        name="stagey",
                        pipeline_stages=[StageType.TO_EDGE_TRANSFORM_AND_LOWER],
                    ),
                    ExportRecipe(name="b"),
                ]
            )

    def test_combine_keeps_edge_transform_passes(self) -> None:
        # QNN supplies these and is combined with XNNPACK by target_recipes.
        first = lambda name, ep: []  # noqa: E731
        second = lambda name, ep: []  # noqa: E731
        combined = ExportRecipe.combine(
            [
                ExportRecipe(
                    name="a",
                    lowering_recipe=self._lowering(edge_transform_passes=[first]),
                ),
                ExportRecipe(
                    name="b",
                    lowering_recipe=self._lowering(edge_transform_passes=[second]),
                ),
            ]
        )
        assert combined.lowering_recipe is not None
        self.assertEqual(
            combined.lowering_recipe.edge_transform_passes, [first, second]
        )
