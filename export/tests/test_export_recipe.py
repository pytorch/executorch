# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import Mock

import torch

from executorch.export.recipe import (
    ExportRecipe,
    LoweringRecipe,
    Mode,
    QuantizationRecipe,
    RecipeType,
)
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
            self.assertIn("preserve_ops", str(cm.exception))
            self.assertIn("a=", str(cm.exception))
            self.assertIn("b=", str(cm.exception))
            self.assertIn("aten.linear.default", str(cm.exception))

    def test_combine_names_the_field_that_conflicts(self) -> None:
        # A summary of hand-picked fields prints identically for two configs
        # that differ elsewhere, so the error names no cause at all.
        from executorch.exir import EdgeCompileConfig

        with self.assertRaises(ValueError) as cm:
            ExportRecipe.combine(
                [
                    ExportRecipe(
                        name=name,
                        lowering_recipe=self._lowering(
                            edge_compile_config=EdgeCompileConfig(_skip_dim_order=skip)
                        ),
                    )
                    for name, skip in (("a", True), ("b", False))
                ]
            )
        self.assertIn("_skip_dim_order (a=True, b=False)", str(cm.exception))
        # Fields they agree on are noise that hides the one that matters.
        self.assertNotIn("preserve_ops", str(cm.exception))

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


# ---------------------------------------------------------------------------
# Helpers shared by combine-recipe tests
# ---------------------------------------------------------------------------


def _make_pass(name: str, call_log: List[str]):
    """Return a graph-module pass that appends *name* to *call_log*."""

    def pass_fn(m):
        call_log.append(name)
        return m

    return pass_fn


class TestCombineRecipesEmpty(unittest.TestCase):
    def test_empty_recipes_raises(self) -> None:
        with self.assertRaises(ValueError):
            ExportRecipe.combine([])


class TestCombineRecipesSingleRecipe(unittest.TestCase):
    def test_single_recipe_returned_unchanged(self) -> None:
        recipe = ExportRecipe(name="solo")
        result = ExportRecipe.combine([recipe])
        self.assertIs(result, recipe)


class TestCombineRecipesScalarFields(unittest.TestCase):
    """Fields that must be identical across all combined recipes."""

    def test_conflicting_strict_raises(self) -> None:
        r1 = ExportRecipe(name="a", strict=True)
        r2 = ExportRecipe(name="b", strict=False)
        with self.assertRaises(ValueError) as cm:
            ExportRecipe.combine([r1, r2])
        self.assertIn("strict", str(cm.exception))

    def test_conflicting_mode_raises(self) -> None:
        r1 = ExportRecipe(name="a", mode=Mode.DEBUG)
        r2 = ExportRecipe(name="b", mode=Mode.RELEASE)
        with self.assertRaises(ValueError) as cm:
            ExportRecipe.combine([r1, r2])
        self.assertIn("mode", str(cm.exception))

    def test_conflicting_source_transform_in_place_raises(self) -> None:
        r1 = ExportRecipe(name="a", source_transform_in_place=True)
        r2 = ExportRecipe(name="b", source_transform_in_place=False)
        with self.assertRaises(ValueError) as cm:
            ExportRecipe.combine([r1, r2])
        self.assertIn("source_transform_in_place", str(cm.exception))

    def test_agreeing_scalar_fields_are_preserved(self) -> None:
        r1 = ExportRecipe(
            name="a", strict=False, mode=Mode.DEBUG, source_transform_in_place=True
        )
        r2 = ExportRecipe(
            name="b", strict=False, mode=Mode.DEBUG, source_transform_in_place=True
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertFalse(result.strict)
        self.assertEqual(result.mode, Mode.DEBUG)
        self.assertTrue(result.source_transform_in_place)

    def test_name_is_joined_from_input_recipe_names(self) -> None:
        r1 = ExportRecipe(name="backend_a")
        r2 = ExportRecipe(name="backend_b")
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.name, "backend_a_backend_b")

    def test_custom_recipe_name_is_used(self) -> None:
        r1 = ExportRecipe(name="a")
        r2 = ExportRecipe(name="b")
        result = ExportRecipe.combine([r1, r2], recipe_name="custom_name")
        self.assertEqual(result.name, "custom_name")


class TestCombineRecipesAtenTransformPasses(unittest.TestCase):
    def test_aten_transform_passes_merged(self) -> None:
        pass1 = Mock()
        pass2 = Mock()
        r1 = ExportRecipe(name="a", aten_transform_passes=[pass1])
        r2 = ExportRecipe(name="b", aten_transform_passes=[pass2])
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.aten_transform_passes, [pass1, pass2])

    def test_aten_transform_passes_none_when_both_empty(self) -> None:
        r1 = ExportRecipe(name="a")
        r2 = ExportRecipe(name="b")
        result = ExportRecipe.combine([r1, r2])
        self.assertIsNone(result.aten_transform_passes)

    def test_aten_transform_passes_one_side_none(self) -> None:
        pass1 = Mock()
        r1 = ExportRecipe(name="a", aten_transform_passes=[pass1])
        r2 = ExportRecipe(name="b")
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.aten_transform_passes, [pass1])


class TestCombineRecipesLowering(unittest.TestCase):
    def test_partitioners_merged(self) -> None:
        p1 = Mock()
        p2 = Mock()
        r1 = ExportRecipe(name="a", lowering_recipe=LoweringRecipe(partitioners=[p1]))
        r2 = ExportRecipe(name="b", lowering_recipe=LoweringRecipe(partitioners=[p2]))
        result = ExportRecipe.combine([r1, r2])
        self.assertIsNotNone(result.lowering_recipe)
        self.assertEqual(result.lowering_recipe.partitioners, [p1, p2])

    def test_edge_transform_passes_merged(self) -> None:
        pass1 = Mock()
        pass2 = Mock()
        r1 = ExportRecipe(
            name="a", lowering_recipe=LoweringRecipe(edge_transform_passes=[pass1])
        )
        r2 = ExportRecipe(
            name="b", lowering_recipe=LoweringRecipe(edge_transform_passes=[pass2])
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.lowering_recipe.edge_transform_passes, [pass1, pass2])

    def test_edge_manager_transform_passes_merged(self) -> None:
        pass1 = Mock()
        pass2 = Mock()
        r1 = ExportRecipe(
            name="a",
            lowering_recipe=LoweringRecipe(edge_manager_transform_passes=[pass1]),
        )
        r2 = ExportRecipe(
            name="b",
            lowering_recipe=LoweringRecipe(edge_manager_transform_passes=[pass2]),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(
            result.lowering_recipe.edge_manager_transform_passes, [pass1, pass2]
        )

    def test_lowering_recipe_none_when_nothing_contributed(self) -> None:
        r1 = ExportRecipe(name="a")
        r2 = ExportRecipe(name="b")
        result = ExportRecipe.combine([r1, r2])
        self.assertIsNone(result.lowering_recipe)

    def test_edge_compile_config_taken_from_first_recipe_with_one(self) -> None:
        from executorch.exir.capture import EdgeCompileConfig

        config = EdgeCompileConfig()
        r1 = ExportRecipe(name="a")
        r2 = ExportRecipe(
            name="b",
            lowering_recipe=LoweringRecipe(
                partitioners=[Mock()], edge_compile_config=config
            ),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertIs(result.lowering_recipe.edge_compile_config, config)


class TestCombineRecipesQuantization(unittest.TestCase):
    def test_quantizers_merged(self) -> None:
        q1 = Mock()
        q2 = Mock()
        r1 = ExportRecipe(
            name="a", quantization_recipe=QuantizationRecipe(quantizers=[q1])
        )
        r2 = ExportRecipe(
            name="b", quantization_recipe=QuantizationRecipe(quantizers=[q2])
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertIsNotNone(result.quantization_recipe)
        self.assertEqual(result.quantization_recipe.quantizers, [q1, q2])

    def test_ao_quantization_configs_merged(self) -> None:
        from executorch.export.recipe import AOQuantizationConfig
        from torchao.core.config import AOBaseConfig

        cfg1 = AOQuantizationConfig(ao_base_config=Mock(spec=AOBaseConfig))
        cfg2 = AOQuantizationConfig(ao_base_config=Mock(spec=AOBaseConfig))
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(ao_quantization_configs=[cfg1]),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(ao_quantization_configs=[cfg2]),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(
            result.quantization_recipe.ao_quantization_configs, [cfg1, cfg2]
        )

    def test_quantization_recipe_none_when_nothing_contributed(self) -> None:
        r1 = ExportRecipe(name="a")
        r2 = ExportRecipe(name="b")
        result = ExportRecipe.combine([r1, r2])
        self.assertIsNone(result.quantization_recipe)

    def test_conflicting_is_qat_raises(self) -> None:
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(quantizers=[Mock()], is_qat=True),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(quantizers=[Mock()], is_qat=False),
        )
        with self.assertRaises(ValueError) as cm:
            ExportRecipe.combine([r1, r2])
        self.assertIn("is_qat", str(cm.exception))

    def test_agreeing_is_qat_preserved(self) -> None:
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(quantizers=[Mock()], is_qat=True),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(quantizers=[Mock()], is_qat=True),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertTrue(result.quantization_recipe.is_qat)

    def test_two_train_fns_raises(self) -> None:
        fn1 = Mock()
        fn2 = Mock()
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], is_qat=True, train_fn=fn1
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], is_qat=True, train_fn=fn2
            ),
        )
        with self.assertRaises(ValueError) as cm:
            ExportRecipe.combine([r1, r2])
        self.assertIn("train_fn", str(cm.exception))

    def test_single_train_fn_preserved(self) -> None:
        fn = Mock()
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], is_qat=True, train_fn=fn
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(quantizers=[Mock()], is_qat=True),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertIs(result.quantization_recipe.train_fn, fn)

    def test_single_calibration_inputs_fn_preserved(self) -> None:
        fn = Mock(return_value=[(1,), (2,)])
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], calibration_inputs_fn=fn
            ),
        )
        r2 = ExportRecipe(
            name="b", quantization_recipe=QuantizationRecipe(quantizers=[Mock()])
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertIs(result.quantization_recipe.calibration_inputs_fn, fn)

    def test_two_calibration_inputs_fns_chained(self) -> None:
        fn1 = Mock(return_value=[(1,), (2,)])
        fn2 = Mock(return_value=[(3,), (4,)])
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], calibration_inputs_fn=fn1
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], calibration_inputs_fn=fn2
            ),
        )
        result = ExportRecipe.combine([r1, r2])
        combined_fn = result.quantization_recipe.calibration_inputs_fn
        self.assertIsNotNone(combined_fn)
        # Each factory must be called exactly once when the combined factory is consumed.
        all_inputs = list(combined_fn())
        fn1.assert_called_once_with()
        fn2.assert_called_once_with()
        self.assertEqual(all_inputs, [(1,), (2,), (3,), (4,)])

    def test_three_calibration_inputs_fns_chained_in_order(self) -> None:
        fn1 = Mock(return_value=[(1,)])
        fn2 = Mock(return_value=[(2,)])
        fn3 = Mock(return_value=[(3,)])
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], calibration_inputs_fn=fn1
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], calibration_inputs_fn=fn2
            ),
        )
        r3 = ExportRecipe(
            name="c",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], calibration_inputs_fn=fn3
            ),
        )
        result = ExportRecipe.combine([r1, r2, r3])
        combined_fn = result.quantization_recipe.calibration_inputs_fn
        self.assertEqual(list(combined_fn()), [(1,), (2,), (3,)])

    def test_no_calibration_inputs_fn_stays_none(self) -> None:
        r1 = ExportRecipe(
            name="a", quantization_recipe=QuantizationRecipe(quantizers=[Mock()])
        )
        r2 = ExportRecipe(
            name="b", quantization_recipe=QuantizationRecipe(quantizers=[Mock()])
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertIsNone(result.quantization_recipe.calibration_inputs_fn)

    def test_pre_prepare_passes_merged(self) -> None:
        log: List[str] = []
        p1 = _make_pass("pre_a", log)
        p2 = _make_pass("pre_b", log)
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], pre_prepare_passes=[p1]
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], pre_prepare_passes=[p2]
            ),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.quantization_recipe.pre_prepare_passes, [p1, p2])

    def test_post_prepare_passes_merged(self) -> None:
        p1 = Mock()
        p2 = Mock()
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], post_prepare_passes=[p1]
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], post_prepare_passes=[p2]
            ),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.quantization_recipe.post_prepare_passes, [p1, p2])

    def test_pre_convert_passes_merged(self) -> None:
        p1 = Mock()
        p2 = Mock()
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], pre_convert_passes=[p1]
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], pre_convert_passes=[p2]
            ),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.quantization_recipe.pre_convert_passes, [p1, p2])

    def test_post_convert_passes_merged(self) -> None:
        p1 = Mock()
        p2 = Mock()
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], post_convert_passes=[p1]
            ),
        )
        r2 = ExportRecipe(
            name="b",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], post_convert_passes=[p2]
            ),
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.quantization_recipe.post_convert_passes, [p1, p2])

    def test_all_pass_lists_none_when_nothing_contributed(self) -> None:
        r1 = ExportRecipe(
            name="a", quantization_recipe=QuantizationRecipe(quantizers=[Mock()])
        )
        r2 = ExportRecipe(
            name="b", quantization_recipe=QuantizationRecipe(quantizers=[Mock()])
        )
        result = ExportRecipe.combine([r1, r2])
        qr = result.quantization_recipe
        self.assertIsNone(qr.pre_prepare_passes)
        self.assertIsNone(qr.post_prepare_passes)
        self.assertIsNone(qr.pre_convert_passes)
        self.assertIsNone(qr.post_convert_passes)

    def test_pass_lists_preserved_when_only_one_recipe_contributes(self) -> None:
        p = Mock()
        r1 = ExportRecipe(
            name="a",
            quantization_recipe=QuantizationRecipe(
                quantizers=[Mock()], pre_prepare_passes=[p]
            ),
        )
        r2 = ExportRecipe(
            name="b", quantization_recipe=QuantizationRecipe(quantizers=[Mock()])
        )
        result = ExportRecipe.combine([r1, r2])
        self.assertEqual(result.quantization_recipe.pre_prepare_passes, [p])
