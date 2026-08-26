# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Cortex-M ExportRecipe provider."""

import unittest

import torch

from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.recipes.cortex_m_recipe_provider import (
    CortexMRecipeProvider,
)
from executorch.backends.cortex_m.recipes.cortex_m_recipe_types import CortexMRecipeType
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.export import export, ExportRecipe, recipe_registry, StageType
from executorch.export.export import ExportSession


class ConvRelu(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


_PROVIDER_LOGGER = "executorch.backends.cortex_m.recipes.cortex_m_recipe_provider"


def _executorch_op_names(program) -> list[str]:
    return [op.name for op in program.execution_plan[0].operators]


def _target_config(recipe: ExportRecipe) -> CortexMTargetConfig:
    assert recipe.lowering_recipe is not None
    passes = recipe.lowering_recipe.op_backends
    assert passes is not None
    return passes[0].target_config  # type: ignore[attr-defined]


def _export(recipe: ExportRecipe, model: torch.nn.Module, example_inputs: tuple):
    session = export(
        model=model,
        example_inputs=[example_inputs],
        export_recipe=recipe,
    )
    return session.get_executorch_program()


class _CortexMRecipeTestCase(unittest.TestCase):
    """Re-registers the Cortex-M provider before each test.

    Registration happens when ``backends.cortex_m.recipes`` is imported, so it
    cannot repeat: the module is already loaded. Other suites in the same
    process clear the singleton registry in teardown, and under ``pytest -n``
    their tests can interleave with these.

    """

    def setUp(self) -> None:
        super().setUp()
        recipe_registry.register_backend_recipe_provider(CortexMRecipeProvider())


class TestCortexMRecipeConstruction(_CortexMRecipeTestCase):
    def test_int8_recipe_shape(self) -> None:
        recipe = ExportRecipe.get_recipe(CortexMRecipeType.INT8)
        self.assertEqual(recipe.name, "cortex_m_int8")

        assert recipe.lowering_recipe is not None
        self.assertIsNone(recipe.lowering_recipe.partitioners)
        self.assertIsNotNone(recipe.lowering_recipe.edge_compile_config)

        # The session schedules the stage from the lowering recipe.
        self.assertTrue(recipe.lowering_recipe.op_backends)
        stages = ExportSession(
            model=ConvRelu().eval(),
            example_inputs=[(torch.randn(1, 3, 8, 8),)],
            export_recipe=recipe,
        )._pipeline_stages
        self.assertGreater(
            stages.index(StageType.EDGE_PROGRAM_MANAGER_TRANSFORM),
            stages.index(StageType.TO_EDGE_TRANSFORM_AND_LOWER),
        )

    def test_isa_overrides_the_derived_backend(self) -> None:
        recipe = ExportRecipe.get_recipe(
            CortexMRecipeType.INT8, target="cortex-m55", isa=cmsis_nn.Backend.DSP
        )
        self.assertEqual(_target_config(recipe).backend, cmsis_nn.Backend.DSP)

    def test_a_documented_kwarg_does_not_warn(self) -> None:
        # Shrinking the allow-list makes target= a silent no-op with a scary
        # log, and nothing else in the suite notices.
        with self.assertNoLogs(_PROVIDER_LOGGER, level="WARNING"):
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, target="cortex-m33")
        with self.assertNoLogs(_PROVIDER_LOGGER, level="WARNING"):
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, isa=cmsis_nn.Backend.DSP)

    def test_a_wrongly_typed_kwarg_raises(self) -> None:
        # Unknown keys warn because a combined recipe fans the same kwargs at
        # every backend; a known key with the wrong type is the caller's error.
        with self.assertRaisesRegex(ValueError, "'isa' must be"):
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, isa="MVE")
        with self.assertRaisesRegex(ValueError, "'target' must be"):
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, target=55)
        # A falsy target must not be read as "unset" and replaced by the default.
        with self.assertRaisesRegex(ValueError, "'target' must be"):
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, target=0)

    def test_unexpected_kwarg_warns(self) -> None:
        with self.assertLogs(_PROVIDER_LOGGER, level="WARNING") as logs:
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, cpu="cortex-m55")
        # Naming the offender is the point; listing the allowed set is not.
        self.assertIn("'cpu'", "".join(logs.output))


class TestCortexMRecipeExport(_CortexMRecipeTestCase):
    def test_channels_last_input_lowers_to_cmsis_kernels(self) -> None:
        program = _export(
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, target="cortex-m55"),
            ConvRelu().eval(),
            (torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),),
        )
        self.assertEqual(
            _executorch_op_names(program),
            [
                "cortex_m::quantize_per_tensor",
                "cortex_m::quantized_conv2d",
                "cortex_m::dequantize_per_tensor",
            ],
        )


class TestCortexMRecipeTargetReachesLowering(_CortexMRecipeTestCase):
    """The target has to reach CortexMPassManager, not just the recipe object.

    ``quantized_linear`` encodes its weights differently for MVE, so a target
    that is parsed and then dropped ships an artifact whose kernel reads an
    argument that is not there.
    """

    class Linear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    def _linear_parameter_count(self, target: str, **kwargs) -> int:
        program = _export(
            ExportRecipe.get_recipe(CortexMRecipeType.INT8, target=target, **kwargs),
            self.Linear().eval(),
            (torch.randn(1, 8),),
        )
        self.assertIn("cortex_m::quantized_linear", _executorch_op_names(program))
        return len(program.execution_plan[0].values)

    def test_mve_and_dsp_targets_differ(self) -> None:
        self.assertNotEqual(
            self._linear_parameter_count("cortex-m55"),
            self._linear_parameter_count("cortex-m33"),
        )

    def test_isa_override_reaches_lowering(self) -> None:
        # An M55 built without MVE has to lower like a DSP core, not merely
        # record the isa on the recipe object.
        self.assertEqual(
            self._linear_parameter_count("cortex-m55", isa=cmsis_nn.Backend.DSP),
            self._linear_parameter_count("cortex-m33"),
        )


class TestCortexMLoweringLeavesItsInputAlone(_CortexMRecipeTestCase):
    """The manager handed to the transform is also the previous stage's
    recorded artifact, so rewriting it in place would destroy the record of
    what the partitioner produced."""

    def _session(self, model, inputs):
        from executorch.export.export import ExportSession

        session = ExportSession(
            model=model,
            example_inputs=[inputs],
            export_recipe=ExportRecipe.get_recipe(CortexMRecipeType.INT8),
        )
        session.export()
        return session

    def test_previous_stage_artifact_survives(self) -> None:
        session = self._session(
            ConvRelu().eval(),
            (torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),),
        )
        artifacts = session.get_stage_artifacts()
        before = artifacts[StageType.TO_EDGE_TRANSFORM_AND_LOWER].data
        after = artifacts[StageType.EDGE_PROGRAM_MANAGER_TRANSFORM].data
        self.assertIsNot(before, after)
        self.assertIn(
            "cortex_m",
            str([n.target for n in after.exported_program().graph.nodes]),
        )


class TestCortexMRecipeMultiMethod(_CortexMRecipeTestCase):
    """The transform loops over methods; a program with two exercises it."""

    def test_every_method_is_lowered(self) -> None:
        from executorch.export import export

        conv_inputs = (torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last),)
        linear_inputs = (torch.randn(1, 8),)

        class Linear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        session = export(
            model={"forward": ConvRelu().eval(), "classify": Linear().eval()},
            example_inputs={"forward": [conv_inputs], "classify": [linear_inputs]},
            export_recipe=ExportRecipe.get_recipe(CortexMRecipeType.INT8),
        )
        program = session.get_executorch_program()
        plans = {plan.name: plan for plan in program.execution_plan}
        self.assertEqual(set(plans), {"forward", "classify"})

        # Naming the kernel matters: the boundary quantize/dequantize pair is
        # itself cortex_m::, so "any cortex_m op" passes even when the real
        # operator stayed on the portable float kernel.
        expected = {
            "forward": "cortex_m::quantized_conv2d",
            "classify": "cortex_m::quantized_linear",
        }
        for name, plan in plans.items():
            with self.subTest(method=name):
                ops = [op.name for op in plan.operators]
                self.assertIn(expected[name], ops)


if __name__ == "__main__":
    unittest.main()
