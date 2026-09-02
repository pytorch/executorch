# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Arm ExportRecipe provider.

Building a recipe only assembles a compile spec, a quantizer and a partitioner,
so no test here needs the model converter or an FVP, and only the Ethos-U
round-trip needs Vela. Class and method names route each test to exactly one of
the existing target-less, TOSA, U55 and VGF suites; see the ``-k`` filters in
``backends/arm/test/test_arm_backend.sh``.

"""

# pyre-strict

import unittest
from typing import Any, Optional

import torch

from executorch.backends.arm.recipes.arm_recipe_provider import ArmRecipeProvider
from executorch.backends.arm.recipes.arm_recipe_types import ARM_BACKEND, ArmRecipeType
from executorch.export import ExportRecipe, recipe_registry, StageType
from executorch.export.export import ExportSession


_PROVIDER_LOGGER = "executorch.backends.arm.recipes.arm_recipe_provider"

try:
    import ethosu.vela.architecture_features  # type: ignore  # noqa: F401

    _VELA_INSTALLED = True
except ImportError:
    # The target-less CI job installs the Arm deps without Vela, and a recipe
    # has to build there; only the accelerator check needs it.
    _VELA_INSTALLED = False


class _AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class _ConvReluModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def _compile_spec_value(partitioner: Any, key: str) -> Optional[str]:
    for spec in partitioner.delegation_spec.compile_specs:
        if spec.key == key:
            value = spec.value
            return value.decode() if isinstance(value, (bytes, bytearray)) else value
    return None


def _backend_id(recipe: ExportRecipe) -> str:
    return _first_partitioner(recipe).delegation_spec.backend_id


def _first_partitioner(recipe: ExportRecipe) -> Any:
    assert recipe.lowering_recipe is not None
    # Arm recipes partition every method the same way, so the list form rather
    # than the per-method dict `LoweringRecipe` also accepts.
    partitioners = recipe.lowering_recipe.partitioners
    assert isinstance(partitioners, list) and partitioners
    return partitioners[0]


def _global_config(recipe: ExportRecipe) -> Any:
    assert recipe.quantization_recipe is not None
    assert recipe.quantization_recipe.quantizers is not None
    return recipe.quantization_recipe.quantizers[0].global_config  # type: ignore[attr-defined]


def _input_activation_dtype(recipe: ExportRecipe) -> Optional[torch.dtype]:
    config = _global_config(recipe)
    if config is None or config.input_activation is None:
        return None
    return config.input_activation.dtype


class _ArmRecipeTestCase(unittest.TestCase):
    """Re-registers the Arm provider before each test.

    Registration happens when ``backends.arm.recipes`` is imported, so it cannot
    repeat: the module is already loaded. Other suites in the same process clear
    the singleton registry in teardown, and under ``pytest -n`` their tests can
    interleave with these.

    """

    def setUp(self) -> None:
        super().setUp()
        recipe_registry.register_backend_recipe_provider(ArmRecipeProvider())


class TestArmRecipeRegistration(_ArmRecipeTestCase):
    def test_backend_registered(self) -> None:
        self.assertIn(ARM_BACKEND, recipe_registry.list_backends())

    def test_supported_recipes_match_enum(self) -> None:
        # Catches an enum member added but never wired into a target table.
        supported = recipe_registry.get_supported_recipes(ARM_BACKEND)
        self.assertEqual(set(supported), set(ArmRecipeType))

    def test_unknown_recipe_returns_none(self) -> None:
        from executorch.export import RecipeType

        class _StubRecipeType(RecipeType):
            FOO = "stub_foo"

            @classmethod
            def get_backend_name(cls) -> str:
                return "stub"

        self.assertIsNone(ArmRecipeProvider().create_recipe(_StubRecipeType.FOO))


class TestTosaRecipes(_ArmRecipeTestCase):
    def test_tosa_construction(self) -> None:
        cases = [
            (ArmRecipeType.TOSA_FP, "arm_tosa_fp", "TOSA-1.0+FP", None),
            (ArmRecipeType.TOSA_INT8, "arm_tosa_int8", "TOSA-1.0+INT", torch.int8),
            (
                ArmRecipeType.TOSA_A16W8,
                "arm_tosa_a16w8",
                "TOSA-1.0+INT+int16",
                torch.int16,
            ),
        ]
        for recipe_type, expected_name, expected_spec, expected_act_dtype in cases:
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertEqual(recipe.name, expected_name)
                # A VGF spec here would emit a container TOSA cannot consume.
                self.assertEqual(_backend_id(recipe), "TOSABackend")
                self.assertEqual(
                    _compile_spec_value(_first_partitioner(recipe), "tosa_spec"),
                    expected_spec,
                )
                if expected_act_dtype is None:
                    self.assertIsNone(recipe.quantization_recipe)
                else:
                    self.assertEqual(
                        _input_activation_dtype(recipe), expected_act_dtype
                    )

    def test_weights_are_per_channel(self) -> None:
        for recipe_type in (ArmRecipeType.TOSA_INT8, ArmRecipeType.TOSA_A16W8):
            with self.subTest(recipe_type=recipe_type):
                weight = _global_config(ExportRecipe.get_recipe(recipe_type)).weight
                self.assertEqual(weight.qscheme, torch.per_channel_symmetric)

    def test_unexpected_kwarg_warns(self) -> None:
        with self.assertLogs(_PROVIDER_LOGGER, level="WARNING"):
            ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8, foo=1)


class TestVgfRecipes(_ArmRecipeTestCase):
    """Named without the ``_vgf_`` token that routes tests to the VKML suite:

    constructing a compile spec needs no model converter, so these belong in the
    target-less suite that runs on every PR.

    """

    def test_construction(self) -> None:
        cases = [
            (ArmRecipeType.VGF_FP, "arm_vgf_fp", "TOSA-1.0+FP", None),
            (ArmRecipeType.VGF_INT8, "arm_vgf_int8", "TOSA-1.0+INT", torch.int8),
        ]
        for recipe_type, expected_name, expected_spec, expected_act_dtype in cases:
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertEqual(recipe.name, expected_name)
                self.assertEqual(
                    _compile_spec_value(_first_partitioner(recipe), "tosa_spec"),
                    expected_spec,
                )
                # A TOSA spec here would emit a flatbuffer VKML cannot load.
                self.assertEqual(_backend_id(recipe), "VgfBackend")
                self.assertEqual(
                    _compile_spec_value(_first_partitioner(recipe), "output_format"),
                    "vgf",
                )
                if expected_act_dtype is None:
                    self.assertIsNone(recipe.quantization_recipe)
                else:
                    self.assertEqual(
                        _input_activation_dtype(recipe), expected_act_dtype
                    )

    def test_keeps_quantized_decomposed_ops(self) -> None:
        # VGF consumes the quantized_decomposed QDQ ops, so ReplaceQuantNodesPass
        # must not run; see _apply_replace_quant_nodes in aot_arm_compiler.py.
        for recipe_type in (ArmRecipeType.VGF_INT8, ArmRecipeType.VGF_FP):
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                assert recipe.lowering_recipe is not None
                self.assertIsNone(recipe.lowering_recipe.op_backends)


class TestEthosURecipes(_ArmRecipeTestCase):
    def test_ethos_recipes_carry_their_pass_pipeline_config(self) -> None:
        # Building the partitioner before the config is materialised silently
        # drops this entry. Only the U55 subset has a non-default config.
        for recipe_type in (
            ArmRecipeType.ETHOS_U55_INT8,
            ArmRecipeType.ETHOS_U65_INT8,
        ):
            with self.subTest(recipe_type=recipe_type):
                partitioner = _first_partitioner(ExportRecipe.get_recipe(recipe_type))
                self.assertIsNotNone(
                    _compile_spec_value(partitioner, "transform_pipeline_config")
                )
        self.assertIsNone(
            _compile_spec_value(
                _first_partitioner(
                    ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U85_INT8)
                ),
                "transform_pipeline_config",
            )
        )

    def test_default_macs(self) -> None:
        cases = [
            (ArmRecipeType.ETHOS_U55_INT8, "ethos-u55-128"),
            (ArmRecipeType.ETHOS_U65_INT8, "ethos-u65-256"),
            (ArmRecipeType.ETHOS_U85_INT8, "ethos-u85-256"),
        ]
        for recipe_type, expected_target in cases:
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertEqual(recipe.name, recipe_type.value)
                self.assertEqual(_input_activation_dtype(recipe), torch.int8)
                partitioner = _first_partitioner(recipe)
                self.assertEqual(
                    _compile_spec_value(partitioner, "target"), expected_target
                )

    def test_custom_macs(self) -> None:
        cases = [
            (ArmRecipeType.ETHOS_U55_INT8, 32, "ethos-u55-32"),
            (ArmRecipeType.ETHOS_U55_INT8, 256, "ethos-u55-256"),
            (ArmRecipeType.ETHOS_U65_INT8, 512, "ethos-u65-512"),
            (ArmRecipeType.ETHOS_U85_INT8, 128, "ethos-u85-128"),
            (ArmRecipeType.ETHOS_U85_INT8, 2048, "ethos-u85-2048"),
        ]
        for recipe_type, macs, expected_target in cases:
            with self.subTest(recipe_type=recipe_type, macs=macs):
                recipe = ExportRecipe.get_recipe(recipe_type, macs=macs)
                partitioner = _first_partitioner(recipe)
                self.assertEqual(
                    _compile_spec_value(partitioner, "target"), expected_target
                )

    @unittest.skipUnless(_VELA_INSTALLED, "accelerator configs come from Vela")
    def test_invalid_macs_raises_u55(self) -> None:
        cases = [
            (ArmRecipeType.ETHOS_U55_INT8, 512),
            (ArmRecipeType.ETHOS_U65_INT8, 128),
            (ArmRecipeType.ETHOS_U85_INT8, 64),
            (ArmRecipeType.ETHOS_U55_INT8, 999),
        ]
        for recipe_type, macs in cases:
            with self.subTest(recipe_type=recipe_type, macs=macs):
                with self.assertRaises(ValueError):
                    ExportRecipe.get_recipe(recipe_type, macs=macs)

    def test_pass_through_kwargs(self) -> None:
        recipe = ExportRecipe.get_recipe(
            ArmRecipeType.ETHOS_U55_INT8,
            macs=128,
            system_config="Custom_System",
            memory_mode="Custom_Memory",
            extra_flags=["--user-flag"],
            config_ini="custom/vela.ini",
        )
        partitioner = _first_partitioner(recipe)
        flags = _compile_spec_value(partitioner, "compile_flags") or ""
        # Vela takes the last occurrence of a repeated flag, so the defaults
        # have to come first for a caller override to win.
        self.assertTrue(
            flags.startswith("--verbose-operators --verbose-cycle-estimate"),
            f"default flags must be prepended, got {flags}",
        )
        self.assertLess(flags.index("--verbose-operators"), flags.index("--user-flag"))
        self.assertIn("--system-config=Custom_System", flags)
        self.assertIn("--memory-mode=Custom_Memory", flags)
        self.assertIn("--verbose-operators", flags)
        self.assertIn("--verbose-cycle-estimate", flags)
        self.assertIn("--user-flag", flags)
        self.assertIn("--config=custom/vela.ini", flags)

    def test_default_vela_flags(self) -> None:
        # test_pass_through_kwargs supplies every kwarg, so the defaults would
        # otherwise never be built. A wrong default config path only surfaces
        # when Vela runs.
        partitioner = _first_partitioner(
            ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8)
        )
        flags = _compile_spec_value(partitioner, "compile_flags") or ""
        self.assertIn("--config=Arm/vela.ini", flags)
        self.assertIn("--verbose-operators", flags)
        self.assertIn("--verbose-cycle-estimate", flags)

    def test_documented_kwargs_do_not_warn(self) -> None:
        # Every one of these is honoured, so reporting it as ignored would be
        # a lie about what the recipe did.
        with self.assertNoLogs(_PROVIDER_LOGGER, level="WARNING"):
            ExportRecipe.get_recipe(
                ArmRecipeType.ETHOS_U55_INT8,
                macs=128,
                system_config="Custom_System",
                memory_mode="Custom_Memory",
                extra_flags=["--user-flag"],
                config_ini="custom/vela.ini",
            )

    def test_unexpected_kwarg_warns(self) -> None:
        # Flags typos like `mac=128` (instead of `macs=128`), which would
        # otherwise silently produce a default-target binary.
        with self.assertLogs(_PROVIDER_LOGGER, level="WARNING"):
            ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8, mac=128)

    def test_extra_flags_must_be_a_list(self) -> None:
        # A bare string is iterable, so it would reach Vela as one flag per
        # character instead of failing.
        with self.assertRaisesRegex(ValueError, "extra_flags must be a list"):
            ExportRecipe.get_recipe(
                ArmRecipeType.ETHOS_U55_INT8, extra_flags="--enable-debug-db"
            )
        with self.assertRaisesRegex(ValueError, "extra_flags must be a list"):
            ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8, extra_flags=[1])
        # Not iterable at all: checking the elements first raises TypeError out
        # of `all` and the caller never sees the real complaint.
        with self.assertRaisesRegex(ValueError, "extra_flags must be a list"):
            ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8, extra_flags=7)

    def test_macs_must_be_an_int(self) -> None:
        with self.assertRaisesRegex(ValueError, "macs must be an int"):
            ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8, macs="128")

    def test_program_config_matches_the_cli(self) -> None:
        # The Arm runtime has only ever been run against an inline delegate
        # payload, and quantized Arm graphs do not survive the edge verifier.
        recipe = ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8)
        assert recipe.executorch_backend_config is not None
        self.assertFalse(recipe.executorch_backend_config.extract_delegate_segments)
        assert recipe.lowering_recipe is not None
        assert recipe.lowering_recipe.edge_compile_config is not None
        self.assertFalse(recipe.lowering_recipe.edge_compile_config._check_ir_validity)

    def test_fp_recipes_run_no_post_partition_transform(self) -> None:
        # No QDQ ops to rewrite, so the extra stage must not be scheduled.
        recipe = ExportRecipe.get_recipe(ArmRecipeType.TOSA_FP)
        assert recipe.lowering_recipe is not None
        self.assertIsNone(recipe.lowering_recipe.op_backends)

    def test_recipes_do_not_share_config_objects(self) -> None:
        # _combine_recipes compares configs by value on the assumption that
        # each provider hands out a fresh one.
        first = ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8)
        second = ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8)
        assert first.lowering_recipe is not None
        assert second.lowering_recipe is not None
        self.assertIsNot(
            first.lowering_recipe.edge_compile_config,
            second.lowering_recipe.edge_compile_config,
        )
        self.assertIsNot(
            first.executorch_backend_config, second.executorch_backend_config
        )


class TestQuantizedRecipeLowering(_ArmRecipeTestCase):
    """Every quantized recipe has to rewrite the QDQ ops left outside the
    delegate, and can only do so from a stage that runs after partitioning.
    """

    QUANTIZED_RECIPES = (
        ArmRecipeType.TOSA_INT8,
        ArmRecipeType.TOSA_A16W8,
        ArmRecipeType.ETHOS_U55_INT8,
        ArmRecipeType.ETHOS_U65_INT8,
        ArmRecipeType.ETHOS_U85_INT8,
    )

    def test_replace_quant_nodes_is_wired(self) -> None:
        for recipe_type in self.QUANTIZED_RECIPES:
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                assert recipe.lowering_recipe is not None
                self.assertTrue(
                    recipe.lowering_recipe.op_backends,
                    "quantized recipes must run ReplaceQuantNodesPass",
                )

    def test_session_schedules_the_stage(self) -> None:
        session = ExportSession(
            model=_ConvReluModule(),
            example_inputs=[(torch.randn(1, 3, 8, 8),)],
            export_recipe=ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8),
        )
        stages = session._pipeline_stages
        self.assertIn(StageType.EDGE_PROGRAM_MANAGER_TRANSFORM, stages)
        self.assertGreater(
            stages.index(StageType.EDGE_PROGRAM_MANAGER_TRANSFORM),
            stages.index(StageType.TO_EDGE_TRANSFORM_AND_LOWER),
        )


def _export_program(
    recipe: ExportRecipe,
    model: torch.nn.Module,
    example_inputs: tuple,
):
    from executorch.export import export

    session = export(
        model=model,
        example_inputs=[example_inputs],
        export_recipe=recipe,
    )
    return session.get_executorch_program()


def _instruction_kinds(program) -> tuple[list, list]:
    from executorch.exir.schema import DelegateCall, KernelCall

    instructions = program.execution_plan[0].chains[0].instructions
    assert instructions is not None
    operators = program.execution_plan[0].operators
    delegate_calls = [i for i in instructions if isinstance(i.instr_args, DelegateCall)]
    kernel_op_names = [
        operators[i.instr_args.op_index].name
        for i in instructions
        if isinstance(i.instr_args, KernelCall)
    ]
    return delegate_calls, kernel_op_names


class TestTosaAOTRoundTrip(_ArmRecipeTestCase):
    """End-to-end exports through the recipe pipeline.

    VGF round-trips need a real compiler and are deferred to an FVP-bearing
    follow-up; the Ethos-U one lives in ``TestEthosUCortexMComposition``.

    """

    def test_tosa_fp_export(self) -> None:
        # FP path: no quant ops, expect full delegation (Add is supported by TOSA).
        program = _export_program(
            ExportRecipe.get_recipe(ArmRecipeType.TOSA_FP),
            _AddModule(),
            (torch.randn(2, 3), torch.randn(2, 3)),
        )
        delegates, kernels = _instruction_kinds(program)
        self.assertEqual(len(delegates), 1, "Add should produce one TOSA delegate")
        self.assertEqual(
            kernels, [], f"Expected full delegation, got kernels {kernels}"
        )

    def test_tosa_int8_export(self) -> None:
        # INT8 path: boundary quantize/dequantize remain outside the delegate
        # and ReplaceQuantNodesPass rewrites them to cortex_m::*.
        program = _export_program(
            ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8),
            _ConvReluModule(),
            (torch.randn(1, 3, 8, 8),),
        )
        delegates, kernels = _instruction_kinds(program)
        self.assertGreaterEqual(len(delegates), 1, "Conv+ReLU should delegate")
        for op_name in kernels:
            self.assertTrue(
                op_name.startswith("cortex_m::"),
                f"Non-delegate kernels must be cortex_m boundary ops; got {op_name}",
            )


class TestEthosUCortexMComposition(_ArmRecipeTestCase):
    """The NPU-plus-CPU-fallback pairing ``ExportRecipe.combine`` serves.

    Without the combining rule the two are refused: they disagree on
    ``preserve_ops``, which Cortex-M fills and Ethos-U leaves to its partitioner.

    The method name carries ``u55`` so ``test_arm_backend.sh`` routes it to the
    Vela-bearing job; the class name avoids ``tosa``, which would drag it into
    the suite that has no compiler.

    """

    @unittest.skipUnless(_VELA_INSTALLED, "Compiling for Ethos-U needs Vela")
    def test_u55_with_cortex_m_fallback_exports(self) -> None:
        from executorch.backends.cortex_m.recipes.cortex_m_recipe_types import (
            CortexMRecipeType,
        )

        combined = ExportRecipe.combine(
            [
                ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8),
                ExportRecipe.get_recipe(CortexMRecipeType.INT8),
            ]
        )
        # Cortex-M's quantizer annotates against the layout the first example
        # carries, and rejects a convolution that is not channels_last.
        program = _export_program(
            combined,
            _ConvReluModule(),
            (torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last),),
        )
        delegates, kernels = _instruction_kinds(program)
        # Conv+ReLU fits the U55 whole, so the fallback only picks up the
        # boundary quantize/dequantize pair.
        self.assertEqual(len(delegates), 1, "Conv+ReLU should be one U55 delegate")
        for op_name in kernels:
            self.assertTrue(
                op_name.startswith("cortex_m::"),
                f"Everything the U55 declined should land on Cortex-M; got {op_name}",
            )


if __name__ == "__main__":
    unittest.main()
