# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Optional

import torch

from executorch.backends.arm.recipes.arm_recipe_provider import ArmRecipeProvider
from executorch.backends.arm.recipes.arm_recipe_types import ARM_BACKEND, ArmRecipeType
from executorch.export import ExportRecipe, recipe_registry


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


def _first_partitioner(recipe: ExportRecipe) -> Any:
    assert recipe.lowering_recipe is not None
    assert recipe.lowering_recipe.partitioners
    return recipe.lowering_recipe.partitioners[0]


def _input_activation_dtype(recipe: ExportRecipe) -> Optional[torch.dtype]:
    assert recipe.quantization_recipe is not None
    assert recipe.quantization_recipe.quantizers is not None
    quantizer = recipe.quantization_recipe.quantizers[0]
    config = quantizer.global_config  # type: ignore[attr-defined]
    if config is None or config.input_activation is None:
        return None
    return config.input_activation.dtype


class TestArmRecipeRegistration(unittest.TestCase):
    def test_backend_registered(self) -> None:
        # Auto-registered via recipes/__init__.py.
        self.assertIn(ARM_BACKEND, recipe_registry.list_backends())

    def test_supported_recipes_match_enum(self) -> None:
        # Guards against the provider drifting from ArmRecipeType (e.g.,
        # adding an enum value but forgetting to wire it).
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


class _ArmRecipeBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import executorch.backends.arm.tosa.compile_spec  # noqa: F401
            import tosa_serializer  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(
                f"Arm Python deps not available, skipping: {exc}"
            ) from exc


class _EthosURecipeBaseTest(_ArmRecipeBaseTest):
    """`EthosUCompileSpec` transitively imports `arm_vela`, which is unavailable
    on `--disable-ethos-u-deps` runners.

    Without this guard, those runs would surface as ImportError instead of a
    clean skip.

    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            import executorch.backends.arm.ethosu  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(
                f"Ethos-U deps not available, skipping: {exc}"
            ) from exc


class TestTosaRecipes(_ArmRecipeBaseTest):
    def test_tosa_construction(self) -> None:
        cases = [
            (ArmRecipeType.TOSA_FP, "arm_tosa_fp", None),
            (ArmRecipeType.TOSA_INT8, "arm_tosa_int8", torch.int8),
            (ArmRecipeType.TOSA_A16W8, "arm_tosa_a16w8", torch.int16),
        ]
        for recipe_type, expected_name, expected_act_dtype in cases:
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertEqual(recipe.name, expected_name)
                self.assertIsNotNone(_first_partitioner(recipe))
                if expected_act_dtype is None:
                    self.assertIsNone(recipe.quantization_recipe)
                else:
                    self.assertEqual(
                        _input_activation_dtype(recipe), expected_act_dtype
                    )

    def test_unexpected_kwarg_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected parameters"):
            ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8, foo=1)


class TestVgfRecipes(_ArmRecipeBaseTest):
    def test_vgf_construction(self) -> None:
        cases = [
            (ArmRecipeType.VGF_FP, "arm_vgf_fp", None),
            (ArmRecipeType.VGF_INT8, "arm_vgf_int8", torch.int8),
        ]
        for recipe_type, expected_name, expected_act_dtype in cases:
            with self.subTest(recipe_type=recipe_type):
                recipe = ExportRecipe.get_recipe(recipe_type)
                self.assertEqual(recipe.name, expected_name)
                self.assertIsNotNone(_first_partitioner(recipe))
                if expected_act_dtype is None:
                    self.assertIsNone(recipe.quantization_recipe)
                else:
                    self.assertEqual(
                        _input_activation_dtype(recipe), expected_act_dtype
                    )


class TestEthosURecipes(_EthosURecipeBaseTest):
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

    def test_invalid_macs_raises(self) -> None:
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
        self.assertIn("--system-config=Custom_System", flags)
        self.assertIn("--memory-mode=Custom_Memory", flags)
        # Verbose flags must be prepended (matches aot_arm_compiler.py:479-484).
        self.assertIn("--verbose-operators", flags)
        self.assertIn("--verbose-cycle-estimate", flags)
        self.assertIn("--user-flag", flags)
        self.assertIn("--config=custom/vela.ini", flags)

    def test_unexpected_kwarg_raises(self) -> None:
        # Catches typos like `mac=128` (instead of `macs=128`) that would
        # otherwise silently produce a wrong-target binary.
        with self.assertRaisesRegex(ValueError, "unexpected parameters"):
            ExportRecipe.get_recipe(ArmRecipeType.ETHOS_U55_INT8, mac=128)


class TestTosaAOTRoundTrip(_ArmRecipeBaseTest):
    """TOSA AOT round-trips run with just ``tosa_serializer`` installed.

    Ethos-U / VGF round-trips need a real compiler and are deferred to an FVP-
    bearing follow-up.

    """

    def _export(
        self,
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

    def _instruction_kinds(self, program) -> tuple[list, list]:
        from executorch.exir.schema import DelegateCall, KernelCall

        instructions = program.execution_plan[0].chains[0].instructions
        assert instructions is not None
        operators = program.execution_plan[0].operators
        delegate_calls = [
            i for i in instructions if isinstance(i.instr_args, DelegateCall)
        ]
        kernel_op_names = [
            operators[i.instr_args.op_index].name
            for i in instructions
            if isinstance(i.instr_args, KernelCall)
        ]
        return delegate_calls, kernel_op_names

    def test_tosa_fp_export(self) -> None:
        # FP path: no quant ops, expect full delegation (Add is supported by TOSA).
        program = self._export(
            ExportRecipe.get_recipe(ArmRecipeType.TOSA_FP),
            _AddModule(),
            (torch.randn(2, 3), torch.randn(2, 3)),
        )
        delegates, kernels = self._instruction_kinds(program)
        self.assertEqual(len(delegates), 1, "Add should produce one TOSA delegate")
        self.assertEqual(
            kernels, [], f"Expected full delegation, got kernels {kernels}"
        )

    def test_tosa_int8_export(self) -> None:
        # INT8 path: boundary quantize/dequantize remain outside the delegate
        # and ReplaceQuantNodesPass rewrites them to cortex_m::* (matches
        # aot_arm_compiler.py:200-201).
        program = self._export(
            ExportRecipe.get_recipe(ArmRecipeType.TOSA_INT8),
            _ConvReluModule(),
            (torch.randn(1, 3, 8, 8),),
        )
        delegates, kernels = self._instruction_kinds(program)
        self.assertGreaterEqual(len(delegates), 1, "Conv+ReLU should delegate")
        for op_name in kernels:
            self.assertTrue(
                op_name.startswith("cortex_m::"),
                f"Non-delegate kernels must be cortex_m boundary ops; got {op_name}",
            )


if __name__ == "__main__":
    unittest.main()
